"""
Real MuRIL-based Toxicity Classifier using HuggingFace Transformers.

Multi-label classification across 4 toxicity categories:
  caste_slur, religious, gender, general

Usage:
    from models.muril_toxicity_classifier import MuRILToxicityClassifier

    clf = MuRILToxicityClassifier()
    log = clf.train(train_set, val_set, seed=42)
    metrics = clf.evaluate(test_set)
    prediction = clf.predict("some sentence")

Requirements: 12.1, 12.2, 12.3, 12.6, 17.1, 17.3
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

from models.data_models import (
    AnnotatedSentence,
    MultiLabelMetrics,
    ToxicityPrediction,
    TrainingLog,
)
from models.reproducibility import set_all_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MURIL_CHECKPOINT = "google/muril-base-cased"
TOXICITY_CATEGORIES = ["caste_slur", "religious", "gender", "general"]
NUM_LABELS = len(TOXICITY_CATEGORIES)
SIGMOID_THRESHOLD = 0.5


class MuRILToxicityClassifier:
    """Fine-tuned MuRIL model for multi-label toxicity classification.

    Requirements: 12.1, 12.2, 12.3, 12.6, 17.1, 17.3
    """

    def __init__(
        self,
        checkpoint: str = MURIL_CHECKPOINT,
        checkpoint_dir: str = "checkpoints/toxicity",
        sigmoid_threshold: float = SIGMOID_THRESHOLD,
    ) -> None:
        self.checkpoint = checkpoint
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sigmoid_threshold = sigmoid_threshold
        self._model = None
        self._tokenizer = None
        self._oversampling_active = False

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        return self._tokenizer

    def _load_model(self):
        from transformers import AutoModelForSequenceClassification
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint,
                num_labels=NUM_LABELS,
                problem_type="multi_label_classification",
            )
        return self._model

    def train(
        self,
        train_set: list[AnnotatedSentence],
        val_set: list[AnnotatedSentence],
        seed: int = 42,
        max_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        oversample_check_epoch: int = 5,
    ) -> TrainingLog:
        """Fine-tune MuRIL for multi-label toxicity classification.

        Applies class-weighted binary cross-entropy loss. If per-category
        minority-class F1 remains below 0.60 after oversample_check_epoch
        epochs, activates random oversampling (minority:majority = 1:3).

        Args:
            train_set: Training partition.
            val_set: Validation partition.
            seed: Random seed (Requirement 17.1).
            max_epochs: Maximum epochs (default 10).
            batch_size: Batch size (default 16).
            learning_rate: AdamW learning rate (default 2e-5).
            oversample_check_epoch: Epoch at which to check oversampling fallback.

        Returns:
            TrainingLog with best epoch and macro-F1.
        """
        try:
            import torch
            import torch.nn as nn
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
            )
            from datasets import Dataset as HFDataset
        except ImportError as e:
            raise ImportError(
                f"Real training requires: pip install transformers datasets torch\n{e}"
            )

        set_all_seeds(seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        # Compute per-category class weights
        class_weights = _compute_toxicity_weights(train_set)
        weight_tensor = torch.tensor(
            [class_weights[cat] for cat in TOXICITY_CATEGORIES],
            dtype=torch.float,
        )
        logger.info("Toxicity class weights: %s", class_weights)

        def _to_hf_dict(sentences: list[AnnotatedSentence]) -> dict:
            texts, labels = [], []
            for s in sentences:
                texts.append(s.text)
                label_vec = [
                    1.0 if cat in s.toxicity_labels else 0.0
                    for cat in TOXICITY_CATEGORIES
                ]
                labels.append(label_vec)
            return {"text": texts, "labels": labels}

        def _tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        train_data = train_set
        best_f1 = 0.0
        best_epoch = 0
        self._oversampling_active = False

        for phase_start in [0, oversample_check_epoch]:
            phase_end = oversample_check_epoch if phase_start == 0 else max_epochs
            if phase_start >= max_epochs:
                break

            train_hf = HFDataset.from_dict(_to_hf_dict(train_data))
            val_hf = HFDataset.from_dict(_to_hf_dict(val_set))
            train_hf = train_hf.map(_tokenize, batched=True)
            val_hf = val_hf.map(_tokenize, batched=True)
            train_hf.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            val_hf.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

            class WeightedBCETrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels").float()
                    outputs = model(**inputs)
                    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight_tensor.to(outputs.logits.device))
                    loss = loss_fn(outputs.logits, labels)
                    return (loss, outputs) if return_outputs else loss

            def _compute_metrics(eval_pred):
                logits, labels = eval_pred
                probs = 1 / (1 + np.exp(-logits))
                preds = (probs >= self.sigmoid_threshold).astype(int)
                per_cat_f1 = {}
                for i, cat in enumerate(TOXICITY_CATEGORIES):
                    tp = int(np.sum((preds[:, i] == 1) & (labels[:, i] == 1)))
                    fp = int(np.sum((preds[:, i] == 1) & (labels[:, i] == 0)))
                    fn = int(np.sum((preds[:, i] == 0) & (labels[:, i] == 1)))
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    per_cat_f1[cat] = f1
                macro = sum(per_cat_f1.values()) / len(TOXICITY_CATEGORIES)
                return {"macro_f1": macro, **{f"f1_{k}": v for k, v in per_cat_f1.items()}}

            training_args = TrainingArguments(
                output_dir=str(self.checkpoint_dir),
                num_train_epochs=phase_end - phase_start,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
                seed=seed,
                logging_steps=50,
                report_to="none",
            )

            trainer = WeightedBCETrainer(
                model=model,
                args=training_args,
                train_dataset=train_hf,
                eval_dataset=val_hf,
                compute_metrics=_compute_metrics,
            )

            trainer.train()
            phase_best_f1 = trainer.state.best_metric or 0.0
            if phase_best_f1 > best_f1:
                best_f1 = phase_best_f1
                best_epoch = int(trainer.state.epoch)
            model = trainer.model

            # Check oversampling fallback after first phase
            if phase_start == 0:
                eval_result = trainer.evaluate()
                needs_oversample = any(
                    eval_result.get(f"eval_f1_{cat}", 1.0) < 0.60
                    for cat in TOXICITY_CATEGORIES
                )
                if needs_oversample:
                    logger.warning(
                        "Oversampling fallback triggered after epoch %d: "
                        "per-category F1 below 0.60. Activating 1:3 oversampling.",
                        oversample_check_epoch,
                    )
                    self._oversampling_active = True
                    train_data = _oversample(train_set, ratio=3)
                    logger.info(
                        "Oversampled training set: %d → %d sentences",
                        len(train_set), len(train_data),
                    )
                else:
                    logger.info("Oversampling not needed — all per-category F1 >= 0.60")
                    break  # No need for second phase

        self._model = model
        return TrainingLog(
            best_epoch=best_epoch,
            best_f1=best_f1,
            total_epochs_run=max_epochs,
            seed=seed,
            class_weights=class_weights,
        )

    def predict(self, sentence: str) -> ToxicityPrediction:
        """Predict toxicity labels for a sentence.

        Args:
            sentence: Input sentence string.

        Returns:
            ToxicityPrediction with labels and per-category sigmoid scores.
        """
        import torch

        tokenizer = self._load_tokenizer()
        model = self._load_model()
        model.eval()

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze()

        scores = {TOXICITY_CATEGORIES[i]: float(probs[i]) for i in range(NUM_LABELS)}
        labels = [cat for cat, score in scores.items() if score >= self.sigmoid_threshold]
        return ToxicityPrediction(labels=labels, per_category_scores=scores)

    def evaluate(self, test_set: list[AnnotatedSentence]) -> MultiLabelMetrics:
        """Evaluate on test set and return MultiLabelMetrics."""
        if not test_set:
            return MultiLabelMetrics(
                macro_f1=0.0,
                per_category_precision={cat: 0.0 for cat in TOXICITY_CATEGORIES},
                per_category_recall={cat: 0.0 for cat in TOXICITY_CATEGORIES},
                per_category_f1={cat: 0.0 for cat in TOXICITY_CATEGORIES},
            )

        from collections import defaultdict
        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)

        for s in test_set:
            gold = set(s.toxicity_labels)
            pred = set(self.predict(s.text).labels)
            for cat in TOXICITY_CATEGORIES:
                if cat in gold and cat in pred:
                    tp[cat] += 1
                elif cat not in gold and cat in pred:
                    fp[cat] += 1
                elif cat in gold and cat not in pred:
                    fn[cat] += 1

        per_category_precision, per_category_recall, per_category_f1 = {}, {}, {}
        for cat in TOXICITY_CATEGORIES:
            prec = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0.0
            rec = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_category_precision[cat] = prec
            per_category_recall[cat] = rec
            per_category_f1[cat] = f1

        macro_f1 = sum(per_category_f1.values()) / len(TOXICITY_CATEGORIES)
        return MultiLabelMetrics(
            macro_f1=macro_f1,
            per_category_precision=per_category_precision,
            per_category_recall=per_category_recall,
            per_category_f1=per_category_f1,
        )

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No model to save — train first.")
        self._model.save_pretrained(path)
        self._load_tokenizer().save_pretrained(path)

    def load(self, path: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)


def _compute_toxicity_weights(sentences: list[AnnotatedSentence]) -> dict[str, float]:
    """Compute per-category class weights inversely proportional to positive frequency."""
    if not sentences:
        return {cat: 1.0 for cat in TOXICITY_CATEGORIES}
    total = len(sentences)
    weights = {}
    for cat in TOXICITY_CATEGORIES:
        count = sum(1 for s in sentences if cat in s.toxicity_labels)
        weights[cat] = total / (2 * count) if count > 0 else 1.0
    return weights


def _oversample(
    sentences: list[AnnotatedSentence], ratio: int = 3
) -> list[AnnotatedSentence]:
    """Oversample minority-class (toxic) examples to minority:majority = 1:ratio."""
    toxic = [s for s in sentences if s.toxicity_labels]
    clean = [s for s in sentences if not s.toxicity_labels]
    if not toxic:
        return sentences
    target_toxic = len(clean) // ratio
    if len(toxic) >= target_toxic:
        return sentences
    import random
    extra = random.choices(toxic, k=target_toxic - len(toxic))
    return sentences + extra
