"""
Real MuRIL-based Sentiment Classifier using HuggingFace Transformers.

Replaces the heuristic stub with a proper fine-tuned MuRIL model.

Usage:
    from models.muril_sentiment_classifier import MuRILSentimentClassifier

    clf = MuRILSentimentClassifier()
    log = clf.train(train_set, val_set, seed=42)
    metrics = clf.evaluate(test_set)
    prediction = clf.predict("म्हारो राजस्थान घणो सुंदर है")

Requirements: 10.1, 10.2, 10.5, 17.1, 17.3
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from models.data_models import (
    AnnotatedSentence,
    ClassificationMetrics,
    SentimentPrediction,
    TrainingLog,
)
from models.reproducibility import set_all_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MURIL_CHECKPOINT = "google/muril-base-cased"
SENTIMENT_LABELS = ["positive", "neutral", "negative"]
LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(SENTIMENT_LABELS)}

# ---------------------------------------------------------------------------
# MuRIL Sentiment Classifier
# ---------------------------------------------------------------------------


class MuRILSentimentClassifier:
    """Fine-tuned MuRIL model for 3-class sentiment classification.

    Initializes from the pre-trained MuRIL checkpoint and fine-tunes on
    the RajNLP-50K training partition.

    Requirements: 10.1, 10.2, 10.5, 17.1, 17.3
    """

    def __init__(
        self,
        checkpoint: str = MURIL_CHECKPOINT,
        checkpoint_dir: str = "checkpoints/sentiment",
    ) -> None:
        self.checkpoint = checkpoint
        self.checkpoint_dir = Path(checkpoint_dir)
        self._model = None
        self._tokenizer = None

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        if self._tokenizer is None:
            logger.info("Loading tokenizer from %s", self.checkpoint)
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        return self._tokenizer

    def _load_model(self, num_labels: int = 3):
        from transformers import AutoModelForSequenceClassification
        if self._model is None:
            logger.info("Loading model from %s", self.checkpoint)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint,
                num_labels=num_labels,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        return self._model

    def train(
        self,
        train_set: list[AnnotatedSentence],
        val_set: list[AnnotatedSentence],
        seed: int = 42,
        max_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        patience: int = 3,
    ) -> TrainingLog:
        """Fine-tune MuRIL on the training set with early stopping.

        Args:
            train_set: Training partition of annotated sentences.
            val_set: Validation partition of annotated sentences.
            seed: Random seed for reproducibility (Requirement 17.1).
            max_epochs: Maximum training epochs (default 10).
            batch_size: Training batch size (default 32).
            learning_rate: AdamW learning rate (default 2e-5).
            patience: Early stopping patience on validation macro-F1 (default 3).

        Returns:
            TrainingLog with best epoch, best F1, and class weights.
        """
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
                EarlyStoppingCallback,
            )
            from datasets import Dataset as HFDataset
            import evaluate
        except ImportError as e:
            raise ImportError(
                f"Real training requires: pip install transformers datasets evaluate torch\n{e}"
            )

        set_all_seeds(seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        # Compute class weights
        train_labels = [s.sentiment for s in train_set]
        class_weights = _compute_class_weights(train_labels)
        logger.info("Class weights: %s", class_weights)

        # Build HuggingFace datasets
        def _to_hf_dict(sentences: list[AnnotatedSentence]) -> dict:
            return {
                "text": [s.text for s in sentences],
                "label": [LABEL2ID[s.sentiment] for s in sentences],
            }

        train_hf = HFDataset.from_dict(_to_hf_dict(train_set))
        val_hf = HFDataset.from_dict(_to_hf_dict(val_set))

        def _tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        train_hf = train_hf.map(_tokenize, batched=True)
        val_hf = val_hf.map(_tokenize, batched=True)
        train_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # Class-weighted loss trainer
        weight_tensor = torch.tensor(
            [class_weights.get(lbl, 1.0) for lbl in SENTIMENT_LABELS],
            dtype=torch.float,
        )

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                import torch.nn as nn
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fn = nn.CrossEntropyLoss(
                    weight=weight_tensor.to(logits.device)
                )
                loss = loss_fn(logits, labels)
                return (loss, outputs) if return_outputs else loss

        # Load seqeval-style metric
        metric = evaluate.load("f1")

        def _compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            result = metric.compute(
                predictions=preds,
                references=labels,
                average="macro",
            )
            return {"macro_f1": result["f1"]}

        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            num_train_epochs=max_epochs,
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

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        logger.info("Starting MuRIL fine-tuning (seed=%d, epochs=%d)", seed, max_epochs)
        train_result = trainer.train()

        best_f1 = trainer.state.best_metric or 0.0
        best_epoch = int(trainer.state.best_model_checkpoint.split("-")[-1]) if trainer.state.best_model_checkpoint else 0
        total_epochs = int(trainer.state.epoch)

        logger.info(
            "Training complete: best_f1=%.4f best_epoch=%d total_epochs=%d",
            best_f1, best_epoch, total_epochs,
        )

        self._model = trainer.model
        return TrainingLog(
            best_epoch=best_epoch,
            best_f1=best_f1,
            total_epochs_run=total_epochs,
            seed=seed,
            class_weights=class_weights,
        )

    def predict(self, sentence: str) -> SentimentPrediction:
        """Predict sentiment for a single sentence.

        Args:
            sentence: Input sentence string.

        Returns:
            SentimentPrediction with label, confidence, and per-class scores.
        """
        import torch
        import torch.nn.functional as F

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
            probs = F.softmax(outputs.logits, dim=-1).squeeze()

        scores = {SENTIMENT_LABELS[i]: float(probs[i]) for i in range(len(SENTIMENT_LABELS))}
        label = max(scores, key=scores.__getitem__)
        return SentimentPrediction(
            label=label,
            confidence=scores[label],
            per_class_scores=scores,
        )

    def evaluate(self, test_set: list[AnnotatedSentence]) -> ClassificationMetrics:
        """Evaluate on a test set and return ClassificationMetrics.

        Args:
            test_set: List of annotated sentences with gold sentiment labels.

        Returns:
            ClassificationMetrics with macro-F1 and per-class metrics.
        """
        if not test_set:
            return ClassificationMetrics(
                macro_f1=0.0,
                per_class_precision={l: 0.0 for l in SENTIMENT_LABELS},
                per_class_recall={l: 0.0 for l in SENTIMENT_LABELS},
                per_class_f1={l: 0.0 for l in SENTIMENT_LABELS},
            )

        gold = [s.sentiment for s in test_set]
        pred = [self.predict(s.text).label for s in test_set]
        return _compute_classification_metrics(gold, pred)

    def save(self, path: str) -> None:
        """Save the fine-tuned model and tokenizer to disk."""
        if self._model is None:
            raise RuntimeError("No model to save — train first.")
        self._model.save_pretrained(path)
        self._load_tokenizer().save_pretrained(path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a fine-tuned model from disk."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info("Model loaded from %s", path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_class_weights(labels: list[str]) -> dict[str, float]:
    """Compute class weights inversely proportional to class frequency."""
    if not labels:
        return {l: 1.0 for l in SENTIMENT_LABELS}
    total = len(labels)
    counts = Counter(labels)
    n_classes = len(counts)
    return {
        cls: total / (n_classes * counts[cls]) if counts[cls] > 0 else 1.0
        for cls in SENTIMENT_LABELS
    }


def _compute_classification_metrics(
    gold: list[str], pred: list[str]
) -> ClassificationMetrics:
    """Compute per-class and macro-averaged F1."""
    from collections import defaultdict
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    per_class_precision, per_class_recall, per_class_f1 = {}, {}, {}
    for cls in SENTIMENT_LABELS:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_precision[cls] = prec
        per_class_recall[cls] = rec
        per_class_f1[cls] = f1
    macro_f1 = sum(per_class_f1.values()) / len(SENTIMENT_LABELS)
    return ClassificationMetrics(
        macro_f1=macro_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
    )
