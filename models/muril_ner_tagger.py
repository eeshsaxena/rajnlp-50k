"""
Real MuRIL-based NER Tagger using HuggingFace Transformers.

Replaces the heuristic stub with a proper fine-tuned MuRIL token classifier.

Usage:
    from models.muril_ner_tagger import MuRILNERTagger

    tagger = MuRILNERTagger()
    log = tagger.train(train_set, val_set, seed=42)
    metrics = tagger.evaluate(test_set)
    spans = tagger.tag("Gehlot ने Jaipur में BJP की बैठक की")

Requirements: 11.1, 11.2, 11.5, 17.1, 17.3
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from models.data_models import (
    AnnotatedSentence,
    EntitySpan,
    NERMetrics,
    NERPrediction,
    TrainingLog,
)
from models.reproducibility import set_all_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MURIL_CHECKPOINT = "google/muril-base-cased"

# BIO label set
BIO_LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(BIO_LABELS)}
ENTITY_TYPES = ["PER", "LOC", "ORG"]


class MuRILNERTagger:
    """Fine-tuned MuRIL model for BIO NER tagging.

    Requirements: 11.1, 11.2, 11.5, 17.1, 17.3
    """

    def __init__(
        self,
        checkpoint: str = MURIL_CHECKPOINT,
        checkpoint_dir: str = "checkpoints/ner",
    ) -> None:
        self.checkpoint = checkpoint
        self.checkpoint_dir = Path(checkpoint_dir)
        self._model = None
        self._tokenizer = None

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        return self._tokenizer

    def _load_model(self):
        from transformers import AutoModelForTokenClassification
        if self._model is None:
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.checkpoint,
                num_labels=len(BIO_LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        return self._model

    def _spans_to_bio(
        self, tokens: list[str], spans: list[EntitySpan], text: str
    ) -> list[str]:
        """Convert EntitySpan list to BIO tag sequence."""
        bio = ["O"] * len(tokens)
        pos = 0
        token_offsets = []
        for tok in tokens:
            start = text.find(tok, pos)
            if start == -1:
                start = pos
            end = start + len(tok)
            token_offsets.append((start, end))
            pos = end

        for span in spans:
            covered = [
                i for i, (ts, te) in enumerate(token_offsets)
                if ts >= span.start and te <= span.end
            ]
            if not covered:
                continue
            bio[covered[0]] = f"B-{span.entity_type}"
            for idx in covered[1:]:
                bio[idx] = f"I-{span.entity_type}"
        return bio

    def train(
        self,
        train_set: list[AnnotatedSentence],
        val_set: list[AnnotatedSentence],
        seed: int = 42,
        max_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        patience: int = 3,
    ) -> TrainingLog:
        """Fine-tune MuRIL for NER.

        Args:
            train_set: Training partition.
            val_set: Validation partition.
            seed: Random seed (Requirement 17.1).
            max_epochs: Maximum epochs (default 5).
            batch_size: Batch size (default 16).
            learning_rate: AdamW learning rate (default 3e-5).
            patience: Early stopping patience (default 3).

        Returns:
            TrainingLog with best epoch and span-F1.
        """
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForTokenClassification,
                TrainingArguments,
                Trainer,
                EarlyStoppingCallback,
                DataCollatorForTokenClassification,
            )
            from datasets import Dataset as HFDataset
            import evaluate
        except ImportError as e:
            raise ImportError(
                f"Real training requires: pip install transformers datasets evaluate torch seqeval\n{e}"
            )

        set_all_seeds(seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        seqeval = evaluate.load("seqeval")

        def _to_hf_dict(sentences: list[AnnotatedSentence]) -> dict:
            all_tokens, all_labels = [], []
            for s in sentences:
                tokens = s.text.split()
                bio = self._spans_to_bio(tokens, s.ner_spans, s.text)
                all_tokens.append(tokens)
                all_labels.append([LABEL2ID[b] for b in bio])
            return {"tokens": all_tokens, "ner_tags": all_labels}

        train_hf = HFDataset.from_dict(_to_hf_dict(train_set))
        val_hf = HFDataset.from_dict(_to_hf_dict(val_set))

        def _tokenize_and_align(batch):
            tokenized = tokenizer(
                batch["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
                max_length=128,
            )
            all_labels = []
            for i, labels in enumerate(batch["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                aligned = []
                prev_word_id = None
                for word_id in word_ids:
                    if word_id is None:
                        aligned.append(-100)
                    elif word_id != prev_word_id:
                        aligned.append(labels[word_id] if word_id < len(labels) else -100)
                    else:
                        aligned.append(-100)
                    prev_word_id = word_id
                all_labels.append(aligned)
            tokenized["labels"] = all_labels
            return tokenized

        train_hf = train_hf.map(_tokenize_and_align, batched=True)
        val_hf = val_hf.map(_tokenize_and_align, batched=True)

        def _compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            true_preds = [
                [ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100]
                for pred_row, label_row in zip(preds, labels)
            ]
            true_labels = [
                [ID2LABEL[l] for l in label_row if l != -100]
                for label_row in labels
            ]
            result = seqeval.compute(predictions=true_preds, references=true_labels)
            return {"span_f1": result["overall_f1"]}

        data_collator = DataCollatorForTokenClassification(tokenizer)

        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            num_train_epochs=max_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="span_f1",
            greater_is_better=True,
            seed=seed,
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        logger.info("Starting MuRIL NER fine-tuning (seed=%d)", seed)
        trainer.train()

        best_f1 = trainer.state.best_metric or 0.0
        total_epochs = int(trainer.state.epoch)
        self._model = trainer.model

        return TrainingLog(
            best_epoch=total_epochs - 1,
            best_f1=best_f1,
            total_epochs_run=total_epochs,
            seed=seed,
            class_weights={et: 1.0 for et in ENTITY_TYPES},
        )

    def tag(self, sentence: str) -> list[EntitySpan]:
        """Tag named entities in a sentence.

        Args:
            sentence: Input sentence string.

        Returns:
            List of EntitySpan objects.
        """
        import torch

        tokenizer = self._load_tokenizer()
        model = self._load_model()
        model.eval()

        tokens = sentence.split()
        if not tokens:
            return []

        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

        word_ids = inputs.word_ids(batch_index=0)
        token_preds = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in token_preds:
                token_preds[word_id] = ID2LABEL[preds[idx]]

        bio_tags = [token_preds.get(i, "O") for i in range(len(tokens))]
        return _bio_to_spans(tokens, bio_tags, sentence)

    def tag_with_bio(self, sentence: str) -> NERPrediction:
        """Tag and return both spans and BIO sequence."""
        tokens = sentence.split()
        spans = self.tag(sentence)
        from models.ner_tagger import spans_to_bio_tags
        bio_tags = spans_to_bio_tags(tokens, spans, sentence)
        return NERPrediction(spans=spans, bio_tags=bio_tags)

    def evaluate(self, test_set: list[AnnotatedSentence]) -> NERMetrics:
        """Evaluate on test set using seqeval span-level metrics."""
        from seqeval.metrics import (
            classification_report,
            f1_score,
        )

        if not test_set:
            return NERMetrics(
                macro_f1=0.0,
                per_type_precision={et: 0.0 for et in ENTITY_TYPES},
                per_type_recall={et: 0.0 for et in ENTITY_TYPES},
                per_type_f1={et: 0.0 for et in ENTITY_TYPES},
            )

        gold_seqs, pred_seqs = [], []
        for s in test_set:
            tokens = s.text.split()
            if not tokens:
                continue
            gold_bio = self._spans_to_bio(tokens, s.ner_spans, s.text)
            pred_spans = self.tag(s.text)
            pred_bio = self._spans_to_bio(tokens, pred_spans, s.text)
            gold_seqs.append(gold_bio)
            pred_seqs.append(pred_bio)

        if not gold_seqs:
            return NERMetrics(
                macro_f1=0.0,
                per_type_precision={et: 0.0 for et in ENTITY_TYPES},
                per_type_recall={et: 0.0 for et in ENTITY_TYPES},
                per_type_f1={et: 0.0 for et in ENTITY_TYPES},
            )

        macro_f1 = float(f1_score(gold_seqs, pred_seqs, average="macro", zero_division=0))
        report = classification_report(gold_seqs, pred_seqs, output_dict=True, zero_division=0)

        per_type_precision, per_type_recall, per_type_f1 = {}, {}, {}
        for et in ENTITY_TYPES:
            if et in report:
                per_type_precision[et] = float(report[et]["precision"])
                per_type_recall[et] = float(report[et]["recall"])
                per_type_f1[et] = float(report[et]["f1-score"])
            else:
                per_type_precision[et] = 0.0
                per_type_recall[et] = 0.0
                per_type_f1[et] = 0.0

        return NERMetrics(
            macro_f1=macro_f1,
            per_type_precision=per_type_precision,
            per_type_recall=per_type_recall,
            per_type_f1=per_type_f1,
        )

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No model to save — train first.")
        self._model.save_pretrained(path)
        self._load_tokenizer().save_pretrained(path)

    def load(self, path: str) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self._model = AutoModelForTokenClassification.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)


def _bio_to_spans(tokens: list[str], bio_tags: list[str], sentence: str) -> list[EntitySpan]:
    """Convert BIO tag sequence back to EntitySpan objects."""
    spans = []
    pos = 0
    token_offsets = []
    for tok in tokens:
        start = sentence.find(tok, pos)
        if start == -1:
            start = pos
        end = start + len(tok)
        token_offsets.append((start, end))
        pos = end

    i = 0
    while i < len(bio_tags):
        tag = bio_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            span_start = token_offsets[i][0]
            span_end = token_offsets[i][1]
            j = i + 1
            while j < len(bio_tags) and bio_tags[j] == f"I-{etype}":
                span_end = token_offsets[j][1]
                j += 1
            spans.append(EntitySpan(
                start=span_start,
                end=span_end,
                entity_type=etype,
                text=sentence[span_start:span_end],
            ))
            i = j
        else:
            i += 1
    return spans
