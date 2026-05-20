#!/usr/bin/env python3
"""
RajNLP-50K — Train all three MuRIL models on annotated corpus.

Directly uses HuggingFace Trainer (same approach as test_training.py which works).
Uses refs/pr/3 revision of MuRIL to get safetensors format (avoids CVE-2025-32434).

Usage:
    py -3.12 train_all.py --annotated-data output\llm_annotations\annotated_corpus.jsonl
    py -3.12 train_all.py --task sentiment  # train only one model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MURIL_CHECKPOINT = "google/muril-base-cased"
MURIL_REVISION = "refs/pr/3"  # safetensors version, avoids CVE-2025-32434 with torch < 2.6

SENTIMENT_LABELS = ["positive", "neutral", "negative"]
SENTIMENT_LABEL2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID2LABEL = {i: l for i, l in enumerate(SENTIMENT_LABELS)}

BIO_LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
BIO_LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
BIO_ID2LABEL = {i: l for i, l in enumerate(BIO_LABELS)}

TOXICITY_CATEGORIES = ["caste_slur", "religious", "gender", "general"]


# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------

def load_annotated_corpus(path: Path):
    from datetime import datetime, timezone
    from models.data_models import AnnotatedSentence, EntitySpan, TokenLabel

    train, val, test = [], [], []
    errors = 0

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                collected_at = obj.get("collected_at", "")
                if isinstance(collected_at, str) and collected_at:
                    try:
                        collected_at = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
                    except Exception:
                        collected_at = datetime.now(timezone.utc)
                else:
                    collected_at = datetime.now(timezone.utc)

                annotated_at = obj.get("annotated_at", "")
                if isinstance(annotated_at, str) and annotated_at:
                    try:
                        annotated_at = datetime.fromisoformat(annotated_at.replace("Z", "+00:00"))
                    except Exception:
                        annotated_at = datetime.now(timezone.utc)
                else:
                    annotated_at = datetime.now(timezone.utc)

                ner_spans = []
                for span in obj.get("ner_spans", []):
                    try:
                        ner_spans.append(EntitySpan(
                            start=int(span["start"]),
                            end=int(span["end"]),
                            entity_type=span["entity_type"],
                            text=span["text"],
                        ))
                    except Exception:
                        pass

                sentence = AnnotatedSentence(
                    sentence_id=obj["sentence_id"],
                    text=obj["text"],
                    platform=obj.get("platform", "other"),
                    split=obj.get("split", "train"),
                    sentiment=obj.get("sentiment", "neutral"),
                    sentiment_annotator_labels=obj.get("sentiment_annotator_labels", ["neutral"] * 3),
                    ner_spans=ner_spans,
                    ner_annotator_spans=[ner_spans, ner_spans, ner_spans],
                    toxicity_labels=obj.get("toxicity_labels", []),
                    toxicity_annotator_labels=[obj.get("toxicity_labels", [])] * 3,
                    token_language_labels=[],
                    source_url=obj.get("source_url", ""),
                    collected_at=collected_at,
                    annotated_at=annotated_at,
                )

                split = obj.get("split", "train")
                if split == "train":
                    train.append(sentence)
                elif split == "validation":
                    val.append(sentence)
                elif split == "test":
                    test.append(sentence)
                else:
                    train.append(sentence)

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning("Parse error on line %d: %s", line_num, e)

    logger.info("Loaded: train=%d val=%d test=%d (errors=%d)", len(train), len(val), len(test), errors)
    return train, val, test


# ---------------------------------------------------------------------------
# Sentiment training
# ---------------------------------------------------------------------------

def train_sentiment(train_set, val_set, test_set, seed, output_dir):
    import numpy as np
    import torch
    from collections import Counter
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
    )
    from datasets import Dataset as HFDataset
    import evaluate

    logger.info("Loading MuRIL tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MURIL_CHECKPOINT, revision=MURIL_REVISION)

    logger.info("Loading MuRIL model for sentiment...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MURIL_CHECKPOINT, revision=MURIL_REVISION,
        num_labels=3, id2label=SENTIMENT_ID2LABEL, label2id=SENTIMENT_LABEL2ID,
    )

    # Class weights
    label_counts = Counter(s.sentiment for s in train_set)
    total = len(train_set)
    weights = torch.tensor([
        total / (3 * max(label_counts.get(l, 1), 1))
        for l in SENTIMENT_LABELS
    ], dtype=torch.float)
    logger.info("Class weights: %s", dict(zip(SENTIMENT_LABELS, weights.tolist())))

    def to_hf(sentences):
        return {
            "text": [s.text for s in sentences],
            "label": [SENTIMENT_LABEL2ID.get(s.sentiment, 1) for s in sentences],
        }

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_hf = HFDataset.from_dict(to_hf(train_set)).map(tokenize, batched=True)
    val_hf = HFDataset.from_dict(to_hf(val_set)).map(tokenize, batched=True)
    train_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": metric.compute(predictions=preds, references=labels, average="macro")["f1"]}

    import torch.nn as nn

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=weights.to(outputs.logits.device))(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    ckpt_dir = str(output_dir / "sentiment")
    args = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=10, per_device_train_batch_size=32,
        per_device_eval_batch_size=32, learning_rate=2e-5, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, seed=seed, logging_steps=100, report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_hf, eval_dataset=val_hf,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Training sentiment classifier (train=%d val=%d)...", len(train_set), len(val_set))
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    best_f1 = trainer.state.best_metric or 0.0
    logger.info("Sentiment training done: best_f1=%.4f time=%.1fm", best_f1, elapsed / 60)

    # Evaluate on test set
    test_hf = HFDataset.from_dict(to_hf(test_set)).map(tokenize, batched=True)
    test_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_result = trainer.evaluate(test_hf)
    test_f1 = test_result.get("eval_macro_f1", 0.0)
    logger.info("Sentiment test macro-F1: %.4f", test_f1)

    # Save
    save_path = str(output_dir / "sentiment" / "best_model")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Sentiment model saved to %s", save_path)

    return {"task": "sentiment", "best_f1": best_f1, "test_macro_f1": test_f1,
            "training_minutes": elapsed / 60, "seed": seed, "model_path": save_path}


# ---------------------------------------------------------------------------
# NER training
# ---------------------------------------------------------------------------

def train_ner(train_set, val_set, test_set, seed, output_dir):
    import numpy as np
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
        DataCollatorForTokenClassification,
    )
    from datasets import Dataset as HFDataset
    import evaluate

    logger.info("Loading MuRIL tokenizer for NER...")
    tokenizer = AutoTokenizer.from_pretrained(MURIL_CHECKPOINT, revision=MURIL_REVISION)

    logger.info("Loading MuRIL model for NER...")
    model = AutoModelForTokenClassification.from_pretrained(
        MURIL_CHECKPOINT, revision=MURIL_REVISION,
        num_labels=len(BIO_LABELS), id2label=BIO_ID2LABEL, label2id=BIO_LABEL2ID,
    )

    def spans_to_bio(tokens, spans, text):
        bio = ["O"] * len(tokens)
        pos = 0
        offsets = []
        for tok in tokens:
            start = text.find(tok, pos)
            if start == -1:
                start = pos
            offsets.append((start, start + len(tok)))
            pos = start + len(tok)
        for span in spans:
            covered = [i for i, (ts, te) in enumerate(offsets) if ts >= span.start and te <= span.end]
            if covered:
                bio[covered[0]] = f"B-{span.entity_type}"
                for idx in covered[1:]:
                    bio[idx] = f"I-{span.entity_type}"
        return bio

    def to_hf(sentences):
        all_tokens, all_labels = [], []
        for s in sentences:
            tokens = s.text.split()
            bio = spans_to_bio(tokens, s.ner_spans, s.text)
            all_tokens.append(tokens)
            all_labels.append([BIO_LABEL2ID[b] for b in bio])
        return {"tokens": all_tokens, "ner_tags": all_labels}

    def tokenize_and_align(batch):
        tokenized = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True,
                              padding="max_length", max_length=128)
        all_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                elif wid != prev:
                    aligned.append(labels[wid] if wid < len(labels) else -100)
                else:
                    aligned.append(-100)
                prev = wid
            all_labels.append(aligned)
        tokenized["labels"] = all_labels
        return tokenized

    train_hf = HFDataset.from_dict(to_hf(train_set)).map(tokenize_and_align, batched=True)
    val_hf = HFDataset.from_dict(to_hf(val_set)).map(tokenize_and_align, batched=True)

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_preds = [[BIO_ID2LABEL[p] for p, l in zip(pr, lr) if l != -100]
                      for pr, lr in zip(preds, labels)]
        true_labels = [[BIO_ID2LABEL[l] for l in lr if l != -100] for lr in labels]
        result = seqeval.compute(predictions=true_preds, references=true_labels)
        return {"span_f1": result["overall_f1"]}

    ckpt_dir = str(output_dir / "ner")
    args = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=5, per_device_train_batch_size=16,
        per_device_eval_batch_size=16, learning_rate=3e-5, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="span_f1",
        greater_is_better=True, seed=seed, logging_steps=100, report_to="none",
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_hf, eval_dataset=val_hf,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Training NER tagger (train=%d val=%d)...", len(train_set), len(val_set))
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    best_f1 = trainer.state.best_metric or 0.0
    logger.info("NER training done: best_f1=%.4f time=%.1fm", best_f1, elapsed / 60)

    test_hf = HFDataset.from_dict(to_hf(test_set)).map(tokenize_and_align, batched=True)
    test_result = trainer.evaluate(test_hf)
    test_f1 = test_result.get("eval_span_f1", 0.0)
    logger.info("NER test span-F1: %.4f", test_f1)

    save_path = str(output_dir / "ner" / "best_model")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("NER model saved to %s", save_path)

    return {"task": "ner", "best_f1": best_f1, "test_macro_f1": test_f1,
            "training_minutes": elapsed / 60, "seed": seed, "model_path": save_path}


# ---------------------------------------------------------------------------
# Toxicity training
# ---------------------------------------------------------------------------

def train_toxicity(train_set, val_set, test_set, seed, output_dir):
    import numpy as np
    import torch
    import torch.nn as nn
    from collections import Counter
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )
    from datasets import Dataset as HFDataset

    logger.info("Loading MuRIL tokenizer for toxicity...")
    tokenizer = AutoTokenizer.from_pretrained(MURIL_CHECKPOINT, revision=MURIL_REVISION)

    logger.info("Loading MuRIL model for toxicity...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MURIL_CHECKPOINT, revision=MURIL_REVISION,
        num_labels=4, problem_type="multi_label_classification",
    )

    # Per-category class weights
    total = len(train_set)
    weights = torch.tensor([
        total / (2 * max(sum(1 for s in train_set if cat in s.toxicity_labels), 1))
        for cat in TOXICITY_CATEGORIES
    ], dtype=torch.float)
    logger.info("Toxicity class weights: %s", dict(zip(TOXICITY_CATEGORIES, weights.tolist())))

    def to_hf(sentences):
        return {
            "text": [s.text for s in sentences],
            "labels": [[1.0 if cat in s.toxicity_labels else 0.0 for cat in TOXICITY_CATEGORIES]
                       for s in sentences],
        }

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_hf = HFDataset.from_dict(to_hf(train_set)).map(tokenize, batched=True)
    val_hf = HFDataset.from_dict(to_hf(val_set)).map(tokenize, batched=True)
    train_hf.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_hf.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        per_cat = {}
        for i, cat in enumerate(TOXICITY_CATEGORIES):
            tp = int(np.sum((preds[:, i] == 1) & (labels[:, i] == 1)))
            fp = int(np.sum((preds[:, i] == 1) & (labels[:, i] == 0)))
            fn = int(np.sum((preds[:, i] == 0) & (labels[:, i] == 1)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_cat[cat] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"macro_f1": sum(per_cat.values()) / 4, **{f"f1_{k}": v for k, v in per_cat.items()}}

    class WeightedBCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            loss = nn.BCEWithLogitsLoss(pos_weight=weights.to(outputs.logits.device))(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    ckpt_dir = str(output_dir / "toxicity")
    args = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=10, per_device_train_batch_size=16,
        per_device_eval_batch_size=16, learning_rate=2e-5, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, seed=seed, logging_steps=100, report_to="none",
    )

    trainer = WeightedBCETrainer(
        model=model, args=args, train_dataset=train_hf, eval_dataset=val_hf,
        compute_metrics=compute_metrics,
    )

    logger.info("Training toxicity classifier (train=%d val=%d)...", len(train_set), len(val_set))
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    best_f1 = trainer.state.best_metric or 0.0
    logger.info("Toxicity training done: best_f1=%.4f time=%.1fm", best_f1, elapsed / 60)

    test_hf = HFDataset.from_dict(to_hf(test_set)).map(tokenize, batched=True)
    test_hf.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_result = trainer.evaluate(test_hf)
    test_f1 = test_result.get("eval_macro_f1", 0.0)
    logger.info("Toxicity test macro-F1: %.4f", test_f1)

    save_path = str(output_dir / "toxicity" / "best_model")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Toxicity model saved to %s", save_path)

    return {"task": "toxicity", "best_f1": best_f1, "test_macro_f1": test_f1,
            "training_minutes": elapsed / 60, "seed": seed, "model_path": save_path}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train MuRIL models on RajNLP-50K.")
    parser.add_argument("--annotated-data", default="output/llm_annotations/annotated_corpus.jsonl")
    parser.add_argument("--output-dir", default="output/trained_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["sentiment", "ner", "toxicity", "all"], default="all")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    import torch
    logger.info("PyTorch: %s  CUDA: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    data_path = Path(args.annotated_data)
    if not data_path.exists():
        logger.error("Annotated data not found: %s", data_path)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus...")
    train_set, val_set, test_set = load_annotated_corpus(data_path)
    logger.info("Dataset: train=%d val=%d test=%d", len(train_set), len(val_set), len(test_set))

    from models.reproducibility import set_all_seeds
    set_all_seeds(args.seed)

    results = []
    tasks = ["sentiment", "ner", "toxicity"] if args.task == "all" else [args.task]

    for task in tasks:
        logger.info("=" * 50)
        logger.info("Starting task: %s", task)
        logger.info("=" * 50)
        try:
            if task == "sentiment":
                result = train_sentiment(train_set, val_set, test_set, args.seed, output_dir)
            elif task == "ner":
                result = train_ner(train_set, val_set, test_set, args.seed, output_dir)
            elif task == "toxicity":
                result = train_toxicity(train_set, val_set, test_set, args.seed, output_dir)
            results.append(result)
            logger.info("Task %s complete: test_f1=%.4f", task, result.get("test_macro_f1", 0))
        except Exception as e:
            traceback.print_exc()
            logger.error("Task %s FAILED: %s", task, e)
            results.append({"task": task, "error": str(e)})

    results_path = output_dir / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['task']:<12} ERROR: {r['error']}")
        else:
            print(f"  {r['task']:<12} test_f1={r.get('test_macro_f1', 0):.4f}  "
                  f"time={r.get('training_minutes', 0):.1f}m  saved={r.get('model_path', 'N/A')}")
    print(f"\n  Results: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
