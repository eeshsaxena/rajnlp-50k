#!/usr/bin/env python3
"""
RajNLP-50K — Train all three MuRIL models on annotated corpus.

Loads the LLM-annotated (or human-annotated) corpus, converts it to
AnnotatedSentence objects, and fine-tunes:
  1. MuRILSentimentClassifier
  2. MuRILNERTagger
  3. MuRILToxicityClassifier

Uses the existing real MuRIL training scripts in models/.

Usage:
    python train_all.py --annotated-data output/llm_annotations/annotated_corpus.jsonl
    python train_all.py --annotated-data output/llm_annotations/annotated_corpus.jsonl --seed 42
    python train_all.py --task sentiment  # train only one model

Requirements: 10.1, 11.1, 12.1, 17.1, 17.3, 17.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load annotated corpus
# ---------------------------------------------------------------------------

def load_annotated_corpus(path: Path) -> tuple[list, list, list]:
    """Load annotated JSONL and return (train, val, test) AnnotatedSentence lists.

    Args:
        path: Path to annotated_corpus.jsonl

    Returns:
        Tuple of (train_set, val_set, test_set) as AnnotatedSentence lists.
    """
    from datetime import datetime, timezone
    from models.data_models import AnnotatedSentence, EntitySpan, TokenLabel

    train, val, test = [], [], []
    errors = 0

    with path.open(encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)

                # Parse collected_at
                collected_at = obj.get("collected_at", "")
                if isinstance(collected_at, str) and collected_at:
                    try:
                        collected_at = datetime.fromisoformat(
                            collected_at.replace("Z", "+00:00")
                        )
                    except Exception:
                        collected_at = datetime.now(timezone.utc)
                else:
                    collected_at = datetime.now(timezone.utc)

                annotated_at = obj.get("annotated_at", "")
                if isinstance(annotated_at, str) and annotated_at:
                    try:
                        annotated_at = datetime.fromisoformat(
                            annotated_at.replace("Z", "+00:00")
                        )
                    except Exception:
                        annotated_at = datetime.now(timezone.utc)
                else:
                    annotated_at = datetime.now(timezone.utc)

                # Parse NER spans
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
                    sentiment_annotator_labels=obj.get(
                        "sentiment_annotator_labels", ["neutral", "neutral", "neutral"]
                    ),
                    ner_spans=ner_spans,
                    ner_annotator_spans=[ner_spans, ner_spans, ner_spans],
                    toxicity_labels=obj.get("toxicity_labels", []),
                    toxicity_annotator_labels=[
                        obj.get("toxicity_labels", []),
                        obj.get("toxicity_labels", []),
                        obj.get("toxicity_labels", []),
                    ],
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

    logger.info(
        "Loaded: train=%d val=%d test=%d (errors=%d)",
        len(train), len(val), len(test), errors,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_sentiment(train_set, val_set, test_set, seed: int, output_dir: Path) -> dict:
    """Fine-tune MuRIL sentiment classifier."""
    from models.muril_sentiment_classifier import MuRILSentimentClassifier

    logger.info("=== Training Sentiment Classifier ===")
    clf = MuRILSentimentClassifier(
        checkpoint_dir=str(output_dir / "sentiment"),
    )

    start = time.time()
    log = clf.train(train_set, val_set, seed=seed)
    elapsed = time.time() - start

    logger.info(
        "Sentiment training done: best_f1=%.4f best_epoch=%d time=%.1fm",
        log.best_f1, log.best_epoch, elapsed / 60,
    )

    # Evaluate on test set
    metrics = clf.evaluate(test_set)
    logger.info("Sentiment test macro-F1: %.4f", metrics.macro_f1)
    logger.info("Per-class F1: %s", metrics.per_class_f1)

    # Save model
    save_path = str(output_dir / "sentiment" / "best_model")
    clf.save(save_path)
    logger.info("Sentiment model saved to %s", save_path)

    return {
        "task": "sentiment",
        "best_f1": log.best_f1,
        "test_macro_f1": metrics.macro_f1,
        "per_class_f1": metrics.per_class_f1,
        "training_minutes": elapsed / 60,
        "seed": seed,
        "model_path": save_path,
    }


def train_ner(train_set, val_set, test_set, seed: int, output_dir: Path) -> dict:
    """Fine-tune MuRIL NER tagger."""
    from models.muril_ner_tagger import MuRILNERTagger

    logger.info("=== Training NER Tagger ===")
    tagger = MuRILNERTagger(
        checkpoint_dir=str(output_dir / "ner"),
    )

    start = time.time()
    log = tagger.train(train_set, val_set, seed=seed)
    elapsed = time.time() - start

    logger.info(
        "NER training done: best_f1=%.4f time=%.1fm",
        log.best_f1, elapsed / 60,
    )

    metrics = tagger.evaluate(test_set)
    logger.info("NER test macro-F1: %.4f", metrics.macro_f1)
    logger.info("Per-type F1: %s", metrics.per_type_f1)

    save_path = str(output_dir / "ner" / "best_model")
    tagger.save(save_path)
    logger.info("NER model saved to %s", save_path)

    return {
        "task": "ner",
        "best_f1": log.best_f1,
        "test_macro_f1": metrics.macro_f1,
        "per_type_f1": metrics.per_type_f1,
        "training_minutes": elapsed / 60,
        "seed": seed,
        "model_path": save_path,
    }


def train_toxicity(train_set, val_set, test_set, seed: int, output_dir: Path) -> dict:
    """Fine-tune MuRIL toxicity classifier."""
    from models.muril_toxicity_classifier import MuRILToxicityClassifier

    logger.info("=== Training Toxicity Classifier ===")
    clf = MuRILToxicityClassifier(
        checkpoint_dir=str(output_dir / "toxicity"),
    )

    start = time.time()
    log = clf.train(train_set, val_set, seed=seed)
    elapsed = time.time() - start

    logger.info(
        "Toxicity training done: best_f1=%.4f time=%.1fm",
        log.best_f1, elapsed / 60,
    )

    metrics = clf.evaluate(test_set)
    logger.info("Toxicity test macro-F1: %.4f", metrics.macro_f1)
    logger.info("Per-category F1: %s", metrics.per_category_f1)

    save_path = str(output_dir / "toxicity" / "best_model")
    clf.save(save_path)
    logger.info("Toxicity model saved to %s", save_path)

    return {
        "task": "toxicity",
        "best_f1": log.best_f1,
        "test_macro_f1": metrics.macro_f1,
        "per_category_f1": metrics.per_category_f1,
        "training_minutes": elapsed / 60,
        "seed": seed,
        "model_path": save_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train all MuRIL models on annotated RajNLP-50K corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotated-data",
        default="output/llm_annotations/annotated_corpus.jsonl",
        help="Path to annotated corpus JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/trained_models",
        help="Directory to save trained models and results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--task",
        choices=["sentiment", "ner", "toxicity", "all"],
        default="all",
        help="Which model to train.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    data_path = Path(args.annotated_data)
    if not data_path.exists():
        logger.error("Annotated data not found: %s", data_path)
        logger.error("Run first: python -m annotator_tool.llm_annotate")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check PyTorch + transformers
    try:
        import torch
        from transformers import AutoTokenizer
        logger.info("PyTorch version: %s", torch.__version__)
        logger.info("CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("No CUDA GPU detected — training will use CPU (slow)")
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        logger.error("Run: pip install transformers datasets evaluate torch seqeval accelerate")
        return 1

    # Load corpus
    logger.info("Loading annotated corpus from %s...", data_path)
    train_set, val_set, test_set = load_annotated_corpus(data_path)

    if not train_set:
        logger.error("No training data found. Check annotated corpus.")
        return 1

    logger.info(
        "Dataset: train=%d val=%d test=%d",
        len(train_set), len(val_set), len(test_set),
    )

    # Train
    results = []
    tasks = ["sentiment", "ner", "toxicity"] if args.task == "all" else [args.task]

    for task in tasks:
        try:
            if task == "sentiment":
                result = train_sentiment(train_set, val_set, test_set, args.seed, output_dir)
            elif task == "ner":
                result = train_ner(train_set, val_set, test_set, args.seed, output_dir)
            elif task == "toxicity":
                result = train_toxicity(train_set, val_set, test_set, args.seed, output_dir)
            results.append(result)
        except Exception as e:
            logger.error("Training failed for %s: %s", task, e, exc_info=True)
            results.append({"task": task, "error": str(e)})

    # Save results
    results_path = output_dir / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Print summary
    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['task']:<12} ERROR: {r['error']}")
        else:
            print(
                f"  {r['task']:<12} test_macro_f1={r.get('test_macro_f1', 0):.4f}  "
                f"time={r.get('training_minutes', 0):.1f}m  "
                f"saved={r.get('model_path', 'N/A')}"
            )
    print()
    print(f"  Results saved to: {results_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
