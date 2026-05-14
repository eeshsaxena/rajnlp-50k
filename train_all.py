#!/usr/bin/env python3
"""
Train all three MuRIL models on the annotated RajNLP-50K corpus.

Usage:
    python train_all.py --seed 42 --data-dir output/annotated --output-dir checkpoints

This script:
1. Loads the annotated corpus from a JSONL file
2. Trains SentimentClassifier, NERTagger, and ToxicityClassifier
3. Saves checkpoints and generates model cards
4. Optionally publishes to HuggingFace

Requirements: 17.1, 17.3, 17.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from models.reproducibility import set_all_seeds

logger = logging.getLogger(__name__)


def load_corpus(data_dir: str):
    """Load the annotated corpus from a JSONL file."""
    from corpus_builder.serialization import deserialize
    jsonl_path = Path(data_dir) / "corpus.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {jsonl_path}. "
            "Run run_pipeline.py first to generate the corpus."
        )
    logger.info("Loading corpus from %s", jsonl_path)
    sentences = deserialize(jsonl_path, fmt="jsonl")
    train = [s for s in sentences if s.split == "train"]
    val = [s for s in sentences if s.split == "validation"]
    test = [s for s in sentences if s.split == "test"]
    logger.info("Loaded: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


def train_sentiment(train, val, test, seed: int, output_dir: Path):
    """Train and evaluate the SentimentClassifier."""
    from models.muril_sentiment_classifier import MuRILSentimentClassifier

    logger.info("=== Training SentimentClassifier ===")
    clf = MuRILSentimentClassifier(
        checkpoint_dir=str(output_dir / "sentiment"),
    )
    log = clf.train(train, val, seed=seed, max_epochs=10, batch_size=32, learning_rate=2e-5)
    logger.info("SentimentClassifier training complete: best_f1=%.4f", log.best_f1)

    metrics = clf.evaluate(test)
    logger.info("SentimentClassifier test macro-F1: %.4f", metrics.macro_f1)

    clf.save(str(output_dir / "sentiment" / "best"))
    return log, metrics


def train_ner(train, val, test, seed: int, output_dir: Path):
    """Train and evaluate the NERTagger."""
    from models.muril_ner_tagger import MuRILNERTagger

    logger.info("=== Training NERTagger ===")
    tagger = MuRILNERTagger(
        checkpoint_dir=str(output_dir / "ner"),
    )
    log = tagger.train(train, val, seed=seed, max_epochs=5, batch_size=16, learning_rate=3e-5)
    logger.info("NERTagger training complete: best_f1=%.4f", log.best_f1)

    metrics = tagger.evaluate(test)
    logger.info("NERTagger test macro-F1: %.4f", metrics.macro_f1)

    tagger.save(str(output_dir / "ner" / "best"))
    return log, metrics


def train_toxicity(train, val, test, seed: int, output_dir: Path):
    """Train and evaluate the ToxicityClassifier."""
    from models.muril_toxicity_classifier import MuRILToxicityClassifier

    logger.info("=== Training ToxicityClassifier ===")
    clf = MuRILToxicityClassifier(
        checkpoint_dir=str(output_dir / "toxicity"),
    )
    log = clf.train(train, val, seed=seed, max_epochs=10, batch_size=16, learning_rate=2e-5)
    logger.info("ToxicityClassifier training complete: best_f1=%.4f", log.best_f1)

    metrics = clf.evaluate(test)
    logger.info("ToxicityClassifier test macro-F1: %.4f", metrics.macro_f1)

    clf.save(str(output_dir / "toxicity" / "best"))
    return log, metrics


def run_baselines(test, output_dir: Path):
    """Run all baseline evaluations."""
    from evaluation.baselines import (
        ZeroShotMBERTEvaluator,
        ZeroShotMuRILEvaluator,
        GPT4o5ShotEvaluator,
    )
    from evaluation.comparison_table import generate_comparison_table, format_comparison_table

    logger.info("=== Running Baseline Evaluations ===")
    results = []
    for evaluator in [ZeroShotMBERTEvaluator(), ZeroShotMuRILEvaluator(), GPT4o5ShotEvaluator()]:
        results.extend(evaluator.run_all(test))

    table_path = output_dir / "comparison_table.txt"
    rows = generate_comparison_table(results)
    table_str = format_comparison_table(rows)
    table_path.write_text(table_str, encoding="utf-8")
    logger.info("Comparison table saved to %s", table_path)
    print("\n" + table_str)
    return results


def publish_to_huggingface(
    dataset_split,
    model_metrics: dict,
    seed: int,
    output_dir: Path,
    hf_repo_prefix: str,
    jsonl_path: str | None = None,
    parquet_path: str | None = None,
):
    """Publish dataset and models to HuggingFace."""
    from release.huggingface_publisher_real import RealHuggingFacePublisher

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — skipping HuggingFace publishing")
        return

    publisher = RealHuggingFacePublisher(token=token)
    results = publisher.publish_all(
        dataset_split=dataset_split,
        dataset_repo_id=f"{hf_repo_prefix}/rajnlp-50k",
        model_checkpoints={
            "SentimentClassifier": str(output_dir / "sentiment" / "best"),
            "NERTagger": str(output_dir / "ner" / "best"),
            "ToxicityClassifier": str(output_dir / "toxicity" / "best"),
        },
        model_metrics=model_metrics,
        seed=seed,
        jsonl_path=jsonl_path,
        parquet_path=parquet_path,
        annotation_guideline_path="docs/annotation_guidelines.md",
    )
    logger.info("HuggingFace publishing results: %s", results)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train all RajNLP-50K models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", default="output", help="Directory containing corpus.jsonl")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-ner", action="store_true")
    parser.add_argument("--skip-toxicity", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--publish", action="store_true", help="Publish to HuggingFace after training")
    parser.add_argument("--hf-repo-prefix", default="eeshsaxena", help="HuggingFace username/org")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fix all random seeds
    set_all_seeds(args.seed)
    logger.info("Random seed fixed: %d", args.seed)

    # Load corpus
    train, val, test = load_corpus(args.data_dir)

    model_metrics = {}

    # Train models
    if not args.skip_sentiment:
        log, metrics = train_sentiment(train, val, test, args.seed, output_dir)
        model_metrics["SentimentClassifier"] = metrics.macro_f1

    if not args.skip_ner:
        log, metrics = train_ner(train, val, test, args.seed, output_dir)
        model_metrics["NERTagger"] = metrics.macro_f1

    if not args.skip_toxicity:
        log, metrics = train_toxicity(train, val, test, args.seed, output_dir)
        model_metrics["ToxicityClassifier"] = metrics.macro_f1

    # Run baselines
    if not args.skip_baselines:
        run_baselines(test, output_dir)

    # Print summary
    logger.info("=== Training Summary ===")
    for model_name, f1 in model_metrics.items():
        from release.huggingface_publisher import TARGET_F1
        target = TARGET_F1.get(model_name, 0.0)
        status = "✓ PASS" if f1 >= target else "✗ FAIL"
        logger.info("%s: F1=%.4f (target=%.2f) %s", model_name, f1, target, status)

    # Publish to HuggingFace
    if args.publish:
        from corpus_builder.serialization import deserialize
        from models.data_models import DatasetSplit
        sentences = deserialize(Path(args.data_dir) / "corpus.jsonl", fmt="jsonl")
        dataset_split = DatasetSplit(
            train=[s for s in sentences if s.split == "train"],
            validation=[s for s in sentences if s.split == "validation"],
            test=[s for s in sentences if s.split == "test"],
        )
        publish_to_huggingface(
            dataset_split=dataset_split,
            model_metrics=model_metrics,
            seed=args.seed,
            output_dir=output_dir,
            hf_repo_prefix=args.hf_repo_prefix,
            jsonl_path=str(Path(args.data_dir) / "corpus.jsonl"),
            parquet_path=str(Path(args.data_dir) / "corpus.parquet"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
