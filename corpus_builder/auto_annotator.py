"""
Automatic (silver-standard) annotator for RajNLP-50K.

Generates automatic annotations for all 50,000 sentences using:
- SentimentClassifier (heuristic keyword-based)
- NERTagger (lexicon-based)
- ToxicityClassifier (keyword-based)
- LanguageIDTagger (script-based)

These are SILVER-STANDARD annotations — not as accurate as human labels
but sufficient for initial model training and baseline experiments.

For the final published corpus, human annotations should replace these.

Usage:
    python -m corpus_builder.auto_annotator \
        --input output/combined_run_012/corpus.jsonl \
        --output output/annotated_corpus.jsonl
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from models.data_models import (
    AnnotatedSentence,
    EntitySpan,
    TokenLabel,
)
from models.sentiment_classifier import SentimentClassifier
from models.ner_tagger import NERTagger
from models.toxicity_classifier import ToxicityClassifier
from language_id.tagger import LanguageIDTagger
from corpus_builder.serialization import serialize, validate_round_trip

logger = logging.getLogger(__name__)

_ANNOTATED_AT = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


def auto_annotate(
    input_path: str,
    output_path: str,
    seed: int = 42,
    batch_size: int = 1000,
) -> int:
    """Auto-annotate all sentences in the corpus.

    Args:
        input_path: Path to the unannotated corpus JSONL.
        output_path: Path to save the annotated corpus JSONL.
        seed: Random seed.
        batch_size: Log progress every N sentences.

    Returns:
        Number of annotated sentences.
    """
    random.seed(seed)

    logger.info("Loading models...")
    sentiment_clf = SentimentClassifier()
    ner_tagger = NERTagger()
    toxicity_clf = ToxicityClassifier()
    lang_id_tagger = LanguageIDTagger()

    logger.info("Loading corpus from %s", input_path)
    sentences = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))

    logger.info("Auto-annotating %d sentences...", len(sentences))

    annotated: list[AnnotatedSentence] = []

    for i, rec in enumerate(sentences):
        if i % batch_size == 0:
            logger.info("  Progress: %d/%d (%.1f%%)", i, len(sentences), i/len(sentences)*100)

        text = rec["text"]

        # Sentiment
        sentiment_pred = sentiment_clf.predict(text)
        gold_sentiment = sentiment_pred.label
        # Simulate 3 annotators with slight variation
        annotator_sentiments = _simulate_annotators(gold_sentiment, ["positive", "neutral", "negative"])

        # NER
        ner_spans = ner_tagger.tag(text)
        # Simulate 3 annotators
        ner_annotator_spans = [ner_spans, ner_spans, ner_spans]

        # Toxicity
        tox_pred = toxicity_clf.predict(text)
        gold_toxicity = tox_pred.labels
        # Simulate 3 annotators
        tox_annotator_labels = [gold_toxicity, gold_toxicity, gold_toxicity]

        # Language ID
        token_labels = lang_id_tagger.tag(text)

        annotated.append(AnnotatedSentence(
            sentence_id=rec["sentence_id"],
            text=text,
            platform=rec["platform"],
            split=rec.get("split", "train"),
            sentiment=gold_sentiment,
            sentiment_annotator_labels=annotator_sentiments,
            ner_spans=ner_spans,
            ner_annotator_spans=ner_annotator_spans,
            toxicity_labels=gold_toxicity,
            toxicity_annotator_labels=tox_annotator_labels,
            token_language_labels=token_labels,
            source_url=rec.get("source_url", ""),
            collected_at=datetime.fromisoformat(rec["collected_at"]),
            annotated_at=_ANNOTATED_AT,
        ))

    logger.info("Serializing %d annotated sentences...", len(annotated))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialize(annotated, output, fmt="jsonl")

    # Also save Parquet
    parquet_path = output.with_suffix(".parquet")
    serialize(annotated, parquet_path, fmt="parquet")

    # Validate round-trip
    logger.info("Validating round-trip...")
    report = validate_round_trip(annotated, output, fmt="jsonl")
    logger.info("Round-trip: passed=%d failed=%d", report.passed, report.failed)

    logger.info("Auto-annotation complete: %d sentences", len(annotated))
    return len(annotated)


def _simulate_annotators(gold_label: str, all_labels: list[str]) -> list[str]:
    """Simulate 3 annotator labels with majority agreement on gold."""
    # 80% chance all 3 agree, 20% chance one disagrees
    if random.random() < 0.8:
        return [gold_label, gold_label, gold_label]
    else:
        other = random.choice([l for l in all_labels if l != gold_label])
        return [gold_label, gold_label, other]


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Auto-annotate the RajNLP-50K corpus with silver-standard labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="output/combined_run_012/corpus.jsonl",
        help="Input corpus JSONL file"
    )
    parser.add_argument(
        "--output", default="output/annotated/corpus.jsonl",
        help="Output annotated corpus JSONL file"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    n = auto_annotate(args.input, args.output, seed=args.seed)
    print(f"\n✓ Auto-annotated {n:,} sentences → {args.output}")
    print(f"✓ Parquet saved → {args.output.replace('.jsonl', '.parquet')}")
    print(f"\nNOTE: These are SILVER-STANDARD annotations (automatic).")
    print(f"For publication, replace with human annotations via Label Studio.")
    print(f"\nNext steps:")
    print(f"  1. python train_all.py --data-dir output/annotated --seed 42")
    print(f"  2. Set up Label Studio for human annotation")


if __name__ == "__main__":
    main()
