#!/usr/bin/env python3
"""
Auto-annotation script for RajNLP-50K.

Uses zero-shot multilingual models (no GPU, no fine-tuning required) to generate
draft labels for all 50K sentences. Human annotators then only need to CORRECT
wrong labels rather than label from scratch — significantly faster.

Models used (all free, CPU-compatible):
  - Sentiment:  cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingual)
  - NER:        Davlan/xlm-roberta-base-ner-hrl (multilingual NER)
  - Toxicity:   zero-shot classification via facebook/bart-large-mnli

Output: output/auto_annotations/auto_annotated.jsonl
  Each record is an AnnotatedSentence-compatible dict with draft labels.
  Confidence scores are included so annotators can prioritize low-confidence items.

Usage:
    python -m annotator_tool.auto_annotate [--corpus output/corpus_build/corpus_raw_split.jsonl]
    python -m annotator_tool.auto_annotate --max-sentences 1000  # test on subset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
TOXICITY_LABELS = ["caste_slur", "religious", "gender", "general", "non-toxic"]

# Label mapping from model output to our schema
SENTIMENT_MAP = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "Positive": "positive",
    "Neutral": "neutral",
    "Negative": "negative",
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_sentiment_pipeline = None
_ner_pipeline = None
_zero_shot_pipeline = None


def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        logger.info("Loading sentiment model: %s", SENTIMENT_MODEL)
        _sentiment_pipeline = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            device=-1,  # CPU
            truncation=True,
            max_length=512,
        )
        logger.info("Sentiment model loaded.")
    return _sentiment_pipeline


def _get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        logger.info("Loading NER model: %s", NER_MODEL)
        _ner_pipeline = pipeline(
            "ner",
            model=NER_MODEL,
            tokenizer=NER_MODEL,
            device=-1,  # CPU
            aggregation_strategy="simple",
            truncation=True,
            max_length=512,
        )
        logger.info("NER model loaded.")
    return _ner_pipeline


def _get_zero_shot_pipeline():
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        from transformers import pipeline
        logger.info("Loading zero-shot classification model for toxicity")
        _zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU
        )
        logger.info("Zero-shot model loaded.")
    return _zero_shot_pipeline


# ---------------------------------------------------------------------------
# Annotation functions
# ---------------------------------------------------------------------------

def annotate_sentiment(text: str) -> tuple[str, float]:
    """Return (label, confidence) for sentiment."""
    try:
        pipe = _get_sentiment_pipeline()
        result = pipe(text[:512])[0]
        raw_label = result["label"]
        label = SENTIMENT_MAP.get(raw_label, "neutral")
        confidence = float(result["score"])
        return label, confidence
    except Exception as e:
        logger.debug("Sentiment error for text %r: %s", text[:50], e)
        return "neutral", 0.0


def annotate_ner(text: str) -> list[dict]:
    """Return list of {start, end, entity_type, text, confidence} dicts."""
    try:
        pipe = _get_ner_pipeline()
        entities = pipe(text[:512])
        spans = []
        for ent in entities:
            # Map entity group to our types
            group = ent.get("entity_group", ent.get("entity", ""))
            if "PER" in group or "per" in group.lower():
                etype = "PER"
            elif "LOC" in group or "loc" in group.lower() or "GPE" in group:
                etype = "LOC"
            elif "ORG" in group or "org" in group.lower():
                etype = "ORG"
            else:
                continue  # skip MISC and others

            span_text = ent.get("word", "").replace("▁", "").strip()
            if not span_text:
                continue

            # Find character offsets in original text
            start = text.find(span_text)
            if start == -1:
                continue
            end = start + len(span_text)

            spans.append({
                "start": start,
                "end": end,
                "entity_type": etype,
                "text": span_text,
                "confidence": float(ent.get("score", 0.0)),
            })
        return spans
    except Exception as e:
        logger.debug("NER error for text %r: %s", text[:50], e)
        return []


def annotate_toxicity(text: str) -> tuple[list[str], dict[str, float]]:
    """Return (labels, per_category_scores) for toxicity.

    Uses zero-shot classification with candidate labels.
    Returns empty list if classified as non-toxic.
    """
    try:
        pipe = _get_zero_shot_pipeline()
        candidate_labels = [
            "caste discrimination or slur",
            "religious hatred or incitement",
            "gender-based harassment or misogyny",
            "general toxic or abusive language",
            "neutral or non-toxic content",
        ]
        result = pipe(text[:512], candidate_labels=candidate_labels, multi_label=True)

        label_map = {
            "caste discrimination or slur": "caste_slur",
            "religious hatred or incitement": "religious",
            "gender-based harassment or misogyny": "gender",
            "general toxic or abusive language": "general",
            "neutral or non-toxic content": None,
        }

        scores = {}
        labels = []
        threshold = 0.5

        for lbl, score in zip(result["labels"], result["scores"]):
            mapped = label_map.get(lbl)
            if mapped:
                scores[mapped] = float(score)
                if score >= threshold:
                    labels.append(mapped)

        return labels, scores
    except Exception as e:
        logger.debug("Toxicity error for text %r: %s", text[:50], e)
        return [], {}


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def auto_annotate_corpus(
    corpus_path: Path,
    output_path: Path,
    max_sentences: int | None = None,
    batch_log_interval: int = 100,
) -> int:
    """Run auto-annotation on the full corpus.

    Args:
        corpus_path: Path to corpus_raw_split.jsonl
        output_path: Path to write auto_annotated.jsonl
        max_sentences: Limit to first N sentences (None = all)
        batch_log_interval: Log progress every N sentences

    Returns:
        Number of sentences annotated.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_at = datetime.now(timezone.utc).isoformat()

    count = 0
    errors = 0

    with corpus_path.open(encoding="utf-8") as in_fh, \
         output_path.open("w", encoding="utf-8") as out_fh:

        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            if max_sentences and count >= max_sentences:
                break

            try:
                obj = json.loads(line)
                text = obj["text"]

                # Run all three annotation tasks
                sentiment, sentiment_conf = annotate_sentiment(text)
                ner_spans = annotate_ner(text)
                toxicity_labels, toxicity_scores = annotate_toxicity(text)

                # Build output record (AnnotatedSentence-compatible)
                record = {
                    "sentence_id": obj["sentence_id"],
                    "text": text,
                    "platform": obj["platform"],
                    "split": obj["split"],
                    "source_url": obj.get("source_url", ""),
                    "collected_at": obj.get("collected_at", ""),
                    "annotated_at": annotated_at,
                    # Draft labels (to be verified/corrected by humans)
                    "sentiment": sentiment,
                    "sentiment_confidence": sentiment_conf,
                    "sentiment_annotator_labels": [sentiment, sentiment, sentiment],
                    "ner_spans": ner_spans,
                    "ner_annotator_spans": [ner_spans, [], []],
                    "toxicity_labels": toxicity_labels,
                    "toxicity_scores": toxicity_scores,
                    "toxicity_annotator_labels": [toxicity_labels, [], []],
                    "token_language_labels": [],  # filled by LanguageIDTagger
                    # Metadata
                    "auto_annotated": True,
                    "needs_human_review": sentiment_conf < 0.8,
                }

                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                if count % batch_log_interval == 0:
                    logger.info(
                        "Annotated %d sentences (errors=%d)...", count, errors
                    )

            except Exception as e:
                logger.warning("Error on sentence %d: %s", count, e)
                errors += 1
                continue

    logger.info("Auto-annotation complete: %d sentences, %d errors", count, errors)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Auto-annotate RajNLP-50K corpus with draft labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        default="output/corpus_build/corpus_raw_split.jsonl",
        help="Path to corpus JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="output/auto_annotations/auto_annotated.jsonl",
        help="Output path for auto-annotated JSONL.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Limit to first N sentences (default: all 50K). Use 100 for a quick test.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["sentiment", "ner", "toxicity", "all"],
        default=["all"],
        help="Which annotation tasks to run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error("Corpus file not found: %s", corpus_path)
        logger.error("Run first: python -m corpus_builder.build_corpus --skip-minhash")
        return 1

    output_path = Path(args.output)

    logger.info("Starting auto-annotation...")
    logger.info("  Corpus:   %s", corpus_path)
    logger.info("  Output:   %s", output_path)
    logger.info("  Max:      %s", args.max_sentences or "all")
    logger.info("")
    logger.info("NOTE: This uses CPU-only inference. Expect ~2-5 seconds per sentence.")
    logger.info("      For 50K sentences, this will take several hours.")
    logger.info("      Use --max-sentences 500 for a quick test first.")
    logger.info("")

    start = time.time()
    count = auto_annotate_corpus(
        corpus_path=corpus_path,
        output_path=output_path,
        max_sentences=args.max_sentences,
    )
    elapsed = time.time() - start

    print()
    print("="*60)
    print("  Auto-Annotation Complete")
    print("="*60)
    print(f"  Sentences annotated: {count:,}")
    print(f"  Time elapsed:        {elapsed/60:.1f} minutes")
    print(f"  Output:              {output_path}")
    print()
    print("  Next steps:")
    print("  1. Import auto_annotated.jsonl into Label Studio")
    print("  2. Annotators review and correct draft labels")
    print("  3. Focus review on low-confidence items (needs_human_review=true)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
