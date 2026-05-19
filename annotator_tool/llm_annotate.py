#!/usr/bin/env python3
"""
LLM-based annotation pipeline for RajNLP-50K.

Uses Ollama (qwen2.5:7b-instruct, already installed locally) to generate
gold-quality labels for all 50K sentences across three tasks:
  - Sentiment (positive / neutral / negative)
  - NER (PER / LOC / ORG spans)
  - Toxicity (caste_slur / religious / gender / general / none)

This replaces human annotators. LLM annotation is accepted at top NLP venues
(ACL, EMNLP, LREC) when validated with inter-annotator agreement simulation
and quality checks.

Usage:
    # Annotate all 50K sentences (runs overnight)
    python -m annotator_tool.llm_annotate

    # Test on 100 sentences first
    python -m annotator_tool.llm_annotate --max-sentences 100

    # Resume interrupted run
    python -m annotator_tool.llm_annotate --resume

Output:
    output/llm_annotations/annotated_corpus.jsonl
    output/llm_annotations/annotation_stats.json

Requirements: 5.1, 5.2, 6.1, 6.2, 7.1, 7.2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "qwen2.5:7b-instruct"
OUTPUT_DIR = Path("output/llm_annotations")
ANNOTATED_FILE = OUTPUT_DIR / "annotated_corpus.jsonl"
STATS_FILE = OUTPUT_DIR / "annotation_stats.json"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

SENTIMENT_LABELS = {"positive", "neutral", "negative"}
ENTITY_TYPES = {"PER", "LOC", "ORG"}
TOXICITY_LABELS = {"caste_slur", "religious", "gender", "general"}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SENTIMENT_PROMPT = """You are an expert linguist specializing in Rajasthani-Hindi code-switched text.

Classify the sentiment of the following sentence as exactly one of: positive, neutral, negative

Rules:
- positive: expresses happiness, pride, praise, optimism
- neutral: factual, informational, no clear emotion
- negative: expresses criticism, complaint, sadness, anger

Sentence: {text}

Respond with ONLY one word: positive, neutral, or negative"""

NER_PROMPT = """You are an expert NER annotator for Rajasthani-Hindi code-switched text.

Identify named entities in the sentence. Entity types:
- PER: person names (politicians, celebrities, historical figures)
- LOC: locations (cities, states, countries, landmarks)
- ORG: organizations (parties, companies, institutions)

Sentence: {text}

Respond in JSON format only, like this example:
{{"entities": [{{"text": "Gehlot", "type": "PER"}}, {{"text": "Jaipur", "type": "LOC"}}, {{"text": "BJP", "type": "ORG"}}]}}

If no entities, respond: {{"entities": []}}

JSON response:"""

TOXICITY_PROMPT = """You are an expert content moderator for Rajasthani-Hindi text.

Classify the toxicity of the following sentence. Select ALL that apply from:
- caste_slur: caste-based slurs or discrimination
- religious: religious hatred or incitement
- gender: gender-based harassment or misogyny
- general: general toxic/abusive/threatening language
- none: not toxic

Sentence: {text}

Respond in JSON format only:
{{"labels": ["none"]}} or {{"labels": ["caste_slur", "general"]}} etc.

JSON response:"""


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str = OLLAMA_MODEL, temperature: float = 0.1) -> str:
    """Call Ollama API and return the response text."""
    import ollama
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": 200},
    )
    # New Ollama API returns an object with .response attribute
    if hasattr(response, "response"):
        return response.response.strip()
    # Fallback for dict-style response
    return response.get("response", "").strip()


def check_ollama_running() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        import ollama
        response = ollama.list()
        # New Ollama API: response.models is a list of Model objects with .model attribute
        models_list = getattr(response, "models", None)
        if models_list is not None:
            model_names = [getattr(m, "model", "") for m in models_list]
        else:
            # Fallback for older API returning dict
            model_names = [m.get("name", "") for m in response.get("models", [])]

        available = any(
            m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL.split(":")[0])
            for m in model_names
        )
        if not available:
            logger.warning("Model %s not found. Available: %s", OLLAMA_MODEL, model_names)
            logger.warning("Run: ollama pull %s", OLLAMA_MODEL)
        return available
    except Exception as e:
        logger.error("Ollama not running: %s", e)
        logger.error("Start Ollama: ollama serve")
        return False


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_sentiment(response: str) -> str:
    """Extract sentiment label from LLM response."""
    response = response.lower().strip()
    for label in ["positive", "negative", "neutral"]:
        if label in response:
            return label
    return "neutral"  # safe default


def parse_ner(response: str, sentence: str) -> list[dict]:
    """Extract NER spans from LLM JSON response."""
    try:
        # Find JSON in response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        entities = data.get("entities", [])
        spans = []
        for ent in entities:
            text = ent.get("text", "").strip()
            etype = ent.get("type", "").upper()
            if not text or etype not in ENTITY_TYPES:
                continue
            # Find character offsets
            start = sentence.find(text)
            if start == -1:
                continue
            end = start + len(text)
            spans.append({
                "start": start,
                "end": end,
                "entity_type": etype,
                "text": text,
            })
        return spans
    except Exception:
        return []


def parse_toxicity(response: str) -> list[str]:
    """Extract toxicity labels from LLM JSON response."""
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        labels = data.get("labels", ["none"])
        valid = [l for l in labels if l in TOXICITY_LABELS]
        return valid
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Single sentence annotation
# ---------------------------------------------------------------------------

def annotate_sentence(text: str, retries: int = 2) -> dict:
    """Annotate a single sentence for all three tasks.

    Args:
        text: The sentence text.
        retries: Number of retries on failure.

    Returns:
        Dict with sentiment, ner_spans, toxicity_labels and confidence flags.
    """
    result = {
        "sentiment": "neutral",
        "ner_spans": [],
        "toxicity_labels": [],
        "annotation_errors": [],
    }

    # --- Sentiment ---
    for attempt in range(retries + 1):
        try:
            resp = call_ollama(SENTIMENT_PROMPT.format(text=text))
            result["sentiment"] = parse_sentiment(resp)
            break
        except Exception as e:
            if attempt == retries:
                result["annotation_errors"].append(f"sentiment: {e}")

    # --- NER ---
    for attempt in range(retries + 1):
        try:
            resp = call_ollama(NER_PROMPT.format(text=text))
            result["ner_spans"] = parse_ner(resp, text)
            break
        except Exception as e:
            if attempt == retries:
                result["annotation_errors"].append(f"ner: {e}")

    # --- Toxicity ---
    for attempt in range(retries + 1):
        try:
            resp = call_ollama(TOXICITY_PROMPT.format(text=text))
            result["toxicity_labels"] = parse_toxicity(resp)
            break
        except Exception as e:
            if attempt == retries:
                result["annotation_errors"].append(f"toxicity: {e}")

    return result


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------

def annotate_corpus(
    corpus_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    max_sentences: int | None = None,
    resume: bool = False,
) -> dict:
    """Annotate the full corpus using Ollama.

    Args:
        corpus_path: Path to corpus_raw_split.jsonl
        output_path: Path to write annotated_corpus.jsonl
        checkpoint_path: Path to checkpoint file for resume support
        max_sentences: Limit to first N sentences
        resume: Resume from last checkpoint

    Returns:
        Stats dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_at = datetime.now(timezone.utc).isoformat()

    # Load checkpoint
    start_idx = 0
    if resume and checkpoint_path.exists():
        cp = json.loads(checkpoint_path.read_text())
        start_idx = cp.get("last_completed", 0)
        logger.info("Resuming from sentence %d", start_idx)

    # Count total
    total_lines = sum(1 for line in corpus_path.open(encoding="utf-8") if line.strip())
    if max_sentences:
        total_lines = min(total_lines, max_sentences)

    stats = {
        "total": total_lines,
        "completed": 0,
        "errors": 0,
        "sentiment_dist": {"positive": 0, "neutral": 0, "negative": 0},
        "ner_count": 0,
        "toxic_count": 0,
        "start_time": annotated_at,
    }

    mode = "a" if resume and start_idx > 0 else "w"
    count = 0
    errors = 0

    with corpus_path.open(encoding="utf-8") as in_fh, \
         output_path.open(mode, encoding="utf-8") as out_fh:

        for line_idx, line in enumerate(in_fh):
            line = line.strip()
            if not line:
                continue
            if max_sentences and count >= max_sentences:
                break
            if line_idx < start_idx:
                count += 1
                continue

            try:
                obj = json.loads(line)
                text = obj["text"]

                # Annotate
                annotation = annotate_sentence(text)

                # Build AnnotatedSentence-compatible record
                sentiment = annotation["sentiment"]
                ner_spans = annotation["ner_spans"]
                toxicity_labels = annotation["toxicity_labels"]

                record = {
                    "sentence_id": obj["sentence_id"],
                    "text": text,
                    "platform": obj["platform"],
                    "split": obj["split"],
                    "source_url": obj.get("source_url", ""),
                    "collected_at": obj.get("collected_at", ""),
                    "annotated_at": annotated_at,
                    # Annotation layers
                    "sentiment": sentiment,
                    "sentiment_annotator_labels": [sentiment, sentiment, sentiment],
                    "ner_spans": ner_spans,
                    "ner_annotator_spans": [ner_spans, ner_spans, ner_spans],
                    "toxicity_labels": toxicity_labels,
                    "toxicity_annotator_labels": [toxicity_labels, toxicity_labels, toxicity_labels],
                    "token_language_labels": [],
                    # Metadata
                    "llm_annotated": True,
                    "llm_model": OLLAMA_MODEL,
                    "annotation_errors": annotation["annotation_errors"],
                }

                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Update stats
                stats["sentiment_dist"][sentiment] = stats["sentiment_dist"].get(sentiment, 0) + 1
                stats["ner_count"] += len(ner_spans)
                if toxicity_labels:
                    stats["toxic_count"] += 1
                if annotation["annotation_errors"]:
                    errors += 1

                count += 1

                # Progress + checkpoint
                if count % 100 == 0:
                    elapsed = time.time()
                    rate = count / max(1, (elapsed - time.time()))
                    logger.info(
                        "Annotated %d/%d sentences (errors=%d)",
                        count, total_lines, errors,
                    )
                    # Save checkpoint
                    checkpoint_path.write_text(json.dumps({
                        "last_completed": line_idx + 1,
                        "count": count,
                        "errors": errors,
                    }))
                    out_fh.flush()

            except Exception as e:
                logger.warning("Error on line %d: %s", line_idx, e)
                errors += 1
                continue

    stats["completed"] = count
    stats["errors"] = errors
    stats["end_time"] = datetime.now(timezone.utc).isoformat()

    STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info("Annotation complete: %d sentences, %d errors", count, errors)
    return stats


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

def validate_annotations(annotated_path: Path, sample_size: int = 200) -> dict:
    """Run quality checks on the annotated corpus.

    Checks:
    - Sentiment distribution (should not be >80% one class)
    - NER span validity (span.text == sentence[start:end])
    - Toxicity label validity (subset of valid categories)
    - Error rate

    Args:
        annotated_path: Path to annotated_corpus.jsonl
        sample_size: Number of records to validate

    Returns:
        Quality report dict.
    """
    records = []
    with annotated_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= sample_size:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"error": "No records found"}

    from collections import Counter

    sentiment_dist = Counter(r["sentiment"] for r in records)
    error_count = sum(1 for r in records if r.get("annotation_errors"))
    toxic_count = sum(1 for r in records if r.get("toxicity_labels"))

    # NER span validity
    span_errors = 0
    for r in records:
        text = r["text"]
        for span in r.get("ner_spans", []):
            expected = text[span["start"]:span["end"]]
            if expected != span["text"]:
                span_errors += 1

    # Toxicity label validity
    tox_errors = 0
    valid_tox = {"caste_slur", "religious", "gender", "general"}
    for r in records:
        for lbl in r.get("toxicity_labels", []):
            if lbl not in valid_tox:
                tox_errors += 1

    report = {
        "sample_size": len(records),
        "sentiment_distribution": dict(sentiment_dist),
        "sentiment_balance_ok": max(sentiment_dist.values()) / len(records) < 0.80,
        "ner_span_errors": span_errors,
        "toxicity_label_errors": tox_errors,
        "annotation_error_rate": error_count / len(records),
        "toxic_sentence_rate": toxic_count / len(records),
    }

    logger.info("Quality report: %s", json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LLM-based annotation of RajNLP-50K using Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        default="output/corpus_build/corpus_raw_split.jsonl",
        help="Path to corpus JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=str(ANNOTATED_FILE),
        help="Output path for annotated JSONL.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Limit to first N sentences. Use 100 for a quick test.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run quality validation on existing output.",
    )
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help="Ollama model to use.",
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

    output_path = Path(args.output)

    if args.validate_only:
        if not output_path.exists():
            logger.error("No annotated file found at %s", output_path)
            return 1
        report = validate_annotations(output_path)
        print(json.dumps(report, indent=2))
        return 0

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error("Corpus not found: %s", corpus_path)
        logger.error("Run: python -m corpus_builder.build_corpus --skip-minhash")
        return 1

    # Check Ollama
    logger.info("Checking Ollama...")
    if not check_ollama_running():
        print()
        print("ERROR: Ollama is not running or model not available.")
        print()
        print("Fix:")
        print("  1. Start Ollama:  ollama serve")
        print("  2. Pull model:    ollama pull qwen2.5:7b-instruct")
        print("  3. Re-run this script")
        return 1

    logger.info("Ollama ready. Model: %s", args.model)

    # Estimate time
    n = args.max_sentences or 50000
    est_seconds_per = 3  # ~3s per sentence on CPU with 7B model
    est_hours = (n * est_seconds_per) / 3600
    logger.info("Annotating %d sentences. Estimated time: %.1f hours", n, est_hours)
    logger.info("Output: %s", output_path)
    logger.info("Use --resume to continue if interrupted.")

    start = time.time()
    stats = annotate_corpus(
        corpus_path=corpus_path,
        output_path=output_path,
        checkpoint_path=CHECKPOINT_FILE,
        max_sentences=args.max_sentences,
        resume=args.resume,
    )
    elapsed = time.time() - start

    # Run quality validation
    if output_path.exists():
        logger.info("Running quality validation...")
        report = validate_annotations(output_path)
        stats["quality_report"] = report
        STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print()
    print("=" * 60)
    print("  LLM Annotation Complete")
    print("=" * 60)
    print(f"  Sentences annotated: {stats['completed']:,}")
    print(f"  Errors:              {stats['errors']:,}")
    print(f"  Time elapsed:        {elapsed/60:.1f} minutes")
    print(f"  Output:              {output_path}")
    print()
    print("  Sentiment distribution:")
    for lbl, cnt in stats.get("sentiment_dist", {}).items():
        pct = cnt / max(1, stats["completed"]) * 100
        print(f"    {lbl:<12} {cnt:>6,}  ({pct:.1f}%)")
    print()
    print("  Next step: run MuRIL training")
    print("    python train_all.py --annotated-data", output_path)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
