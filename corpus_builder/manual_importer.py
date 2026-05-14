"""
Manual sentence importer for RajNLP-50K.

Allows importing sentences from a CSV or JSON file without requiring
any API credentials. Useful when:
- Twitter/X API approval is pending
- You want to manually curate sentences
- You have sentences from other sources (WhatsApp, Facebook, news sites)

Usage:
    # From CSV
    python -m corpus_builder.manual_importer \
        --input data/manual_sentences.csv \
        --output data/manual_raw.jsonl

    # From JSON
    python -m corpus_builder.manual_importer \
        --input data/manual_sentences.json \
        --output data/manual_raw.jsonl

CSV format (with header row):
    text,source_url,platform,notes
    "म्हारो राजस्थान घणो सुंदर है","https://twitter.com/...","twitter",""
    "Gehlot ne Jaipur mein rally ki","https://sharechat.com/...","sharechat",""

JSON format (list of objects):
    [
        {"text": "...", "source_url": "...", "platform": "twitter"},
        ...
    ]

The importer:
1. Reads your file
2. Assigns a UUID to each sentence
3. Normalizes text to NFC Unicode
4. Filters out empty/too-short sentences
5. Outputs a JSONL file compatible with the rest of the pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from models.data_models import RawSentence

logger = logging.getLogger(__name__)

# Minimum number of characters for a sentence to be kept
MIN_CHARS = 10

# Valid platform values
VALID_PLATFORMS = {"twitter", "sharechat", "facebook", "whatsapp", "news", "other"}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _make_raw_sentence(
    text: str,
    source_url: str = "",
    platform: str = "other",
    collected_at: datetime | None = None,
) -> RawSentence | None:
    """Create a RawSentence from raw input, or return None if invalid."""
    text = _nfc(text.strip())
    if len(text) < MIN_CHARS:
        return None

    if platform not in VALID_PLATFORMS:
        logger.warning("Unknown platform %r — using 'other'", platform)
        platform = "other"

    # Map non-standard platforms to the two official ones for pipeline compatibility
    platform_mapped = "twitter" if platform in {"twitter", "facebook", "whatsapp", "news", "other"} else "sharechat"

    return RawSentence(
        text=text,
        source_url=source_url or f"manual://{platform}",
        collected_at=collected_at or datetime.now(tz=timezone.utc),
        platform=platform_mapped,  # type: ignore[arg-type]
        sentence_id=str(uuid.uuid4()),
    )


def import_from_csv(path: Path) -> list[RawSentence]:
    """Import sentences from a CSV file.

    Expected columns: text, source_url (optional), platform (optional)
    The first row must be a header row.

    Args:
        path: Path to the CSV file.

    Returns:
        List of RawSentence objects.
    """
    sentences: list[RawSentence] = []
    skipped = 0

    with path.open(encoding="utf-8-sig") as fh:  # utf-8-sig handles Excel BOM
        reader = csv.DictReader(fh)

        if "text" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV must have a 'text' column. Found columns: {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=2):  # start=2 because row 1 is header
            text = row.get("text", "").strip()
            source_url = row.get("source_url", "").strip()
            platform = row.get("platform", "other").strip().lower()

            sentence = _make_raw_sentence(text, source_url, platform)
            if sentence is None:
                logger.debug("Row %d skipped (too short or empty): %r", i, text[:50])
                skipped += 1
                continue

            sentences.append(sentence)

    logger.info(
        "CSV import: %d sentences imported, %d skipped from %s",
        len(sentences), skipped, path,
    )
    return sentences


def import_from_json(path: Path) -> list[RawSentence]:
    """Import sentences from a JSON file (list of objects).

    Each object must have a "text" field. Optional: "source_url", "platform".

    Args:
        path: Path to the JSON file.

    Returns:
        List of RawSentence objects.
    """
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects at the top level.")

    sentences: list[RawSentence] = []
    skipped = 0

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Item %d is not a dict — skipping", i)
            skipped += 1
            continue

        text = str(item.get("text", "")).strip()
        source_url = str(item.get("source_url", "")).strip()
        platform = str(item.get("platform", "other")).strip().lower()

        sentence = _make_raw_sentence(text, source_url, platform)
        if sentence is None:
            skipped += 1
            continue

        sentences.append(sentence)

    logger.info(
        "JSON import: %d sentences imported, %d skipped from %s",
        len(sentences), skipped, path,
    )
    return sentences


def import_from_txt(path: Path, platform: str = "other") -> list[RawSentence]:
    """Import sentences from a plain text file (one sentence per line).

    Args:
        path: Path to the text file.
        platform: Platform to assign to all sentences (default "other").

    Returns:
        List of RawSentence objects.
    """
    sentences: list[RawSentence] = []
    skipped = 0

    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            text = line.strip()
            if not text or text.startswith("#"):  # skip empty lines and comments
                continue

            sentence = _make_raw_sentence(text, platform=platform)
            if sentence is None:
                skipped += 1
                continue

            sentences.append(sentence)

    logger.info(
        "TXT import: %d sentences imported, %d skipped from %s",
        len(sentences), skipped, path,
    )
    return sentences


def save_to_jsonl(sentences: list[RawSentence], output_path: Path) -> None:
    """Save RawSentence objects to a JSONL file.

    Args:
        sentences: List of RawSentence objects.
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for s in sentences:
            record = {
                "text": s.text,
                "source_url": s.source_url,
                "collected_at": s.collected_at.isoformat(),
                "platform": s.platform,
                "sentence_id": s.sentence_id,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d sentences to %s", len(sentences), output_path)


def load_from_jsonl(path: Path) -> list[RawSentence]:
    """Load RawSentence objects from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of RawSentence objects.
    """
    sentences: list[RawSentence] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sentences.append(RawSentence(
                text=d["text"],
                source_url=d.get("source_url", ""),
                collected_at=datetime.fromisoformat(d["collected_at"]),
                platform=d["platform"],  # type: ignore[arg-type]
                sentence_id=d["sentence_id"],
            ))
    return sentences


def merge_jsonl_files(input_paths: list[Path], output_path: Path) -> int:
    """Merge multiple JSONL files into one, deduplicating by sentence_id.

    Args:
        input_paths: List of input JSONL file paths.
        output_path: Output file path.

    Returns:
        Number of sentences in the merged output.
    """
    seen_ids: set[str] = set()
    all_sentences: list[RawSentence] = []

    for path in input_paths:
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue
        sentences = load_from_jsonl(path)
        for s in sentences:
            if s.sentence_id not in seen_ids:
                seen_ids.add(s.sentence_id)
                all_sentences.append(s)

    save_to_jsonl(all_sentences, output_path)
    logger.info(
        "Merged %d files → %d unique sentences → %s",
        len(input_paths), len(all_sentences), output_path,
    )
    return len(all_sentences)


def main(argv=None):
    """CLI entry point for the manual importer."""
    parser = argparse.ArgumentParser(
        description="Import sentences manually into the RajNLP-50K pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file path (.csv, .json, or .txt)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/manual_raw.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--platform",
        default="other",
        choices=list(VALID_PLATFORMS),
        help="Platform to assign (used for .txt files; CSV/JSON use the 'platform' column)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        sentences = import_from_csv(input_path)
    elif suffix == ".json":
        sentences = import_from_json(input_path)
    elif suffix == ".txt":
        sentences = import_from_txt(input_path, platform=args.platform)
    else:
        logger.error("Unsupported file format: %s (use .csv, .json, or .txt)", suffix)
        sys.exit(1)

    if not sentences:
        logger.error("No valid sentences found in %s", input_path)
        sys.exit(1)

    save_to_jsonl(sentences, output_path)
    print(f"\n✓ Imported {len(sentences)} sentences → {output_path}")
    print(f"\nNext step: run the corpus pipeline:")
    print(f"  python run_pipeline.py --seed 42 --output-dir output/run_001")


if __name__ == "__main__":
    main()
