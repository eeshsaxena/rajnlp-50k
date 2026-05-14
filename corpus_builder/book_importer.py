"""
Book / PDF / text file importer for RajNLP-50K.

Imports sentences from:
  - PDF files (e.g., Sahitya Sujas, other Rajasthani books)
  - Plain text files (.txt)
  - Already-extracted text

IMPORTANT: Only use this with texts you have permission to use,
or texts that are in the public domain.

Usage:
    # From PDF
    python -m corpus_builder.book_importer \
        --input "Sahitya_Sujas_234.pdf" \
        --output data/sahitya_sujas_raw.jsonl \
        --source-name "Sahitya Sujas" \
        --platform other

    # From text file
    python -m corpus_builder.book_importer \
        --input "rajasthani_text.txt" \
        --output data/book_raw.jsonl

    # Then feed into pipeline
    python run_pipeline.py \
        --manual-data data/sahitya_sujas_raw.jsonl \
        --seed 42 --output-dir output/run_001
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from models.data_models import RawSentence

logger = logging.getLogger(__name__)

# Try to import Krutidev converter for legacy font PDFs
try:
    from corpus_builder.krutidev_converter import convert_pdf_text, is_likely_krutidev
    _KRUTIDEV_AVAILABLE = True
except ImportError:
    _KRUTIDEV_AVAILABLE = False

MIN_SENTENCE_CHARS = 15
MAX_SENTENCE_CHARS = 1000  # increased from 500 to handle longer literary sentences


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clean_text(text: str) -> str:
    """Clean extracted text — remove page numbers, headers, artifacts."""
    # Remove page numbers (standalone numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove repeated dashes/lines (section dividers)
    text = re.sub(r'[-_=]{3,}', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using Devanagari and Latin punctuation."""
    text = _nfc(text)
    text = _clean_text(text)

    # Split on sentence-ending punctuation
    # Devanagari danda (।), double danda (॥), and standard punctuation
    parts = re.split(r'(?<=[।॥?!])\s*|(?<=[.?!])\s+(?=[A-Z\u0900-\u097F])', text)

    sentences = []
    for part in parts:
        part = part.strip()
        if (MIN_SENTENCE_CHARS <= len(part) <= MAX_SENTENCE_CHARS
                and not re.match(r'^[\d\s\W]+$', part)
                and not part.startswith("http")):
            sentences.append(part)

    return sentences


def import_from_pdf(
    path: Path,
    source_name: str = "book",
    platform: str = "other",
) -> list[RawSentence]:
    """Extract sentences from a PDF file.

    Requires: pip install pdfplumber

    Args:
        path: Path to the PDF file.
        source_name: Name of the source (used in source_url).
        platform: Platform label for the sentences.

    Returns:
        List of RawSentence objects.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF import.\n"
            "Install with: pip install pdfplumber"
        )

    platform_mapped = "twitter" if platform in {"twitter"} else "sharechat"
    sentences: list[RawSentence] = []
    collected_at = datetime.now(tz=timezone.utc)

    logger.info("Extracting text from PDF: %s", path)

    with pdfplumber.open(str(path)) as pdf:
        total_pages = len(pdf.pages)
        logger.info("PDF has %d pages — processing ALL pages (no limit)", total_pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num % 10 == 0:
                logger.info("Processing page %d/%d...", page_num, total_pages)

            text = page.extract_text()
            if not text:
                continue

            # Auto-detect and convert Krutidev/legacy font encoding
            if _KRUTIDEV_AVAILABLE and is_likely_krutidev(text):
                text = convert_pdf_text(text)
                logger.debug("Page %d: converted from Krutidev encoding", page_num)

            page_sentences = _split_sentences(text)
            for sent in page_sentences:
                sentences.append(RawSentence(
                    text=sent,
                    source_url=f"book://{source_name}/page/{page_num}",
                    collected_at=collected_at,
                    platform=platform_mapped,  # type: ignore[arg-type]
                    sentence_id=str(uuid.uuid4()),
                ))

    logger.info("PDF import: extracted %d sentences from %d pages", len(sentences), total_pages)
    return sentences


def import_from_txt(
    path: Path,
    source_name: str = "text",
    platform: str = "other",
) -> list[RawSentence]:
    """Extract sentences from a plain text file.

    Args:
        path: Path to the text file.
        source_name: Name of the source.
        platform: Platform label.

    Returns:
        List of RawSentence objects.
    """
    platform_mapped = "twitter" if platform in {"twitter"} else "sharechat"
    collected_at = datetime.now(tz=timezone.utc)

    with path.open(encoding="utf-8", errors="replace") as fh:
        full_text = fh.read()

    # Split into paragraphs first, then sentences
    paragraphs = re.split(r'\n\s*\n', full_text)
    sentences: list[RawSentence] = []

    for para in paragraphs:
        para_sentences = _split_sentences(para)
        for sent in para_sentences:
            sentences.append(RawSentence(
                text=sent,
                source_url=f"book://{source_name}",
                collected_at=collected_at,
                platform=platform_mapped,  # type: ignore[arg-type]
                sentence_id=str(uuid.uuid4()),
            ))

    logger.info("TXT import: extracted %d sentences from %s", len(sentences), path)
    return sentences


def save_to_jsonl(sentences: list[RawSentence], output_path: Path) -> None:
    """Save sentences to JSONL format."""
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import sentences from a book/PDF/text file into RajNLP-50K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input file (.pdf or .txt)")
    parser.add_argument("--output", "-o", default="data/book_raw.jsonl", help="Output JSONL file")
    parser.add_argument("--source-name", default="book", help="Name of the source book/magazine")
    parser.add_argument("--platform", default="other",
                        choices=["twitter", "sharechat", "other"],
                        help="Platform label for imported sentences")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        sentences = import_from_pdf(input_path, args.source_name, args.platform)
    elif suffix in (".txt", ".text"):
        sentences = import_from_txt(input_path, args.source_name, args.platform)
    else:
        print(f"ERROR: Unsupported format '{suffix}'. Use .pdf or .txt")
        sys.exit(1)

    if not sentences:
        print("No sentences extracted. Check the file content.")
        sys.exit(1)

    output_path = Path(args.output)
    save_to_jsonl(sentences, output_path)

    print(f"\n✓ Extracted {len(sentences)} sentences → {output_path}")
    print(f"\nNext step:")
    print(f"  python run_pipeline.py --manual-data {output_path} --seed 42 --output-dir output/run_001")
    print(f"\nOr combine with other sources:")
    print(f"  python run_pipeline.py --manual-data {output_path},data/patrika_raw.jsonl --seed 42 --output-dir output/run_001")


if __name__ == "__main__":
    main()
