#!/usr/bin/env python3
"""
Manual data importer for RajNLP-50K.

Loads existing JSONL files from the data/ directory, normalizes platform labels,
and merges them into a single unified corpus for the pipeline.

Usage:
    python -m corpus_builder.manual_importer --output data/merged_corpus.jsonl

Platform mapping:
    - bhaskar*.jsonl → platform="news_bhaskar"
    - patrika*.jsonl → platform="news_patrika"
    - *vyakaran*.jsonl, *lock*.jsonl, *gusain*.jsonl, *loka*.jsonl, rj06*.jsonl → platform="books"
    - wikipedia*.jsonl → platform="wikipedia"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from models.data_models import RawSentence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform mapping rules
# ---------------------------------------------------------------------------

def _infer_platform(filename: str) -> str:
    """Infer the platform label from the filename.

    Args:
        filename: The name of the JSONL file (e.g., "bhaskar_raw.jsonl").

    Returns:
        A platform string: "news_bhaskar", "news_patrika", "books", or "wikipedia".
    """
    name_lower = filename.lower()
    
    if "bhaskar" in name_lower:
        return "news_bhaskar"
    elif "patrika" in name_lower:
        return "news_patrika"
    elif any(kw in name_lower for kw in ["vyakaran", "lock", "gusain", "loka", "rj06", "kaumudi"]):
        return "books"
    elif "wikipedia" in name_lower or "gov" in name_lower:
        return "wikipedia"
    else:
        # Default fallback
        return "other"


# ---------------------------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------------------------

def load_from_jsonl(path: Path, override_platform: str | None = None) -> list[RawSentence]:
    """Load RawSentence objects from a JSONL file.

    Args:
        path: Path to the JSONL file.
        override_platform: If provided, override the platform field in all loaded sentences.

    Returns:
        A list of :class:`~models.data_models.RawSentence` objects.
    """
    sentences: list[RawSentence] = []
    inferred_platform = _infer_platform(path.name)
    
    with path.open(encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Normalize platform
                platform = override_platform or inferred_platform
                
                # Parse collected_at if it's a string
                collected_at = obj.get("collected_at")
                if isinstance(collected_at, str):
                    try:
                        collected_at = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
                    except:
                        collected_at = datetime.now(timezone.utc)
                elif collected_at is None:
                    collected_at = datetime.now(timezone.utc)
                
                sentence = RawSentence(
                    text=obj["text"],
                    source_url=obj.get("source_url", ""),
                    collected_at=collected_at,
                    platform=platform,  # type: ignore[arg-type]
                    sentence_id=obj.get("sentence_id", ""),
                )
                sentences.append(sentence)
            except Exception as exc:
                logger.warning(
                    "Failed to parse line %d in %s: %s",
                    line_num, path.name, exc
                )
                continue
    
    logger.info("Loaded %d sentences from %s (platform=%s)", len(sentences), path.name, inferred_platform)
    return sentences


def merge_jsonl_files(
    input_paths: list[Path],
    output_path: Path,
    override_platform: str | None = None,
) -> int:
    """Merge multiple JSONL files into a single output file.

    Args:
        input_paths: List of input JSONL file paths.
        output_path: Path to the output merged JSONL file.
        override_platform: If provided, override platform for all sentences.

    Returns:
        Total number of sentences written.
    """
    total = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for path in input_paths:
            if not path.exists():
                logger.warning("File not found: %s — skipping", path)
                continue
            
            sentences = load_from_jsonl(path, override_platform=override_platform)
            for sentence in sentences:
                obj = {
                    "text": sentence.text,
                    "source_url": sentence.source_url,
                    "collected_at": sentence.collected_at.isoformat(),
                    "platform": sentence.platform,
                    "sentence_id": sentence.sentence_id,
                }
                out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total += 1
    
    logger.info("Merged %d sentences into %s", total, output_path)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    """Main entry point for manual importer."""
    parser = argparse.ArgumentParser(
        description="Merge existing JSONL data files into a unified corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="Input JSONL files to merge. If not specified, auto-discovers all data/*.jsonl files.",
    )
    parser.add_argument(
        "--output",
        default="data/merged_corpus.jsonl",
        help="Output path for the merged corpus.",
    )
    parser.add_argument(
        "--override-platform",
        default=None,
        help="Override platform label for all sentences (optional).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    
    args = parser.parse_args(argv)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    
    # Auto-discover input files if not specified
    if args.input:
        input_paths = [Path(p) for p in args.input]
    else:
        data_dir = Path("data")
        # Prioritize the largest/most recent files
        input_paths = [
            data_dir / "bhaskar_xlarge.jsonl",
            data_dir / "bhaskar_run2.jsonl",
            data_dir / "bhaskar_large.jsonl",
            data_dir / "bhaskar_raw.jsonl",
            data_dir / "patrika_large.jsonl",
            data_dir / "patrika_xlarge.jsonl",
            data_dir / "patrika_full.jsonl",
            data_dir / "patrika_run2.jsonl",
            data_dir / "patrika_raw.jsonl",
            data_dir / "rajasthani_lock.jsonl",
            data_dir / "rajasthani_vyakaran.jsonl",
            data_dir / "vyakaran_kaumudi.jsonl",
            data_dir / "marwari_gusain.jsonl",
            data_dir / "loka_sahitya.jsonl",
            data_dir / "rj06_part1.jsonl",
            data_dir / "rj06_part2.jsonl",
            data_dir / "rj06_part3.jsonl",
            data_dir / "rj06_part4.jsonl",
            data_dir / "wikipedia_gov_raw.jsonl",
            data_dir / "internet_pdfs_raw.jsonl",
            data_dir / "pdf_Rajasthani_Loka_Sahitya.jsonl",
            data_dir / "pdf_Rajasthani_Loka_Sahitya_2.jsonl",
        ]
        # Filter to only existing files
        input_paths = [p for p in input_paths if p.exists()]
        logger.info("Auto-discovered %d input files", len(input_paths))
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = merge_jsonl_files(
        input_paths=input_paths,
        output_path=output_path,
        override_platform=args.override_platform,
    )
    
    logger.info("✓ Merged corpus written to %s (%d sentences)", output_path, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
