#!/usr/bin/env python3
"""
Build the RajNLP-50K corpus from existing local data files.

This script replaces the Twitter/X and ShareChat API collectors with
local data sources that are already collected:

  - Dainik Bhaskar news articles  → platform="news_bhaskar"
  - Patrika news articles          → platform="news_patrika"
  - Books / PDFs                   → platform="books"
  - Wikipedia / government text    → platform="wikipedia"

Runs the full pipeline:
  load → filter → deduplicate → stratified_sample (50K) → split → serialize

Usage:
    python -m corpus_builder.build_corpus [--output-dir output] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from corpus_builder.filter_dedup import filter_rajasthani, deduplicate
from corpus_builder.manual_importer import load_from_jsonl
from corpus_builder.sampling import stratified_sample, split, InsufficientDataError
from corpus_builder.serialization import serialize, validate_round_trip
from models.data_models import RawSentence
from models.reproducibility import set_all_seeds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source file registry
# ---------------------------------------------------------------------------

# Each entry: (path, platform_label)
# Ordered by quality/size preference — larger, cleaner sources first.
SOURCE_FILES: list[tuple[str, str]] = [
    # Dainik Bhaskar — Rajasthan regional news
    ("data/bhaskar_xlarge.jsonl",   "news_bhaskar"),
    ("data/bhaskar_run2.jsonl",     "news_bhaskar"),
    ("data/bhaskar_large.jsonl",    "news_bhaskar"),
    ("data/bhaskar_raw.jsonl",      "news_bhaskar"),
    # Patrika — Rajasthan regional news
    ("data/patrika_large.jsonl",    "news_patrika"),
    ("data/patrika_xlarge.jsonl",   "news_patrika"),
    ("data/patrika_full.jsonl",     "news_patrika"),
    ("data/patrika_run2.jsonl",     "news_patrika"),
    ("data/patrika_raw.jsonl",      "news_patrika"),
    # Books and literary texts
    ("data/rajasthani_lock.jsonl",          "books"),
    ("data/rajasthani_vyakaran.jsonl",      "books"),
    ("data/vyakaran_kaumudi.jsonl",         "books"),
    ("data/marwari_gusain.jsonl",           "books"),
    ("data/loka_sahitya.jsonl",             "books"),
    ("data/rj06_part1.jsonl",               "books"),
    ("data/rj06_part2.jsonl",               "books"),
    ("data/rj06_part3.jsonl",               "books"),
    ("data/rj06_part4.jsonl",               "books"),
    ("data/pdf_Rajasthani_Loka_Sahitya.jsonl",   "books"),
    ("data/pdf_Rajasthani_Loka_Sahitya_2.jsonl", "books"),
    ("data/internet_pdfs_raw.jsonl",        "books"),
    # Wikipedia / government
    ("data/wikipedia_gov_raw.jsonl", "wikipedia"),
]


def load_all_sources() -> list[RawSentence]:
    """Load all source files with correct platform labels.

    Returns:
        Combined list of all :class:`~models.data_models.RawSentence` objects.
    """
    all_sentences: list[RawSentence] = []
    for rel_path, platform in SOURCE_FILES:
        path = Path(rel_path)
        if not path.exists():
            logger.warning("Source file not found, skipping: %s", path)
            continue
        sentences = load_from_jsonl(path, override_platform=platform)
        all_sentences.extend(sentences)
        logger.info("  %-45s %6d sentences  platform=%s", path.name, len(sentences), platform)
    return all_sentences


def build_corpus(
    output_dir: Path,
    seed: int = 42,
    target_n: int = 50_000,
    skip_minhash: bool = False,
) -> dict:
    """Run the full corpus building pipeline.

    Args:
        output_dir: Directory to write output files.
        seed: Random seed for reproducibility.
        target_n: Target corpus size (default 50,000).

    Returns:
        A dict with pipeline statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)

    # --- Load ---
    logger.info("=== Step 1: Loading source files ===")
    raw = load_all_sources()
    logger.info("Total loaded: %d sentences", len(raw))

    platform_counts = Counter(s.platform for s in raw)
    logger.info("Platform breakdown (raw): %s", dict(platform_counts))

    # --- Filter ---
    logger.info("=== Step 2: Filtering for Rajasthani content ===")
    filtered = filter_rajasthani(raw)
    logger.info("After filter: %d sentences (removed %d)", len(filtered), len(raw) - len(filtered))

    # --- Deduplicate ---
    logger.info("=== Step 3: Deduplicating ===")
    if skip_minhash:
        # Fast path: exact dedup only (NFC string match)
        import unicodedata
        seen: set[str] = set()
        deduped = []
        for s in filtered:
            key = unicodedata.normalize("NFC", s.text)
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        logger.info("After exact dedup (no MinHash): %d sentences (removed %d)", len(deduped), len(filtered) - len(deduped))
    else:
        deduped = deduplicate(filtered)
        logger.info("After dedup: %d sentences (removed %d)", len(deduped), len(filtered) - len(deduped))

    platform_counts_deduped = Counter(s.platform for s in deduped)
    logger.info("Platform breakdown (deduped): %s", dict(platform_counts_deduped))

    # --- Sample ---
    logger.info("=== Step 4: Stratified sampling to %d ===", target_n)
    if len(deduped) >= target_n:
        try:
            sampled = stratified_sample(deduped, n=target_n)
            logger.info("Sampled: %d sentences", len(sampled))
        except InsufficientDataError as exc:
            logger.error("Sampling failed: %s", exc)
            sampled = deduped
    else:
        logger.warning(
            "Only %d sentences after dedup (target %d) — using all",
            len(deduped), target_n
        )
        sampled = deduped

    platform_counts_sampled = Counter(s.platform for s in sampled)
    logger.info("Platform breakdown (sampled): %s", dict(platform_counts_sampled))

    # --- Split ---
    logger.info("=== Step 5: Splitting 80/10/10 ===")
    dataset_split = split(sampled)
    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(dataset_split.train),
        len(dataset_split.validation),
        len(dataset_split.test),
    )

    # --- Save raw split as JSONL for inspection ---
    raw_output = output_dir / "corpus_raw_split.jsonl"
    logger.info("=== Step 6: Saving raw split to %s ===", raw_output)
    with raw_output.open("w", encoding="utf-8") as fh:
        for partition, sentences in [
            ("train", dataset_split.train),
            ("validation", dataset_split.validation),
            ("test", dataset_split.test),
        ]:
            for s in sentences:
                obj = {
                    "sentence_id": s.sentence_id,
                    "text": s.text,
                    "platform": s.platform,
                    "split": partition,
                    "source_url": s.source_url,
                    "collected_at": s.collected_at.isoformat(),
                }
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(sampled), raw_output)

    # --- Stats ---
    stats = {
        "seed": seed,
        "total_loaded": len(raw),
        "after_filter": len(filtered),
        "after_dedup": len(deduped),
        "sampled": len(sampled),
        "train": len(dataset_split.train),
        "validation": len(dataset_split.validation),
        "test": len(dataset_split.test),
        "platform_breakdown_raw": dict(platform_counts),
        "platform_breakdown_final": dict(platform_counts_sampled),
    }

    stats_path = output_dir / "corpus_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info("Stats written to %s", stats_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build RajNLP-50K corpus from local data files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--target-n", type=int, default=50_000, help="Target corpus size.")
    parser.add_argument(
        "--skip-minhash",
        action="store_true",
        help="Skip MinHash LSH near-duplicate pass (faster; use when exact dedup is sufficient).",
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

    stats = build_corpus(
        output_dir=Path(args.output_dir),
        seed=args.seed,
        target_n=args.target_n,
        skip_minhash=args.skip_minhash,
    )

    print("\n=== CORPUS BUILD COMPLETE ===")
    print(f"  Loaded:        {stats['total_loaded']:>8,}")
    print(f"  After filter:  {stats['after_filter']:>8,}")
    print(f"  After dedup:   {stats['after_dedup']:>8,}")
    print(f"  Sampled:       {stats['sampled']:>8,}")
    print(f"  Train:         {stats['train']:>8,}")
    print(f"  Validation:    {stats['validation']:>8,}")
    print(f"  Test:          {stats['test']:>8,}")
    print(f"\n  Platform breakdown (final):")
    for plat, count in sorted(stats['platform_breakdown_final'].items()):
        pct = count / stats['sampled'] * 100
        print(f"    {plat:<20} {count:>6,}  ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
