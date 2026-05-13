"""
Corpus_Builder — stratified sampling and train/val/test splitting for RajNLP-50K.

Provides two public functions:

- ``stratified_sample``: Draws exactly ``n`` sentences from a pool while
  preserving the per-platform distribution (Requirements 3.1, 3.2, 3.5).

- ``split``: Splits a list of sentences into train / validation / test
  partitions at an 80/10/10 ratio using a stratified random split, ensuring
  no sentence appears in more than one partition (Requirements 3.3, 3.4).
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from typing import Literal

from models.data_models import AnnotatedSentence, DatasetSplit, RawSentence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default target corpus size (Requirements 3.1).
DEFAULT_TARGET_N: int = 50_000

#: Default train / validation / test split ratios (Requirements 3.3).
DEFAULT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)

#: Train partition size when using DEFAULT_TARGET_N and DEFAULT_RATIOS.
TRAIN_SIZE: int = 40_000

#: Validation partition size.
VAL_SIZE: int = 5_000

#: Test partition size.
TEST_SIZE: int = 5_000


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


class InsufficientDataError(ValueError):
    """Raised when the collected pool is smaller than the requested sample size."""


def stratified_sample(
    sentences: list[RawSentence],
    n: int = DEFAULT_TARGET_N,
    seed: int = 42,
) -> list[RawSentence]:
    """Draw exactly *n* sentences from *sentences* preserving platform distribution.

    Algorithm:
        1. Group sentences by ``platform``.
        2. Compute each platform's proportion of the total pool.
        3. Allocate ``floor(proportion × n)`` slots per platform.
        4. Distribute any remaining slots (due to rounding) to the largest
           platforms first, so the total is exactly *n*.
        5. Draw a random sample of the allocated size from each platform stratum
           using *seed* for reproducibility.

    Args:
        sentences: Full collected pool of :class:`~models.data_models.RawSentence`
            objects (after filtering and deduplication).
        n: Target sample size.  Defaults to 50,000.
        seed: Random seed for reproducible sampling.  Defaults to 42.

    Returns:
        A list of exactly *n* :class:`~models.data_models.RawSentence` objects
        drawn proportionally from each platform stratum.

    Raises:
        InsufficientDataError: If ``len(sentences) < n``.
    """
    total = len(sentences)
    if total < n:
        shortfall = n - total
        logger.error(
            "stratified_sample: insufficient data — collected %d sentences, "
            "need %d (shortfall=%d)",
            total,
            n,
            shortfall,
        )
        raise InsufficientDataError(
            f"Cannot sample {n} sentences from a pool of {total} "
            f"(shortfall={shortfall}). Collect more data before sampling."
        )

    # Group by platform
    strata: dict[str, list[RawSentence]] = defaultdict(list)
    for sentence in sentences:
        strata[sentence.platform].append(sentence)

    platforms = sorted(strata.keys())  # deterministic ordering
    platform_counts = {p: len(strata[p]) for p in platforms}

    # Compute proportional allocations (floor)
    allocations: dict[str, int] = {}
    for platform in platforms:
        proportion = platform_counts[platform] / total
        allocations[platform] = math.floor(proportion * n)

    # Distribute remainder to largest platforms first
    remainder = n - sum(allocations.values())
    platforms_by_size = sorted(platforms, key=lambda p: platform_counts[p], reverse=True)
    for i in range(remainder):
        allocations[platforms_by_size[i % len(platforms_by_size)]] += 1

    assert sum(allocations.values()) == n, "Allocation arithmetic error"

    # Sample from each stratum
    rng = random.Random(seed)
    sampled: list[RawSentence] = []
    for platform in platforms:
        k = allocations[platform]
        stratum = strata[platform]
        drawn = rng.sample(stratum, k)
        sampled.extend(drawn)
        logger.info(
            "stratified_sample: platform=%s pool=%d allocated=%d (%.1f%%)",
            platform,
            platform_counts[platform],
            k,
            100.0 * k / n,
        )

    logger.info(
        "stratified_sample: total sampled=%d (target=%d)",
        len(sampled),
        n,
    )
    return sampled


# ---------------------------------------------------------------------------
# Train / validation / test splitting
# ---------------------------------------------------------------------------


def split(
    sentences: list[RawSentence],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> DatasetSplit:
    """Split *sentences* into train / validation / test partitions.

    The split is stratified by platform so that each partition preserves the
    platform distribution of the input.  No sentence appears in more than one
    partition (disjoint ``sentence_id`` sets).

    Partition sizes are computed as ``floor(ratio × total)`` with any remainder
    assigned to the train partition.

    Args:
        sentences: List of :class:`~models.data_models.RawSentence` objects to
            split.  Typically the output of :func:`stratified_sample`.
        ratios: A 3-tuple ``(train_ratio, val_ratio, test_ratio)`` that must
            sum to 1.0.  Defaults to ``(0.8, 0.1, 0.1)``.
        seed: Random seed for reproducible shuffling.  Defaults to 42.

    Returns:
        A :class:`~models.data_models.DatasetSplit` whose ``train``,
        ``validation``, and ``test`` fields contain
        :class:`~models.data_models.RawSentence` objects.

        .. note::
            The ``DatasetSplit`` dataclass is defined with
            ``list[AnnotatedSentence]`` fields, but at this pipeline stage the
            sentences are still ``RawSentence`` objects (annotation happens
            later).  The split is returned as a ``DatasetSplit`` with the
            ``train``/``validation``/``test`` lists populated with
            ``RawSentence`` objects cast to the field type.  Downstream code
            that annotates the corpus will replace these with
            ``AnnotatedSentence`` objects.

    Raises:
        ValueError: If ``ratios`` does not sum to approximately 1.0.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"ratios must sum to 1.0; got {train_ratio + val_ratio + test_ratio}"
        )

    total = len(sentences)

    # Stratify by platform to preserve distribution within each partition
    strata: dict[str, list[RawSentence]] = defaultdict(list)
    for sentence in sentences:
        strata[sentence.platform].append(sentence)

    rng = random.Random(seed)

    train_sentences: list[RawSentence] = []
    val_sentences: list[RawSentence] = []
    test_sentences: list[RawSentence] = []

    for platform, stratum in sorted(strata.items()):
        shuffled = list(stratum)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_val = math.floor(val_ratio * n)
        n_test = math.floor(test_ratio * n)
        n_train = n - n_val - n_test  # remainder goes to train

        train_sentences.extend(shuffled[:n_train])
        val_sentences.extend(shuffled[n_train : n_train + n_val])
        test_sentences.extend(shuffled[n_train + n_val :])

    # Final shuffle within each partition so platform order is mixed
    rng.shuffle(train_sentences)
    rng.shuffle(val_sentences)
    rng.shuffle(test_sentences)

    logger.info(
        "split: train=%d val=%d test=%d (total=%d)",
        len(train_sentences),
        len(val_sentences),
        len(test_sentences),
        total,
    )

    # Verify disjointness
    train_ids = {s.sentence_id for s in train_sentences}
    val_ids = {s.sentence_id for s in val_sentences}
    test_ids = {s.sentence_id for s in test_sentences}
    assert not (train_ids & val_ids), "train/val overlap detected"
    assert not (train_ids & test_ids), "train/test overlap detected"
    assert not (val_ids & test_ids), "val/test overlap detected"

    # DatasetSplit expects AnnotatedSentence; cast at runtime (annotation fills these later)
    return DatasetSplit(
        train=train_sentences,      # type: ignore[arg-type]
        validation=val_sentences,   # type: ignore[arg-type]
        test=test_sentences,        # type: ignore[arg-type]
    )
