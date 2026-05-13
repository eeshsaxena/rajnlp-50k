"""
Tests for corpus_builder.sampling — stratified sampling and train/val/test splitting.

Covers:
- Property 3: Stratified sampling preserves platform distribution (±1%)
  (Validates: Requirement 3.2)
- Property 4: Train/val/test split is exhaustive and disjoint
  (Validates: Requirements 3.3, 3.4)
- Unit tests: exact 50K sample size, exact 40K/5K/5K split sizes
  (Validates: Requirements 3.1, 3.3)

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from corpus_builder.sampling import (
    DEFAULT_RATIOS,
    DEFAULT_TARGET_N,
    InsufficientDataError,
    split,
    stratified_sample,
)
from models.data_models import RawSentence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


def _make_sentence(
    platform: str = "twitter",
    text: str = "म्हारो राजस्थान",
    sentence_id: str | None = None,
) -> RawSentence:
    return RawSentence(
        text=text,
        source_url="https://example.com",
        collected_at=_FIXED_TS,
        platform=platform,  # type: ignore[arg-type]
        sentence_id=sentence_id or str(uuid.uuid4()),
    )


def _make_pool(
    n_twitter: int,
    n_sharechat: int,
    seed: int = 0,
) -> list[RawSentence]:
    """Build a pool with the given per-platform counts."""
    sentences: list[RawSentence] = []
    for i in range(n_twitter):
        sentences.append(_make_sentence(platform="twitter", text=f"twitter text {i}"))
    for i in range(n_sharechat):
        sentences.append(_make_sentence(platform="sharechat", text=f"sharechat text {i}"))
    return sentences


def _platform_proportion(sentences: list[RawSentence], platform: str) -> float:
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if s.platform == platform) / len(sentences)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def _pool_strategy(draw, min_total: int = 10, max_total: int = 500):
    """Generate a pool with a random split between twitter and sharechat."""
    total = draw(st.integers(min_value=min_total, max_value=max_total))
    n_twitter = draw(st.integers(min_value=1, max_value=total - 1))
    n_sharechat = total - n_twitter
    return _make_pool(n_twitter, n_sharechat)


# ---------------------------------------------------------------------------
# Property 3: Stratified sampling preserves platform distribution (±1%)
# ---------------------------------------------------------------------------


class TestProperty3StratifiedSamplingDistribution:
    """
    Property 3: For any collection of sentences with a known platform
    distribution, stratified_sample SHALL produce a subset whose platform
    distribution is within ±1% of the original.

    Validates: Requirement 3.2
    """

    @given(pool=_pool_strategy(min_total=20, max_total=300))
    @settings(max_examples=100)
    def test_platform_distribution_preserved_within_1_percent(self, pool):
        """
        GIVEN a pool of sentences with any platform distribution,
        WHEN  stratified_sample draws n ≤ len(pool) sentences,
        THEN  the platform proportions in the sample are within ±1% of the pool.
        """
        n = max(2, len(pool) // 2)  # sample half the pool
        sample = stratified_sample(pool, n=n, seed=42)

        assert len(sample) == n

        for platform in ("twitter", "sharechat"):
            pool_prop = _platform_proportion(pool, platform)
            sample_prop = _platform_proportion(sample, platform)
            assert abs(pool_prop - sample_prop) <= 0.01 + (1 / n), (
                f"Platform '{platform}' distribution drifted: "
                f"pool={pool_prop:.3f} sample={sample_prop:.3f}"
            )

    @given(pool=_pool_strategy(min_total=20, max_total=300))
    @settings(max_examples=100)
    def test_sample_size_is_exactly_n(self, pool):
        """
        GIVEN any pool with at least n sentences,
        WHEN  stratified_sample is called with target n,
        THEN  the result contains exactly n sentences.
        """
        n = max(2, len(pool) // 3)
        sample = stratified_sample(pool, n=n, seed=42)
        assert len(sample) == n

    @given(pool=_pool_strategy(min_total=20, max_total=300))
    @settings(max_examples=100)
    def test_sample_is_subset_of_pool(self, pool):
        """
        GIVEN any pool,
        WHEN  stratified_sample is called,
        THEN  every sentence in the sample was in the pool.
        """
        n = max(2, len(pool) // 2)
        sample = stratified_sample(pool, n=n, seed=42)
        pool_ids = {s.sentence_id for s in pool}
        for s in sample:
            assert s.sentence_id in pool_ids

    @given(pool=_pool_strategy(min_total=20, max_total=300))
    @settings(max_examples=100)
    def test_sample_has_no_duplicate_sentence_ids(self, pool):
        """
        GIVEN any pool,
        WHEN  stratified_sample is called,
        THEN  the sample contains no duplicate sentence_ids.
        """
        n = max(2, len(pool) // 2)
        sample = stratified_sample(pool, n=n, seed=42)
        ids = [s.sentence_id for s in sample]
        assert len(ids) == len(set(ids))

    @given(pool=_pool_strategy(min_total=20, max_total=300))
    @settings(max_examples=100)
    def test_same_seed_produces_same_sample(self, pool):
        """
        GIVEN the same pool and seed,
        WHEN  stratified_sample is called twice,
        THEN  both calls return the same sentence_ids in the same order.
        """
        n = max(2, len(pool) // 2)
        sample1 = stratified_sample(pool, n=n, seed=99)
        sample2 = stratified_sample(pool, n=n, seed=99)
        assert [s.sentence_id for s in sample1] == [s.sentence_id for s in sample2]


# ---------------------------------------------------------------------------
# Property 4: Train/val/test split is exhaustive and disjoint
# ---------------------------------------------------------------------------


class TestProperty4SplitExhaustiveDisjoint:
    """
    Property 4: For any corpus of N sentences, splitting at 80/10/10 SHALL
    produce three partitions whose sizes sum to N, and whose sentence_id sets
    are pairwise disjoint.

    Validates: Requirements 3.3, 3.4
    """

    @given(pool=_pool_strategy(min_total=10, max_total=300))
    @settings(max_examples=100)
    def test_split_is_exhaustive(self, pool):
        """
        GIVEN any pool of N sentences,
        WHEN  split is called,
        THEN  len(train) + len(val) + len(test) == N.
        """
        result = split(pool, seed=42)
        total = len(result.train) + len(result.validation) + len(result.test)
        assert total == len(pool), (
            f"Split is not exhaustive: {len(result.train)} + "
            f"{len(result.validation)} + {len(result.test)} = {total} ≠ {len(pool)}"
        )

    @given(pool=_pool_strategy(min_total=10, max_total=300))
    @settings(max_examples=100)
    def test_split_is_disjoint(self, pool):
        """
        GIVEN any pool of N sentences,
        WHEN  split is called,
        THEN  no sentence_id appears in more than one partition.
        """
        result = split(pool, seed=42)
        train_ids = {s.sentence_id for s in result.train}
        val_ids = {s.sentence_id for s in result.validation}
        test_ids = {s.sentence_id for s in result.test}

        assert not (train_ids & val_ids), "train and val share sentence_ids"
        assert not (train_ids & test_ids), "train and test share sentence_ids"
        assert not (val_ids & test_ids), "val and test share sentence_ids"

    @given(pool=_pool_strategy(min_total=10, max_total=300))
    @settings(max_examples=100)
    def test_split_covers_all_input_ids(self, pool):
        """
        GIVEN any pool,
        WHEN  split is called,
        THEN  the union of all partition sentence_ids equals the pool sentence_ids.
        """
        result = split(pool, seed=42)
        pool_ids = {s.sentence_id for s in pool}
        split_ids = (
            {s.sentence_id for s in result.train}
            | {s.sentence_id for s in result.validation}
            | {s.sentence_id for s in result.test}
        )
        assert pool_ids == split_ids

    @given(pool=_pool_strategy(min_total=10, max_total=300))
    @settings(max_examples=100)
    def test_same_seed_produces_same_split(self, pool):
        """
        GIVEN the same pool and seed,
        WHEN  split is called twice,
        THEN  both calls produce identical partition sentence_id lists.
        """
        r1 = split(pool, seed=7)
        r2 = split(pool, seed=7)
        assert [s.sentence_id for s in r1.train] == [s.sentence_id for s in r2.train]
        assert [s.sentence_id for s in r1.validation] == [s.sentence_id for s in r2.validation]
        assert [s.sentence_id for s in r1.test] == [s.sentence_id for s in r2.test]

    @given(pool=_pool_strategy(min_total=30, max_total=300))
    @settings(max_examples=100)
    def test_train_is_largest_partition(self, pool):
        """
        GIVEN any pool with at least 30 sentences,
        WHEN  split is called with default 80/10/10 ratios,
        THEN  the train partition is larger than val and test.
        """
        result = split(pool, seed=42)
        assert len(result.train) >= len(result.validation)
        assert len(result.train) >= len(result.test)


# ---------------------------------------------------------------------------
# Unit tests — stratified_sample
# ---------------------------------------------------------------------------


class TestStratifiedSampleUnit:
    """Unit tests for stratified_sample with concrete inputs."""

    def test_exact_50k_sample_from_large_pool(self):
        """
        GIVEN a pool of 60,000 sentences (40K twitter, 20K sharechat),
        WHEN  stratified_sample is called with n=50_000,
        THEN  exactly 50,000 sentences are returned.
        """
        pool = _make_pool(n_twitter=40_000, n_sharechat=20_000)
        sample = stratified_sample(pool, n=50_000, seed=42)
        assert len(sample) == 50_000

    def test_platform_proportions_match_pool(self):
        """
        GIVEN a pool with 70% twitter and 30% sharechat,
        WHEN  stratified_sample draws 1,000 sentences,
        THEN  the sample has approximately 70% twitter and 30% sharechat (±1%).
        """
        pool = _make_pool(n_twitter=700, n_sharechat=300)
        sample = stratified_sample(pool, n=1_000, seed=42)

        twitter_prop = _platform_proportion(sample, "twitter")
        sharechat_prop = _platform_proportion(sample, "sharechat")

        assert abs(twitter_prop - 0.70) <= 0.01, f"twitter proportion={twitter_prop:.3f}"
        assert abs(sharechat_prop - 0.30) <= 0.01, f"sharechat proportion={sharechat_prop:.3f}"

    def test_insufficient_data_raises_error(self):
        """
        GIVEN a pool of 100 sentences,
        WHEN  stratified_sample is called with n=200,
        THEN  InsufficientDataError is raised.
        """
        pool = _make_pool(n_twitter=60, n_sharechat=40)
        with pytest.raises(InsufficientDataError) as exc_info:
            stratified_sample(pool, n=200)
        assert "shortfall" in str(exc_info.value).lower()

    def test_insufficient_data_logs_shortfall(self, caplog):
        """
        GIVEN a pool smaller than n,
        WHEN  stratified_sample is called,
        THEN  an ERROR log entry mentioning the shortfall is emitted.
        """
        import logging
        pool = _make_pool(n_twitter=30, n_sharechat=20)
        with caplog.at_level(logging.ERROR, logger="corpus_builder.sampling"):
            with pytest.raises(InsufficientDataError):
                stratified_sample(pool, n=100)
        assert any("shortfall" in r.message.lower() for r in caplog.records)

    def test_sample_contains_no_duplicates(self):
        """
        GIVEN a pool with all unique sentences,
        WHEN  stratified_sample is called,
        THEN  the sample contains no duplicate sentence_ids.
        """
        pool = _make_pool(n_twitter=300, n_sharechat=200)
        sample = stratified_sample(pool, n=400, seed=42)
        ids = [s.sentence_id for s in sample]
        assert len(ids) == len(set(ids))

    def test_different_seeds_produce_different_samples(self):
        """
        GIVEN the same pool,
        WHEN  stratified_sample is called with two different seeds,
        THEN  the resulting sentence_id lists differ.
        """
        pool = _make_pool(n_twitter=300, n_sharechat=200)
        s1 = [s.sentence_id for s in stratified_sample(pool, n=400, seed=1)]
        s2 = [s.sentence_id for s in stratified_sample(pool, n=400, seed=2)]
        assert s1 != s2

    def test_sample_equal_to_pool_size_returns_all(self):
        """
        GIVEN a pool of exactly n sentences,
        WHEN  stratified_sample is called with n equal to pool size,
        THEN  all sentences are returned (no shortfall).
        """
        pool = _make_pool(n_twitter=30, n_sharechat=20)
        sample = stratified_sample(pool, n=50, seed=42)
        assert len(sample) == 50
        pool_ids = {s.sentence_id for s in pool}
        sample_ids = {s.sentence_id for s in sample}
        assert pool_ids == sample_ids

    def test_source_platform_recorded_for_every_sentence(self):
        """
        GIVEN a sample drawn from a mixed pool,
        WHEN  stratified_sample returns results,
        THEN  every sentence has a non-empty platform field.
        """
        pool = _make_pool(n_twitter=300, n_sharechat=200)
        sample = stratified_sample(pool, n=400, seed=42)
        for s in sample:
            assert s.platform in ("twitter", "sharechat")


# ---------------------------------------------------------------------------
# Unit tests — split
# ---------------------------------------------------------------------------


class TestSplitUnit:
    """Unit tests for split with concrete inputs."""

    def test_split_sizes_are_40k_5k_5k(self):
        """
        GIVEN a pool of exactly 50,000 sentences,
        WHEN  split is called with default 80/10/10 ratios,
        THEN  train=40,000, val=5,000, test=5,000.
        """
        pool = _make_pool(n_twitter=30_000, n_sharechat=20_000)
        result = split(pool, seed=42)
        assert len(result.train) == 40_000
        assert len(result.validation) == 5_000
        assert len(result.test) == 5_000

    def test_no_sentence_in_multiple_partitions(self):
        """
        GIVEN a pool of 1,000 sentences,
        WHEN  split is called,
        THEN  no sentence_id appears in more than one partition.
        """
        pool = _make_pool(n_twitter=600, n_sharechat=400)
        result = split(pool, seed=42)
        train_ids = {s.sentence_id for s in result.train}
        val_ids = {s.sentence_id for s in result.validation}
        test_ids = {s.sentence_id for s in result.test}
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_all_sentences_appear_in_exactly_one_partition(self):
        """
        GIVEN a pool of 500 sentences,
        WHEN  split is called,
        THEN  every sentence appears in exactly one partition.
        """
        pool = _make_pool(n_twitter=300, n_sharechat=200)
        result = split(pool, seed=42)
        all_ids = (
            [s.sentence_id for s in result.train]
            + [s.sentence_id for s in result.validation]
            + [s.sentence_id for s in result.test]
        )
        assert len(all_ids) == len(pool)
        assert len(set(all_ids)) == len(pool)

    def test_invalid_ratios_raises_value_error(self):
        """
        GIVEN ratios that do not sum to 1.0,
        WHEN  split is called,
        THEN  ValueError is raised.
        """
        pool = _make_pool(n_twitter=60, n_sharechat=40)
        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            split(pool, ratios=(0.7, 0.2, 0.2))

    def test_split_is_reproducible_with_same_seed(self):
        """
        GIVEN the same pool and seed,
        WHEN  split is called twice,
        THEN  both results are identical.
        """
        pool = _make_pool(n_twitter=300, n_sharechat=200)
        r1 = split(pool, seed=42)
        r2 = split(pool, seed=42)
        assert [s.sentence_id for s in r1.train] == [s.sentence_id for s in r2.train]
        assert [s.sentence_id for s in r1.validation] == [s.sentence_id for s in r2.validation]
        assert [s.sentence_id for s in r1.test] == [s.sentence_id for s in r2.test]

    def test_split_returns_dataset_split_object(self):
        """
        GIVEN a pool of sentences,
        WHEN  split is called,
        THEN  the return value is a DatasetSplit instance.
        """
        from models.data_models import DatasetSplit
        pool = _make_pool(n_twitter=60, n_sharechat=40)
        result = split(pool, seed=42)
        assert isinstance(result, DatasetSplit)

    def test_small_pool_split_is_exhaustive(self):
        """
        GIVEN a small pool of 10 sentences,
        WHEN  split is called,
        THEN  all 10 sentences appear across the three partitions.
        """
        pool = _make_pool(n_twitter=6, n_sharechat=4)
        result = split(pool, seed=42)
        total = len(result.train) + len(result.validation) + len(result.test)
        assert total == 10
