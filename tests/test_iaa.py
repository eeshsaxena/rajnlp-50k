"""
Tests for annotator_tool.iaa and annotator_tool.majority_vote.

Covers:
- Property 9: IAA threshold flagging
  For any annotated sentence batch and any annotation task, if the computed
  Cohen's Kappa falls below the task-specific threshold, the batch SHALL be
  flagged; if κ is at or above the threshold, the batch SHALL NOT be flagged.
  Thresholds: 0.72 (sentiment), 0.78 (NER), 0.65 (toxicity)
  (Validates: Requirements 5.4, 6.4, 7.5)

- Property 10: Majority vote correctness
  For any set of 3 annotator labels, the majority-vote function SHALL return
  the label that appears in at least 2 of the 3 annotations, and SHALL never
  return a label that appears in fewer than 2 annotations.
  (Validates: Requirements 5.2, 6.2)

- Unit tests for IAA threshold flagging (Requirements 5.4, 6.4)
- Unit tests for majority vote (Requirements 5.2, 6.2)

Requirements: 5.2, 5.4, 6.2, 6.4, 7.5
"""

from __future__ import annotations

import logging
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from annotator_tool.iaa import (
    NER_KAPPA_THRESHOLD,
    SENTIMENT_KAPPA_THRESHOLD,
    TOXICITY_KAPPA_THRESHOLD,
    IAAResult,
    flag_batch,
)
from annotator_tool.majority_vote import (
    majority_vote_ner,
    majority_vote_sentiment,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_SENTIMENT_LABELS = ["positive", "neutral", "negative"]
_TOXICITY_CATS = ["caste_slur", "religious", "gender", "general"]
_ENTITY_TYPES = ["PER", "LOC", "ORG"]
_TASK_NAMES = ["sentiment", "ner", "toxicity"]
_TASK_THRESHOLDS = {
    "sentiment": SENTIMENT_KAPPA_THRESHOLD,
    "ner": NER_KAPPA_THRESHOLD,
    "toxicity": TOXICITY_KAPPA_THRESHOLD,
}


@st.composite
def _kappa_below_threshold(draw, task: str) -> float:
    """Generate a kappa value strictly below the task threshold."""
    threshold = _TASK_THRESHOLDS[task]
    # Generate a value in [-1.0, threshold)
    return draw(st.floats(
        min_value=-1.0,
        max_value=threshold - 1e-9,
        allow_nan=False,
        allow_infinity=False,
    ))


@st.composite
def _kappa_at_or_above_threshold(draw, task: str) -> float:
    """Generate a kappa value at or above the task threshold."""
    threshold = _TASK_THRESHOLDS[task]
    # Generate a value in [threshold, 1.0]
    return draw(st.floats(
        min_value=threshold,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ))


@st.composite
def _three_sentiment_labels(draw) -> list[str]:
    """Generate exactly 3 sentiment labels."""
    return draw(st.lists(
        st.sampled_from(_SENTIMENT_LABELS),
        min_size=3,
        max_size=3,
    ))


@st.composite
def _three_ner_span_sets(draw) -> list[list[tuple[int, int, str]]]:
    """Generate 3 annotator span sets for NER majority vote testing."""
    # Generate a small universe of possible spans
    n_possible_spans = draw(st.integers(min_value=0, max_value=4))
    possible_spans = []
    for _ in range(n_possible_spans):
        start = draw(st.integers(min_value=0, max_value=50))
        end = draw(st.integers(min_value=start + 1, max_value=start + 20))
        entity_type = draw(st.sampled_from(_ENTITY_TYPES))
        possible_spans.append((start, end, entity_type))

    # Each annotator picks a subset of the possible spans
    annotator_spans = []
    for _ in range(3):
        if possible_spans:
            chosen = draw(st.lists(
                st.sampled_from(possible_spans),
                min_size=0,
                max_size=len(possible_spans),
                unique=True,
            ))
        else:
            chosen = []
        annotator_spans.append(chosen)

    return annotator_spans


# ---------------------------------------------------------------------------
# Property 9: IAA threshold flagging
# ---------------------------------------------------------------------------


class TestProperty9IAAThresholdFlagging:
    """
    Property 9: IAA threshold flagging.

    For any annotated sentence batch and any annotation task, if the computed
    Cohen's Kappa falls below the task-specific threshold, the batch SHALL be
    flagged; if κ is at or above the threshold, the batch SHALL NOT be flagged.

    Validates: Requirements 5.4, 6.4, 7.5
    """

    @given(
        task=st.sampled_from(_TASK_NAMES),
        kappa=st.one_of(
            _kappa_below_threshold("sentiment"),
            _kappa_below_threshold("ner"),
            _kappa_below_threshold("toxicity"),
        ),
    )
    @settings(max_examples=100)
    def test_kappa_below_threshold_is_flagged(self, task, kappa):
        """
        GIVEN a kappa value strictly below the task-specific threshold,
        WHEN  flag_batch is called,
        THEN  the batch IS flagged (result.flagged == True).
        """
        threshold = _TASK_THRESHOLDS[task]
        # Only test when kappa is actually below this task's threshold
        if kappa >= threshold:
            return  # skip — kappa was drawn for a different task

        result = flag_batch("batch-001", task, kappa)  # type: ignore[arg-type]
        assert result.flagged is True, (
            f"Expected batch to be flagged for task={task}, "
            f"κ={kappa:.4f} < threshold={threshold:.2f}"
        )

    @given(
        task=st.sampled_from(_TASK_NAMES),
        kappa=st.one_of(
            _kappa_at_or_above_threshold("sentiment"),
            _kappa_at_or_above_threshold("ner"),
            _kappa_at_or_above_threshold("toxicity"),
        ),
    )
    @settings(max_examples=100)
    def test_kappa_at_or_above_threshold_is_not_flagged(self, task, kappa):
        """
        GIVEN a kappa value at or above the task-specific threshold,
        WHEN  flag_batch is called,
        THEN  the batch is NOT flagged (result.flagged == False).
        """
        threshold = _TASK_THRESHOLDS[task]
        # Only test when kappa is actually at or above this task's threshold
        if kappa < threshold:
            return  # skip — kappa was drawn for a different task

        result = flag_batch("batch-001", task, kappa)  # type: ignore[arg-type]
        assert result.flagged is False, (
            f"Expected batch NOT to be flagged for task={task}, "
            f"κ={kappa:.4f} >= threshold={threshold:.2f}"
        )

    @given(
        task=st.sampled_from(_TASK_NAMES),
        kappa=_kappa_below_threshold("sentiment"),
    )
    @settings(max_examples=100)
    def test_flagged_result_contains_correct_task_and_threshold(self, task, kappa):
        """
        GIVEN any task and kappa,
        WHEN  flag_batch is called,
        THEN  the result contains the correct task name and threshold.
        """
        result = flag_batch("batch-test", task, kappa)  # type: ignore[arg-type]
        assert result.task == task
        assert result.threshold == _TASK_THRESHOLDS[task]
        assert result.batch_id == "batch-test"
        assert result.kappa == kappa


# ---------------------------------------------------------------------------
# Property 9 — per-task targeted tests
# ---------------------------------------------------------------------------


class TestProperty9PerTask:
    """Per-task property tests for IAA threshold flagging."""

    @given(kappa=_kappa_below_threshold("sentiment"))
    @settings(max_examples=100)
    def test_sentiment_below_threshold_flagged(self, kappa):
        """κ < 0.72 for sentiment → flagged."""
        result = flag_batch("b", "sentiment", kappa)
        assert result.flagged is True

    @given(kappa=_kappa_at_or_above_threshold("sentiment"))
    @settings(max_examples=100)
    def test_sentiment_at_or_above_threshold_not_flagged(self, kappa):
        """κ >= 0.72 for sentiment → not flagged."""
        result = flag_batch("b", "sentiment", kappa)
        assert result.flagged is False

    @given(kappa=_kappa_below_threshold("ner"))
    @settings(max_examples=100)
    def test_ner_below_threshold_flagged(self, kappa):
        """κ < 0.78 for NER → flagged."""
        result = flag_batch("b", "ner", kappa)
        assert result.flagged is True

    @given(kappa=_kappa_at_or_above_threshold("ner"))
    @settings(max_examples=100)
    def test_ner_at_or_above_threshold_not_flagged(self, kappa):
        """κ >= 0.78 for NER → not flagged."""
        result = flag_batch("b", "ner", kappa)
        assert result.flagged is False

    @given(kappa=_kappa_below_threshold("toxicity"))
    @settings(max_examples=100)
    def test_toxicity_below_threshold_flagged(self, kappa):
        """κ < 0.65 for toxicity → flagged."""
        result = flag_batch("b", "toxicity", kappa)
        assert result.flagged is True

    @given(kappa=_kappa_at_or_above_threshold("toxicity"))
    @settings(max_examples=100)
    def test_toxicity_at_or_above_threshold_not_flagged(self, kappa):
        """κ >= 0.65 for toxicity → not flagged."""
        result = flag_batch("b", "toxicity", kappa)
        assert result.flagged is False


# ---------------------------------------------------------------------------
# Property 10: Majority vote correctness
# ---------------------------------------------------------------------------


class TestProperty10MajorityVoteCorrectness:
    """
    Property 10: Majority vote correctness.

    For any set of 3 annotator labels, the majority-vote function SHALL return
    the label that appears in at least 2 of the 3 annotations, and SHALL never
    return a label that appears in fewer than 2 annotations.

    Validates: Requirements 5.2, 6.2
    """

    @given(labels=_three_sentiment_labels())
    @settings(max_examples=100)
    def test_majority_vote_returns_label_with_count_ge_2(self, labels):
        """
        GIVEN any 3 sentiment labels,
        WHEN  majority_vote_sentiment is called,
        THEN  if a majority exists, the returned label appears ≥ 2 times.
        """
        result = majority_vote_sentiment(labels)
        counts = Counter(labels)

        if result is not None:
            assert counts[result] >= 2, (
                f"majority_vote_sentiment returned {result!r} which appears "
                f"only {counts[result]} time(s) in {labels}"
            )

    @given(labels=_three_sentiment_labels())
    @settings(max_examples=100)
    def test_majority_vote_never_returns_label_with_count_lt_2(self, labels):
        """
        GIVEN any 3 sentiment labels,
        WHEN  majority_vote_sentiment is called,
        THEN  the returned label (if any) NEVER appears fewer than 2 times.
        """
        result = majority_vote_sentiment(labels)
        counts = Counter(labels)

        if result is not None:
            assert counts[result] >= 2, (
                f"majority_vote_sentiment returned {result!r} which appears "
                f"only {counts[result]} time(s) — violates 'never < 2' rule"
            )

    @given(labels=_three_sentiment_labels())
    @settings(max_examples=100)
    def test_majority_vote_returns_none_when_no_majority(self, labels):
        """
        GIVEN 3 labels where no label appears ≥ 2 times,
        WHEN  majority_vote_sentiment is called,
        THEN  None is returned.
        """
        counts = Counter(labels)
        has_majority = any(c >= 2 for c in counts.values())

        result = majority_vote_sentiment(labels)

        if not has_majority:
            assert result is None, (
                f"Expected None for labels {labels} (no majority), got {result!r}"
            )

    @given(spans=_three_ner_span_sets())
    @settings(max_examples=100)
    def test_ner_majority_vote_only_includes_spans_with_count_ge_2(self, spans):
        """
        GIVEN any 3 annotator NER span sets,
        WHEN  majority_vote_ner is called,
        THEN  every span in the result was marked by ≥ 2 annotators.
        """
        result = majority_vote_ner(spans)

        # Count how many annotators marked each span
        span_counts: Counter[tuple] = Counter()
        for annotator_spans in spans:
            for span in set(annotator_spans):
                span_counts[span] += 1

        for span in result:
            assert span_counts[span] >= 2, (
                f"majority_vote_ner included span {span} which was marked by "
                f"only {span_counts[span]} annotator(s)"
            )

    @given(spans=_three_ner_span_sets())
    @settings(max_examples=100)
    def test_ner_majority_vote_never_includes_spans_with_count_lt_2(self, spans):
        """
        GIVEN any 3 annotator NER span sets,
        WHEN  majority_vote_ner is called,
        THEN  no span in the result was marked by fewer than 2 annotators.
        """
        result = majority_vote_ner(spans)

        span_counts: Counter[tuple] = Counter()
        for annotator_spans in spans:
            for span in set(annotator_spans):
                span_counts[span] += 1

        for span in result:
            assert span_counts[span] >= 2, (
                f"majority_vote_ner included span {span} with count "
                f"{span_counts[span]} < 2 — violates 'never < 2' rule"
            )


# ---------------------------------------------------------------------------
# Unit tests — IAA threshold flagging
# ---------------------------------------------------------------------------


class TestIAAThresholdFlaggingUnit:
    """Unit tests for IAA threshold flagging (Requirements 5.4, 6.4, 7.5)."""

    def test_sentiment_kappa_070_is_flagged(self):
        """
        GIVEN κ = 0.70 for a sentiment batch,
        WHEN  flag_batch is called,
        THEN  the batch is flagged (0.70 < 0.72 threshold).
        """
        result = flag_batch("batch-sentiment-001", "sentiment", 0.70)
        assert result.flagged is True
        assert result.kappa == 0.70
        assert result.threshold == SENTIMENT_KAPPA_THRESHOLD

    def test_sentiment_kappa_072_is_not_flagged(self):
        """
        GIVEN κ = 0.72 (exactly at threshold) for a sentiment batch,
        WHEN  flag_batch is called,
        THEN  the batch is NOT flagged (0.72 >= 0.72 threshold).
        """
        result = flag_batch("batch-sentiment-002", "sentiment", 0.72)
        assert result.flagged is False

    def test_sentiment_kappa_080_is_not_flagged(self):
        """
        GIVEN κ = 0.80 for a sentiment batch,
        WHEN  flag_batch is called,
        THEN  the batch is NOT flagged (0.80 >= 0.72 threshold).
        """
        result = flag_batch("batch-sentiment-003", "sentiment", 0.80)
        assert result.flagged is False

    def test_ner_kappa_075_is_flagged(self):
        """
        GIVEN κ = 0.75 for a NER batch,
        WHEN  flag_batch is called,
        THEN  the batch is flagged (0.75 < 0.78 threshold).
        """
        result = flag_batch("batch-ner-001", "ner", 0.75)
        assert result.flagged is True

    def test_ner_kappa_078_is_not_flagged(self):
        """
        GIVEN κ = 0.78 (exactly at threshold) for a NER batch,
        WHEN  flag_batch is called,
        THEN  the batch is NOT flagged.
        """
        result = flag_batch("batch-ner-002", "ner", 0.78)
        assert result.flagged is False

    def test_toxicity_kappa_060_is_flagged(self):
        """
        GIVEN κ = 0.60 for a toxicity batch,
        WHEN  flag_batch is called,
        THEN  the batch is flagged (0.60 < 0.65 threshold).
        """
        result = flag_batch("batch-tox-001", "toxicity", 0.60)
        assert result.flagged is True

    def test_toxicity_kappa_065_is_not_flagged(self):
        """
        GIVEN κ = 0.65 (exactly at threshold) for a toxicity batch,
        WHEN  flag_batch is called,
        THEN  the batch is NOT flagged.
        """
        result = flag_batch("batch-tox-002", "toxicity", 0.65)
        assert result.flagged is False

    def test_flagged_batch_is_logged_at_warning(self, caplog):
        """
        GIVEN a batch with κ below threshold,
        WHEN  flag_batch is called,
        THEN  a WARNING log message is emitted containing the batch ID and κ value.
        """
        with caplog.at_level(logging.WARNING, logger="annotator_tool.iaa"):
            flag_batch("batch-log-001", "sentiment", 0.50)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("batch-log-001" in msg for msg in warning_messages), (
            f"Expected batch ID in warning log; got: {warning_messages}"
        )

    def test_iaa_result_fields_are_correct(self):
        """
        GIVEN a call to flag_batch,
        WHEN  the result is returned,
        THEN  all IAAResult fields are populated correctly.
        """
        result = flag_batch("my-batch", "ner", 0.60)
        assert isinstance(result, IAAResult)
        assert result.batch_id == "my-batch"
        assert result.task == "ner"
        assert result.kappa == 0.60
        assert result.threshold == NER_KAPPA_THRESHOLD
        assert result.flagged is True

    def test_invalid_task_raises_value_error(self):
        """
        GIVEN an invalid task name,
        WHEN  flag_batch is called,
        THEN  ValueError is raised.
        """
        with pytest.raises(ValueError, match="Unknown task"):
            flag_batch("batch-001", "invalid_task", 0.50)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unit tests — majority vote for sentiment
# ---------------------------------------------------------------------------


class TestMajorityVoteSentimentUnit:
    """Unit tests for majority_vote_sentiment (Requirement 5.2)."""

    def test_two_one_split_returns_majority(self):
        """
        GIVEN labels ["positive", "positive", "neutral"] (2-1 split),
        WHEN  majority_vote_sentiment is called,
        THEN  "positive" is returned.
        """
        result = majority_vote_sentiment(["positive", "positive", "neutral"])
        assert result == "positive"

    def test_three_zero_split_returns_unanimous(self):
        """
        GIVEN labels ["negative", "negative", "negative"] (3-0 split),
        WHEN  majority_vote_sentiment is called,
        THEN  "negative" is returned.
        """
        result = majority_vote_sentiment(["negative", "negative", "negative"])
        assert result == "negative"

    def test_all_different_returns_none(self):
        """
        GIVEN labels ["positive", "neutral", "negative"] (all different),
        WHEN  majority_vote_sentiment is called,
        THEN  None is returned (no majority).
        """
        result = majority_vote_sentiment(["positive", "neutral", "negative"])
        assert result is None

    def test_two_one_split_neutral(self):
        """
        GIVEN labels ["neutral", "neutral", "positive"] (2-1 split for neutral),
        WHEN  majority_vote_sentiment is called,
        THEN  "neutral" is returned.
        """
        result = majority_vote_sentiment(["neutral", "neutral", "positive"])
        assert result == "neutral"

    def test_empty_labels_returns_none(self):
        """
        GIVEN an empty label list,
        WHEN  majority_vote_sentiment is called,
        THEN  None is returned.
        """
        result = majority_vote_sentiment([])
        assert result is None

    def test_single_label_returns_none(self):
        """
        GIVEN a single label (count=1 < 2),
        WHEN  majority_vote_sentiment is called,
        THEN  None is returned (count < 2).
        """
        result = majority_vote_sentiment(["positive"])
        assert result is None

    def test_two_identical_labels_returns_majority(self):
        """
        GIVEN two identical labels (count=2 >= 2),
        WHEN  majority_vote_sentiment is called,
        THEN  the repeated label is returned.
        """
        result = majority_vote_sentiment(["positive", "positive"])
        assert result == "positive"


# ---------------------------------------------------------------------------
# Unit tests — majority vote for NER
# ---------------------------------------------------------------------------


class TestMajorityVoteNERUnit:
    """Unit tests for majority_vote_ner (Requirement 6.2)."""

    def test_span_agreed_by_two_annotators_is_included(self):
        """
        GIVEN span (0, 6, "PER") marked by annotators 1 and 2 but not 3,
        WHEN  majority_vote_ner is called,
        THEN  the span is included in the result.
        """
        spans_a = [(0, 6, "PER"), (10, 15, "LOC")]
        spans_b = [(0, 6, "PER")]
        spans_c = [(20, 25, "ORG")]

        result = majority_vote_ner([spans_a, spans_b, spans_c])
        assert (0, 6, "PER") in result

    def test_span_agreed_by_only_one_annotator_is_excluded(self):
        """
        GIVEN span (10, 15, "LOC") marked by only annotator 1,
        WHEN  majority_vote_ner is called,
        THEN  the span is NOT included in the result.
        """
        spans_a = [(0, 6, "PER"), (10, 15, "LOC")]
        spans_b = [(0, 6, "PER")]
        spans_c = [(0, 6, "PER")]

        result = majority_vote_ner([spans_a, spans_b, spans_c])
        assert (10, 15, "LOC") not in result
        assert (0, 6, "PER") in result

    def test_all_three_agree_on_span(self):
        """
        GIVEN a span marked by all 3 annotators,
        WHEN  majority_vote_ner is called,
        THEN  the span is included.
        """
        span = (0, 6, "PER")
        result = majority_vote_ner([[span], [span], [span]])
        assert span in result

    def test_no_spans_returns_empty(self):
        """
        GIVEN all annotators mark zero spans,
        WHEN  majority_vote_ner is called,
        THEN  an empty list is returned.
        """
        result = majority_vote_ner([[], [], []])
        assert result == []

    def test_result_is_sorted_by_start_offset(self):
        """
        GIVEN multiple spans agreed by ≥ 2 annotators,
        WHEN  majority_vote_ner is called,
        THEN  the result is sorted by start offset.
        """
        spans_a = [(10, 15, "LOC"), (0, 6, "PER")]
        spans_b = [(10, 15, "LOC"), (0, 6, "PER")]
        spans_c = []

        result = majority_vote_ner([spans_a, spans_b, spans_c])
        assert result == [(0, 6, "PER"), (10, 15, "LOC")]

    def test_span_disagreement_on_entity_type_treated_separately(self):
        """
        GIVEN annotators disagree on entity type for the same offsets,
        WHEN  majority_vote_ner is called,
        THEN  only the entity type agreed by ≥ 2 annotators is included.
        """
        # Annotators 1 and 2 agree on PER; annotator 3 says LOC
        spans_a = [(0, 6, "PER")]
        spans_b = [(0, 6, "PER")]
        spans_c = [(0, 6, "LOC")]

        result = majority_vote_ner([spans_a, spans_b, spans_c])
        assert (0, 6, "PER") in result
        assert (0, 6, "LOC") not in result

    def test_empty_annotator_list_returns_empty(self):
        """
        GIVEN an empty list of annotators,
        WHEN  majority_vote_ner is called,
        THEN  an empty list is returned.
        """
        result = majority_vote_ner([])
        assert result == []
