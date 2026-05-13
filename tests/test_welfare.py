"""
Tests for annotator_tool.welfare — annotator welfare controls.

Covers:
- Opt-out mechanism: reassigns sentences to replacement queue (Req 16.3)
- Daily toxicity exposure limiter: blocks assignment after threshold (Req 16.5)

Requirements: 16.3, 16.5
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pytest

from annotator_tool.welfare import AnnotatorWelfareManager, CONTENT_WARNING


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 3, 15, 9, 0, 0, tzinfo=timezone.utc)
_TODAY = date(2024, 3, 15)
_TOMORROW = date(2024, 3, 16)


def _manager(sentences_per_hour: int = 100) -> AnnotatorWelfareManager:
    """Return a fresh AnnotatorWelfareManager with the given rate."""
    return AnnotatorWelfareManager(sentences_per_hour=sentences_per_hour)


# ---------------------------------------------------------------------------
# Opt-out tests (Requirement 16.3)
# ---------------------------------------------------------------------------


class TestOptOut:
    """Tests for the opt-out mechanism (Requirement 16.3)."""

    def test_opt_out_reassigns_sentences_to_replacement_queue(self):
        """
        GIVEN an annotator with assigned sentences,
        WHEN  record_opt_out is called,
        THEN  all their sentences appear in the replacement queue.
        """
        mgr = _manager()
        sentence_ids = ["s1", "s2", "s3"]
        mgr.assign_sentences("ann-001", sentence_ids)

        reassigned = mgr.record_opt_out("ann-001", _NOW)

        assert set(reassigned) == set(sentence_ids)
        queue = mgr.get_replacement_queue()
        assert set(queue) == set(sentence_ids)

    def test_opt_out_returns_list_of_reassigned_sentence_ids(self):
        """
        GIVEN an annotator with 2 assigned sentences,
        WHEN  record_opt_out is called,
        THEN  the returned list contains exactly those 2 sentence_ids.
        """
        mgr = _manager()
        mgr.assign_sentences("ann-002", ["s10", "s11"])

        result = mgr.record_opt_out("ann-002", _NOW)

        assert sorted(result) == ["s10", "s11"]

    def test_opt_out_with_no_assigned_sentences_returns_empty_list(self):
        """
        GIVEN an annotator with no assigned sentences,
        WHEN  record_opt_out is called,
        THEN  an empty list is returned and the queue is unchanged.
        """
        mgr = _manager()
        result = mgr.record_opt_out("ann-003", _NOW)
        assert result == []
        assert mgr.get_replacement_queue() == []

    def test_opt_out_is_logged(self, caplog):
        """
        GIVEN an annotator with assigned sentences,
        WHEN  record_opt_out is called,
        THEN  a WARNING log message is emitted containing the annotator_id
              and timestamp.
        """
        mgr = _manager()
        mgr.assign_sentences("ann-004", ["s20"])

        with caplog.at_level(logging.WARNING, logger="annotator_tool.welfare"):
            mgr.record_opt_out("ann-004", _NOW)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("ann-004" in msg for msg in warning_messages), (
            f"Expected annotator_id in warning log; got: {warning_messages}"
        )
        assert any(_NOW.isoformat() in msg for msg in warning_messages), (
            f"Expected timestamp in warning log; got: {warning_messages}"
        )

    def test_replacement_queue_contains_reassigned_sentences(self):
        """
        GIVEN two annotators who both opt out,
        WHEN  get_replacement_queue is called,
        THEN  the queue contains all sentences from both annotators.
        """
        mgr = _manager()
        mgr.assign_sentences("ann-005", ["s30", "s31"])
        mgr.assign_sentences("ann-006", ["s32"])

        mgr.record_opt_out("ann-005", _NOW)
        mgr.record_opt_out("ann-006", _NOW)

        queue = mgr.get_replacement_queue()
        assert set(queue) == {"s30", "s31", "s32"}

    def test_assign_replacement_removes_from_queue(self):
        """
        GIVEN a sentence in the replacement queue,
        WHEN  assign_replacement is called for that sentence,
        THEN  the sentence is removed from the replacement queue.
        """
        mgr = _manager()
        mgr.assign_sentences("ann-007", ["s40", "s41"])
        mgr.record_opt_out("ann-007", _NOW)

        mgr.assign_replacement("s40", "ann-replacement-001", _NOW)

        queue = mgr.get_replacement_queue()
        assert "s40" not in queue
        assert "s41" in queue

    def test_assign_replacement_raises_if_sentence_not_in_queue(self):
        """
        GIVEN a sentence_id that is NOT in the replacement queue,
        WHEN  assign_replacement is called,
        THEN  ValueError is raised.
        """
        mgr = _manager()
        with pytest.raises(ValueError, match="not in the replacement queue"):
            mgr.assign_replacement("nonexistent-sentence", "ann-replacement-002", _NOW)

    def test_get_replacement_queue_returns_copy(self):
        """
        GIVEN a non-empty replacement queue,
        WHEN  get_replacement_queue is called twice,
        THEN  modifying the returned list does not affect the internal queue.
        """
        mgr = _manager()
        mgr.assign_sentences("ann-008", ["s50"])
        mgr.record_opt_out("ann-008", _NOW)

        queue1 = mgr.get_replacement_queue()
        queue1.clear()

        queue2 = mgr.get_replacement_queue()
        assert "s50" in queue2


# ---------------------------------------------------------------------------
# Daily exposure limiter tests (Requirement 16.5)
# ---------------------------------------------------------------------------


class TestDailyExposureLimiter:
    """Tests for the daily toxicity exposure limiter (Requirement 16.5)."""

    def test_daily_limit_allows_assignment_below_threshold(self):
        """
        GIVEN an annotator who has labeled fewer sentences than the daily max,
        WHEN  check_daily_limit is called,
        THEN  True is returned (assignment is allowed).
        """
        mgr = _manager(sentences_per_hour=100)  # max = 200
        mgr.record_toxicity_exposure("ann-101", 50, _TODAY)

        assert mgr.check_daily_limit("ann-101", _TODAY) is True

    def test_daily_limit_blocks_assignment_after_threshold_is_reached(self):
        """
        GIVEN an annotator who has reached the daily maximum sentence count,
        WHEN  check_daily_limit is called,
        THEN  False is returned (assignment is blocked).
        """
        mgr = _manager(sentences_per_hour=100)  # max = 200
        mgr.record_toxicity_exposure("ann-102", 200, _TODAY)

        assert mgr.check_daily_limit("ann-102", _TODAY) is False

    def test_daily_limit_blocks_when_exceeded(self):
        """
        GIVEN an annotator who has exceeded the daily maximum,
        WHEN  check_daily_limit is called,
        THEN  False is returned.
        """
        mgr = _manager(sentences_per_hour=100)  # max = 200
        mgr.record_toxicity_exposure("ann-103", 250, _TODAY)

        assert mgr.check_daily_limit("ann-103", _TODAY) is False

    def test_daily_limit_allows_when_no_exposure_recorded(self):
        """
        GIVEN an annotator with no recorded exposure for today,
        WHEN  check_daily_limit is called,
        THEN  True is returned.
        """
        mgr = _manager()
        assert mgr.check_daily_limit("ann-104", _TODAY) is True

    def test_daily_limit_is_logged_when_reached(self, caplog):
        """
        GIVEN an annotator who has reached the daily limit,
        WHEN  check_daily_limit is called,
        THEN  a WARNING log message is emitted containing the annotator_id.
        """
        mgr = _manager(sentences_per_hour=100)  # max = 200
        mgr.record_toxicity_exposure("ann-105", 200, _TODAY)

        with caplog.at_level(logging.WARNING, logger="annotator_tool.welfare"):
            mgr.check_daily_limit("ann-105", _TODAY)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("ann-105" in msg for msg in warning_messages), (
            f"Expected annotator_id in warning log; got: {warning_messages}"
        )

    def test_daily_exposure_accumulates(self):
        """
        GIVEN multiple calls to record_toxicity_exposure for the same annotator
              and day,
        WHEN  get_daily_exposure is called,
        THEN  the total is the sum of all recorded counts.
        """
        mgr = _manager()
        mgr.record_toxicity_exposure("ann-106", 50, _TODAY)
        mgr.record_toxicity_exposure("ann-106", 30, _TODAY)
        mgr.record_toxicity_exposure("ann-106", 20, _TODAY)

        assert mgr.get_daily_exposure("ann-106", _TODAY) == 100

    def test_different_annotators_tracked_independently(self):
        """
        GIVEN two annotators with different exposure levels on the same day,
        WHEN  check_daily_limit is called for each,
        THEN  each annotator's limit is evaluated independently.
        """
        mgr = _manager(sentences_per_hour=100)  # max = 200
        mgr.record_toxicity_exposure("ann-107", 200, _TODAY)  # at limit
        mgr.record_toxicity_exposure("ann-108", 50, _TODAY)   # under limit

        assert mgr.check_daily_limit("ann-107", _TODAY) is False
        assert mgr.check_daily_limit("ann-108", _TODAY) is True

    def test_configurable_sentences_per_hour(self):
        """
        GIVEN a manager configured with sentences_per_hour=50,
        WHEN  max_daily_sentences is accessed,
        THEN  it equals 50 * 2 = 100.
        """
        mgr = _manager(sentences_per_hour=50)
        assert mgr.max_daily_sentences == 100

    def test_max_daily_sentences_default_is_200(self):
        """
        GIVEN a manager with the default sentences_per_hour=100,
        WHEN  max_daily_sentences is accessed,
        THEN  it equals 200.
        """
        mgr = _manager()
        assert mgr.max_daily_sentences == 200

    def test_daily_exposure_resets_across_days(self):
        """
        GIVEN an annotator who reached the limit yesterday,
        WHEN  check_daily_limit is called for today,
        THEN  True is returned (limits are per-day).
        """
        mgr = _manager(sentences_per_hour=100)  # max = 200
        mgr.record_toxicity_exposure("ann-109", 200, _TOMORROW)

        # Today has no exposure recorded → should be allowed
        assert mgr.check_daily_limit("ann-109", _TODAY) is True

    def test_get_daily_exposure_returns_zero_when_no_exposure(self):
        """
        GIVEN an annotator with no recorded exposure,
        WHEN  get_daily_exposure is called,
        THEN  0 is returned.
        """
        mgr = _manager()
        assert mgr.get_daily_exposure("ann-110", _TODAY) == 0

    def test_daily_limit_at_exactly_max_is_blocked(self):
        """
        GIVEN an annotator whose exposure equals exactly max_daily_sentences,
        WHEN  check_daily_limit is called,
        THEN  False is returned (limit reached, not just exceeded).
        """
        mgr = _manager(sentences_per_hour=60)  # max = 120
        mgr.record_toxicity_exposure("ann-111", 120, _TODAY)

        assert mgr.check_daily_limit("ann-111", _TODAY) is False

    def test_daily_limit_one_below_max_is_allowed(self):
        """
        GIVEN an annotator whose exposure is one below max_daily_sentences,
        WHEN  check_daily_limit is called,
        THEN  True is returned.
        """
        mgr = _manager(sentences_per_hour=60)  # max = 120
        mgr.record_toxicity_exposure("ann-112", 119, _TODAY)

        assert mgr.check_daily_limit("ann-112", _TODAY) is True


# ---------------------------------------------------------------------------
# Content warning constant
# ---------------------------------------------------------------------------


class TestContentWarning:
    """Verify the CONTENT_WARNING constant is present and informative."""

    def test_content_warning_is_non_empty_string(self):
        """CONTENT_WARNING must be a non-empty string."""
        assert isinstance(CONTENT_WARNING, str)
        assert len(CONTENT_WARNING) > 0

    def test_content_warning_mentions_caste_slur(self):
        """Content warning must mention caste-based content."""
        assert "caste" in CONTENT_WARNING.lower()

    def test_content_warning_mentions_opt_out(self):
        """Content warning must mention the opt-out option."""
        assert "opt out" in CONTENT_WARNING.lower() or "opt-out" in CONTENT_WARNING.lower()

    def test_content_warning_mentions_two_hour_limit(self):
        """Content warning must mention the 2-hour daily limit."""
        assert "2 hour" in CONTENT_WARNING.lower() or "two hour" in CONTENT_WARNING.lower()
