"""
Annotator_Tool — Annotator welfare controls for the toxicity labeling task.

This module implements ethics onboarding and welfare protections required
before and during toxicity annotation:

  - Content warning text presented to annotators before they begin (Req 16.2)
  - Opt-out mechanism: annotators may withdraw at any time; their queued
    sentences are reassigned to a replacement queue (Req 16.3)
  - Daily toxicity exposure limiter: maximum 2 hours per annotator per day,
    with a configurable sentences-per-hour rate (Req 16.5)

All opt-out reassignment events and daily-limit events are logged at WARNING
level with annotator_id and timestamp.

Requirements: 16.2, 16.3, 16.5
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content warning text (Requirement 16.2)
# ---------------------------------------------------------------------------

#: Written content warning presented to every annotator before they begin
#: the toxicity labeling task.
CONTENT_WARNING: str = """
CONTENT WARNING — PLEASE READ CAREFULLY BEFORE PROCEEDING

This annotation task involves labeling text for toxic content. You will
encounter sentences that may contain:

  • Caste-based slurs and derogatory references to caste identity
  • Religious hatred, incitement, or sectarian abuse
  • Gender-based harassment, misogyny, and sexist language
  • General toxic, abusive, or threatening language

Exposure to this material can be distressing. Before you begin, please be
aware of the following:

  1. You are not required to complete this task. You may opt out at any time
     without penalty. Your sentences will be reassigned to another annotator.

  2. Your daily exposure is limited to a maximum of 2 hours of toxicity
     annotation per day to protect your wellbeing.

  3. A check-in with the annotation lead will be conducted at least once per
     week to assess your wellbeing and address any concerns.

  4. If you feel distressed at any point, please stop immediately and contact
     the annotation lead.

By proceeding, you confirm that you have read and understood this content
warning and consent to annotating toxic content under the conditions described
above.
""".strip()


# ---------------------------------------------------------------------------
# AnnotatorWelfareManager
# ---------------------------------------------------------------------------


class AnnotatorWelfareManager:
    """Manages annotator welfare controls for the toxicity labeling task.

    Tracks opt-outs, maintains a replacement queue for reassigned sentences,
    and enforces the daily toxicity exposure limit.

    Args:
        sentences_per_hour: Estimated number of sentences an annotator can
            label per hour.  Used to compute the maximum daily sentence count
            (``sentences_per_hour * 2``).  Defaults to 100.
    """

    def __init__(self, sentences_per_hour: int = 100) -> None:
        self._sentences_per_hour: int = sentences_per_hour

        # annotator_id → list of sentence_ids assigned to that annotator
        self._annotator_sentences: dict[str, list[str]] = defaultdict(list)

        # Replacement queue: sentence_ids waiting for reassignment
        self._replacement_queue: list[str] = []

        # Set of annotators who have opted out
        self._opted_out: set[str] = set()

        # (annotator_id, date) → total sentence count for that day
        self._daily_exposure: dict[tuple[str, date], int] = defaultdict(int)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def max_daily_sentences(self) -> int:
        """Maximum number of toxicity sentences an annotator may label per day.

        Computed as ``sentences_per_hour * 2`` (2 hours maximum daily exposure).
        """
        return self._sentences_per_hour * 2

    # -----------------------------------------------------------------------
    # Sentence assignment tracking
    # -----------------------------------------------------------------------

    def assign_sentences(
        self,
        annotator_id: str,
        sentence_ids: list[str],
    ) -> None:
        """Record that a set of sentences has been assigned to an annotator.

        This must be called before :meth:`record_opt_out` so that the manager
        knows which sentences belong to the annotator.

        Args:
            annotator_id: The annotator's unique identifier.
            sentence_ids: List of sentence IDs assigned to this annotator.
        """
        self._annotator_sentences[annotator_id].extend(sentence_ids)

    # -----------------------------------------------------------------------
    # Opt-out mechanism (Requirement 16.3)
    # -----------------------------------------------------------------------

    def record_opt_out(
        self,
        annotator_id: str,
        timestamp: datetime,
    ) -> list[str]:
        """Record an annotator's opt-out from the toxicity labeling task.

        Moves all sentences currently assigned to the annotator into the
        replacement queue and logs the event at WARNING level.

        Args:
            annotator_id: The annotator who is opting out.
            timestamp: The datetime at which the opt-out was recorded (UTC).

        Returns:
            A list of sentence_ids that were moved to the replacement queue.
        """
        self._opted_out.add(annotator_id)

        reassigned = list(self._annotator_sentences.get(annotator_id, []))
        # Clear the annotator's assignment list
        self._annotator_sentences[annotator_id] = []

        # Add to replacement queue (avoid duplicates)
        for sid in reassigned:
            if sid not in self._replacement_queue:
                self._replacement_queue.append(sid)

        logger.warning(
            "Annotator opt-out recorded: annotator_id=%r, timestamp=%s, "
            "sentences_reassigned=%d, sentence_ids=%r",
            annotator_id,
            timestamp.isoformat(),
            len(reassigned),
            reassigned,
        )

        return reassigned

    def get_replacement_queue(self) -> list[str]:
        """Return the list of sentence_ids waiting for reassignment.

        Returns:
            A copy of the current replacement queue.
        """
        return list(self._replacement_queue)

    def assign_replacement(
        self,
        sentence_id: str,
        new_annotator_id: str,
        timestamp: datetime,
    ) -> None:
        """Assign a queued sentence to a replacement annotator.

        Removes the sentence from the replacement queue and records it as
        assigned to the new annotator.

        Args:
            sentence_id: The sentence to reassign.
            new_annotator_id: The replacement annotator's unique identifier.
            timestamp: The datetime of the reassignment (UTC).

        Raises:
            ValueError: If *sentence_id* is not in the replacement queue.
        """
        if sentence_id not in self._replacement_queue:
            raise ValueError(
                f"sentence_id {sentence_id!r} is not in the replacement queue"
            )

        self._replacement_queue.remove(sentence_id)
        self._annotator_sentences[new_annotator_id].append(sentence_id)

        logger.info(
            "Replacement assignment: sentence_id=%r → annotator_id=%r, timestamp=%s",
            sentence_id,
            new_annotator_id,
            timestamp.isoformat(),
        )

    # -----------------------------------------------------------------------
    # Daily exposure limiter (Requirement 16.5)
    # -----------------------------------------------------------------------

    def record_toxicity_exposure(
        self,
        annotator_id: str,
        sentence_count: int,
        day: date,
    ) -> None:
        """Record that an annotator has labeled a batch of toxicity sentences.

        Accumulates the sentence count for the given day.  Multiple calls for
        the same annotator and day are additive.

        Args:
            annotator_id: The annotator's unique identifier.
            sentence_count: Number of toxicity sentences labeled in this batch.
            day: The calendar date on which the annotation occurred.
        """
        self._daily_exposure[(annotator_id, day)] += sentence_count

    def check_daily_limit(
        self,
        annotator_id: str,
        day: date,
    ) -> bool:
        """Check whether an annotator can still be assigned toxicity sentences today.

        Args:
            annotator_id: The annotator's unique identifier.
            day: The calendar date to check.

        Returns:
            ``True`` if the annotator is under the daily limit and can receive
            more sentences.  ``False`` if the daily limit has been reached or
            exceeded, in which case a WARNING is logged.
        """
        current = self._daily_exposure.get((annotator_id, day), 0)
        if current >= self.max_daily_sentences:
            logger.warning(
                "Daily toxicity limit reached: annotator_id=%r, date=%s, "
                "sentences_today=%d, max_daily_sentences=%d",
                annotator_id,
                day.isoformat(),
                current,
                self.max_daily_sentences,
            )
            return False
        return True

    def get_daily_exposure(
        self,
        annotator_id: str,
        day: date,
    ) -> int:
        """Return the current daily sentence count for an annotator.

        Args:
            annotator_id: The annotator's unique identifier.
            day: The calendar date to query.

        Returns:
            Total number of toxicity sentences labeled by this annotator on
            the given day.  Returns 0 if no exposure has been recorded.
        """
        return self._daily_exposure.get((annotator_id, day), 0)
