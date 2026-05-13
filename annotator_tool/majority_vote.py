"""
Annotator_Tool — Majority vote resolution for annotation disagreements.

Provides majority vote functions for sentiment and NER annotation tasks.

Rules (from design.md):
  - Sentiment: return the label appearing in ≥ 2 of 3 annotations.
  - NER: a span is included in the gold set if ≥ 2 annotators marked it.
  - Never return a label appearing in fewer than 2 annotations.

Requirements: 5.2, 6.2
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentiment majority vote
# ---------------------------------------------------------------------------


def majority_vote_sentiment(labels: Sequence[str]) -> str | None:
    """Return the majority-vote sentiment label from a sequence of annotator labels.

    The majority label is the one appearing in ≥ 2 of the provided labels.
    If no label appears in ≥ 2 annotations (e.g., all three annotators chose
    different labels), ``None`` is returned to indicate that adjudication is
    required.

    Args:
        labels: A sequence of sentiment label strings from individual annotators.
            Typically 3 labels (one per annotator).

    Returns:
        The majority label string if one exists (appears ≥ 2 times), or
        ``None`` if no majority exists.

    Examples:
        >>> majority_vote_sentiment(["positive", "positive", "neutral"])
        'positive'
        >>> majority_vote_sentiment(["positive", "positive", "positive"])
        'positive'
        >>> majority_vote_sentiment(["positive", "neutral", "negative"])
        None
    """
    if not labels:
        return None

    counts = Counter(labels)
    # Find labels that appear ≥ 2 times
    majority_candidates = [label for label, count in counts.items() if count >= 2]

    if not majority_candidates:
        logger.warning(
            "No majority label found for annotations %r — adjudication required",
            list(labels),
        )
        return None

    # If multiple labels tie at ≥ 2 (only possible with > 3 annotators),
    # return the one with the highest count; break ties lexicographically.
    best_label = max(majority_candidates, key=lambda lbl: (counts[lbl], lbl))
    return best_label


# ---------------------------------------------------------------------------
# NER span-level majority vote
# ---------------------------------------------------------------------------


def majority_vote_ner(
    annotator_spans: Sequence[Sequence[tuple[int, int, str]]],
) -> list[tuple[int, int, str]]:
    """Return the majority-vote NER span set from multiple annotators' span sets.

    A span is included in the gold set if ≥ 2 annotators marked it.  Spans
    are identified by their ``(start, end, entity_type)`` tuple.

    Args:
        annotator_spans: A sequence of per-annotator span sequences.  Each
            inner sequence contains ``(start, end, entity_type)`` tuples for
            one annotator.  Typically 3 annotators.

    Returns:
        A sorted list of ``(start, end, entity_type)`` tuples that appeared
        in ≥ 2 annotators' annotations.  Returns an empty list if no span
        meets the threshold.

    Examples:
        >>> spans_a = [(0, 6, "PER"), (10, 15, "LOC")]
        >>> spans_b = [(0, 6, "PER")]
        >>> spans_c = [(0, 6, "PER"), (20, 25, "ORG")]
        >>> majority_vote_ner([spans_a, spans_b, spans_c])
        [(0, 6, 'PER')]
    """
    if not annotator_spans:
        return []

    # Count how many annotators marked each span
    span_counts: Counter[tuple[int, int, str]] = Counter()
    for spans in annotator_spans:
        # Use a set per annotator to avoid double-counting if an annotator
        # somehow marked the same span twice
        for span in set(spans):
            span_counts[span] += 1

    # Keep only spans that ≥ 2 annotators marked
    gold_spans = [
        span for span, count in span_counts.items() if count >= 2
    ]

    # Sort by start offset for deterministic output
    gold_spans.sort(key=lambda s: (s[0], s[1], s[2]))
    return gold_spans


# ---------------------------------------------------------------------------
# Convenience wrappers for EntitySpan objects
# ---------------------------------------------------------------------------


def majority_vote_sentiment_from_annotated(
    sentiment_annotator_labels: list[str],
) -> str | None:
    """Wrapper around :func:`majority_vote_sentiment` for AnnotatedSentence fields.

    Args:
        sentiment_annotator_labels: The ``sentiment_annotator_labels`` field
            from an ``AnnotatedSentence`` (list of 3 raw labels).

    Returns:
        The majority-vote sentiment label, or ``None`` if no majority exists.
    """
    return majority_vote_sentiment(sentiment_annotator_labels)


def majority_vote_ner_from_entity_spans(
    ner_annotator_spans: list[list],
) -> list:
    """Wrapper around :func:`majority_vote_ner` for EntitySpan objects.

    Converts ``EntitySpan`` objects to ``(start, end, entity_type)`` tuples,
    applies majority vote, and returns the result as a list of tuples.

    Args:
        ner_annotator_spans: The ``ner_annotator_spans`` field from an
            ``AnnotatedSentence`` — a list of 3 lists of ``EntitySpan`` objects.

    Returns:
        A sorted list of ``(start, end, entity_type)`` tuples for the gold
        NER spans.
    """
    # Convert EntitySpan objects to tuples
    tuple_spans = [
        [(span.start, span.end, span.entity_type) for span in annotator_span_list]
        for annotator_span_list in ner_annotator_spans
    ]
    return majority_vote_ner(tuple_spans)
