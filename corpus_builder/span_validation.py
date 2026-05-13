"""
Corpus_Builder — entity span and toxicity label validators for RajNLP-50K.

Provides four public functions:

- ``validate_span_text_invariant``: For every ``EntitySpan`` in an
  ``AnnotatedSentence``, asserts ``sentence.text[span.start:span.end] == span.text``.
  Also checks all three annotator span sets (``ner_annotator_spans``).

- ``validate_all_span_text_invariants``: Runs ``validate_span_text_invariant``
  on every sentence in a list and returns all error messages combined.

- ``validate_toxicity_labels``: Asserts ``toxicity_labels`` is a subset of
  ``{"caste_slur", "religious", "gender", "general"}`` with no duplicates.

- ``validate_all_toxicity_labels``: Runs ``validate_toxicity_labels`` on every
  sentence in a list and returns all error messages combined.

These validators are intended to be called after every annotation export and
before serialization (Requirements 6.2, 7.1, 7.2, 11.2, 12.2).
"""

from __future__ import annotations

import logging

from models.data_models import AnnotatedSentence, EntitySpan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The set of valid toxicity category strings.
VALID_TOXICITY_CATEGORIES: frozenset[str] = frozenset(
    {"caste_slur", "religious", "gender", "general"}
)


# ---------------------------------------------------------------------------
# Span text invariant validator
# ---------------------------------------------------------------------------


def _check_spans(
    sentence_id: str,
    text: str,
    spans: list[EntitySpan],
    span_set_label: str,
) -> list[str]:
    """Check the span text invariant for a single list of spans.

    For each span, verifies that ``text[span.start:span.end] == span.text``.

    Args:
        sentence_id: The sentence UUID (used in error messages).
        text: The sentence text.
        spans: The list of :class:`~models.data_models.EntitySpan` objects to check.
        span_set_label: A human-readable label for the span set (e.g. ``"ner_spans"``
            or ``"ner_annotator_spans[1]"``), used in error messages.

    Returns:
        A list of error message strings.  Empty list means all spans are valid.
    """
    errors: list[str] = []
    for i, span in enumerate(spans):
        expected = text[span.start : span.end]
        if expected != span.text:
            errors.append(
                f"sentence_id={sentence_id!r}: {span_set_label}[{i}] "
                f"span.text={span.text!r} does not match "
                f"sentence.text[{span.start}:{span.end}]={expected!r}"
            )
    return errors


def validate_span_text_invariant(sentence: AnnotatedSentence) -> list[str]:
    """Validate the entity span text invariant for a single ``AnnotatedSentence``.

    For every :class:`~models.data_models.EntitySpan` in ``sentence.ner_spans``
    and in each of the three ``sentence.ner_annotator_spans`` sets, asserts that
    ``sentence.text[span.start:span.end] == span.text``.

    Args:
        sentence: The :class:`~models.data_models.AnnotatedSentence` to validate.

    Returns:
        A list of error message strings.  An empty list means the sentence is valid.
    """
    errors: list[str] = []

    # Check gold NER spans
    errors.extend(
        _check_spans(sentence.sentence_id, sentence.text, sentence.ner_spans, "ner_spans")
    )

    # Check each annotator's span set
    for annotator_idx, annotator_spans in enumerate(sentence.ner_annotator_spans):
        errors.extend(
            _check_spans(
                sentence.sentence_id,
                sentence.text,
                annotator_spans,
                f"ner_annotator_spans[{annotator_idx}]",
            )
        )

    return errors


def validate_all_span_text_invariants(sentences: list[AnnotatedSentence]) -> list[str]:
    """Validate the entity span text invariant for every sentence in a list.

    Runs :func:`validate_span_text_invariant` on each sentence and collects all
    error messages.

    Args:
        sentences: A list of :class:`~models.data_models.AnnotatedSentence` objects.

    Returns:
        A combined list of all error messages across all sentences.  An empty
        list means every sentence is valid.
    """
    all_errors: list[str] = []
    for sentence in sentences:
        errors = validate_span_text_invariant(sentence)
        if errors:
            for msg in errors:
                logger.error("Span text invariant violation: %s", msg)
            all_errors.extend(errors)
    return all_errors


# ---------------------------------------------------------------------------
# Toxicity label set validator
# ---------------------------------------------------------------------------


def validate_toxicity_labels(sentence: AnnotatedSentence) -> list[str]:
    """Validate the toxicity labels for a single ``AnnotatedSentence``.

    Checks that:
    1. Every label in ``sentence.toxicity_labels`` is one of the four valid
       categories: ``"caste_slur"``, ``"religious"``, ``"gender"``, ``"general"``.
    2. There are no duplicate labels in ``sentence.toxicity_labels``.

    Args:
        sentence: The :class:`~models.data_models.AnnotatedSentence` to validate.

    Returns:
        A list of error message strings.  An empty list means the labels are valid.
    """
    errors: list[str] = []
    labels = sentence.toxicity_labels

    # Check for invalid categories
    invalid = [lbl for lbl in labels if lbl not in VALID_TOXICITY_CATEGORIES]
    if invalid:
        errors.append(
            f"sentence_id={sentence.sentence_id!r}: toxicity_labels contains "
            f"invalid categories: {invalid!r}. "
            f"Valid categories are: {sorted(VALID_TOXICITY_CATEGORIES)!r}"
        )

    # Check for duplicates
    seen: set[str] = set()
    duplicates: list[str] = []
    for lbl in labels:
        if lbl in seen:
            duplicates.append(lbl)
        else:
            seen.add(lbl)
    if duplicates:
        errors.append(
            f"sentence_id={sentence.sentence_id!r}: toxicity_labels contains "
            f"duplicate entries: {duplicates!r}"
        )

    return errors


def validate_all_toxicity_labels(sentences: list[AnnotatedSentence]) -> list[str]:
    """Validate toxicity labels for every sentence in a list.

    Runs :func:`validate_toxicity_labels` on each sentence and collects all
    error messages.

    Args:
        sentences: A list of :class:`~models.data_models.AnnotatedSentence` objects.

    Returns:
        A combined list of all error messages across all sentences.  An empty
        list means every sentence has valid toxicity labels.
    """
    all_errors: list[str] = []
    for sentence in sentences:
        errors = validate_toxicity_labels(sentence)
        if errors:
            for msg in errors:
                logger.error("Toxicity label validation error: %s", msg)
            all_errors.extend(errors)
    return all_errors
