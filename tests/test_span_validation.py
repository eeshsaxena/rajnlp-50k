"""
Tests for corpus_builder.span_validation — entity span and toxicity label validators.

Covers:
- Property 5: Entity span text invariant
  For any AnnotatedSentence with NER spans, for every span in ner_spans,
  sentence.text[span.start:span.end] SHALL equal span.text.
  (Validates: Requirements 6.2, 11.2)

- Property 6: Toxicity label set is a subset of valid categories
  For any AnnotatedSentence, toxicity_labels SHALL be a subset of
  {"caste_slur", "religious", "gender", "general"} and SHALL contain no
  duplicate entries.
  (Validates: Requirements 7.1, 7.2, 12.2)

- Unit tests for both validators.

Requirements: 6.2, 7.1, 7.2, 11.2, 12.2
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from corpus_builder.span_validation import (
    VALID_TOXICITY_CATEGORIES,
    validate_all_span_text_invariants,
    validate_all_toxicity_labels,
    validate_span_text_invariant,
    validate_toxicity_labels,
)
from models.data_models import AnnotatedSentence, EntitySpan, TokenLabel

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FIXED_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
_ANNOTATED_TS = datetime(2024, 2, 1, 14, 0, 0, tzinfo=timezone.utc)

_PLATFORMS = ["twitter", "sharechat"]
_SPLITS = ["train", "validation", "test"]
_SENTIMENTS = ["positive", "neutral", "negative"]
_ENTITY_TYPES = ["PER", "LOC", "ORG"]
_LANG_LABELS = ["RAJ", "HIN", "ENG", "TRL"]
_TOXICITY_CATS = sorted(VALID_TOXICITY_CATEGORIES)  # deterministic order


def _make_sentence(
    text: str = "Gehlot ne Jaipur mein bada ailan kiya",
    ner_spans: list[EntitySpan] | None = None,
    ner_annotator_spans: list[list[EntitySpan]] | None = None,
    toxicity_labels: list[str] | None = None,
    sentence_id: str | None = None,
) -> AnnotatedSentence:
    """Factory for AnnotatedSentence with sensible defaults."""
    if sentence_id is None:
        sentence_id = str(uuid.uuid4())
    if ner_spans is None:
        ner_spans = []
    if ner_annotator_spans is None:
        ner_annotator_spans = [[], [], []]
    if toxicity_labels is None:
        toxicity_labels = []
    return AnnotatedSentence(
        sentence_id=sentence_id,
        text=text,
        platform="twitter",
        split="train",
        sentiment="positive",
        sentiment_annotator_labels=["positive", "positive", "neutral"],
        ner_spans=ner_spans,
        ner_annotator_spans=ner_annotator_spans,
        toxicity_labels=toxicity_labels,  # type: ignore[arg-type]
        toxicity_annotator_labels=[[], [], []],
        token_language_labels=[],
        source_url="https://twitter.com/example/status/123",
        collected_at=_FIXED_TS,
        annotated_at=_ANNOTATED_TS,
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def _valid_span_strategy(draw, text: str) -> EntitySpan:
    """Generate an EntitySpan whose text matches sentence.text[start:end]."""
    if not text:
        return EntitySpan(start=0, end=0, entity_type="PER", text="")
    max_end = len(text)
    start = draw(st.integers(min_value=0, max_value=max(0, max_end - 1)))
    end = draw(st.integers(min_value=start, max_value=max_end))
    entity_type = draw(st.sampled_from(_ENTITY_TYPES))
    return EntitySpan(start=start, end=end, entity_type=entity_type, text=text[start:end])


@st.composite
def _sentence_with_valid_spans_strategy(draw) -> AnnotatedSentence:
    """Generate an AnnotatedSentence where all spans satisfy the text invariant."""
    text = draw(
        st.text(
            min_size=1,
            max_size=80,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
                whitelist_characters=(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
                ),
            ),
        )
    )
    ner_spans = draw(st.lists(_valid_span_strategy(text), min_size=0, max_size=3))
    ner_annotator_spans = draw(
        st.lists(
            st.lists(_valid_span_strategy(text), min_size=0, max_size=2),
            min_size=3,
            max_size=3,
        )
    )
    toxicity_labels = draw(
        st.lists(st.sampled_from(_TOXICITY_CATS), min_size=0, max_size=4, unique=True)
    )
    return _make_sentence(
        text=text,
        ner_spans=ner_spans,
        ner_annotator_spans=ner_annotator_spans,
        toxicity_labels=toxicity_labels,
    )


@st.composite
def _sentence_with_valid_toxicity_strategy(draw) -> AnnotatedSentence:
    """Generate an AnnotatedSentence with valid toxicity labels (subset, no duplicates)."""
    toxicity_labels = draw(
        st.lists(st.sampled_from(_TOXICITY_CATS), min_size=0, max_size=4, unique=True)
    )
    return _make_sentence(toxicity_labels=toxicity_labels)


# ---------------------------------------------------------------------------
# Property 5: Entity span text invariant
# ---------------------------------------------------------------------------


class TestProperty5EntitySpanTextInvariant:
    """
    Property 5: Entity span text invariant

    For any AnnotatedSentence with NER spans, for every span in ner_spans,
    sentence.text[span.start:span.end] SHALL equal span.text.

    **Validates: Requirements 6.2, 11.2**
    """

    @given(sentence=_sentence_with_valid_spans_strategy())
    @settings(max_examples=100)
    def test_valid_spans_produce_no_errors(self, sentence: AnnotatedSentence) -> None:
        """
        GIVEN an AnnotatedSentence where all spans are constructed so that
              span.text == sentence.text[span.start:span.end],
        WHEN  validate_span_text_invariant is called,
        THEN  it returns an empty error list.
        """
        errors = validate_span_text_invariant(sentence)
        assert errors == [], (
            f"Expected no errors for valid spans, got: {errors}"
        )

    @given(sentence=_sentence_with_valid_spans_strategy())
    @settings(max_examples=100)
    def test_broken_span_text_produces_errors(self, sentence: AnnotatedSentence) -> None:
        """
        GIVEN an AnnotatedSentence where one span has a deliberately wrong span.text,
        WHEN  validate_span_text_invariant is called,
        THEN  it returns at least one error message.
        """
        # Inject a broken span into ner_spans
        broken_span = EntitySpan(start=0, end=0, entity_type="PER", text="WRONG_TEXT")
        broken_sentence = _make_sentence(
            text=sentence.text,
            ner_spans=[broken_span],
            ner_annotator_spans=sentence.ner_annotator_spans,
        )
        errors = validate_span_text_invariant(broken_sentence)
        # "WRONG_TEXT" != sentence.text[0:0] == "" so there must be an error
        assert len(errors) >= 1, (
            "Expected at least one error for a broken span, got none"
        )


# ---------------------------------------------------------------------------
# Property 6: Toxicity label set is a subset of valid categories
# ---------------------------------------------------------------------------


class TestProperty6ToxicityLabelValidity:
    """
    Property 6: Toxicity label set is a subset of valid categories

    For any AnnotatedSentence, toxicity_labels SHALL be a subset of
    {"caste_slur", "religious", "gender", "general"} and SHALL contain no
    duplicate entries.

    **Validates: Requirements 7.1, 7.2, 12.2**
    """

    @given(sentence=_sentence_with_valid_toxicity_strategy())
    @settings(max_examples=100)
    def test_valid_toxicity_labels_produce_no_errors(self, sentence: AnnotatedSentence) -> None:
        """
        GIVEN an AnnotatedSentence with toxicity_labels that are a subset of the
              four valid categories and contain no duplicates,
        WHEN  validate_toxicity_labels is called,
        THEN  it returns an empty error list.
        """
        errors = validate_toxicity_labels(sentence)
        assert errors == [], (
            f"Expected no errors for valid toxicity labels, got: {errors}"
        )

    @given(
        invalid_label=st.text(min_size=1, max_size=20).filter(
            lambda s: s not in VALID_TOXICITY_CATEGORIES
        )
    )
    @settings(max_examples=100)
    def test_invalid_category_produces_errors(self, invalid_label: str) -> None:
        """
        GIVEN an AnnotatedSentence with a toxicity label that is not in the
              valid category set,
        WHEN  validate_toxicity_labels is called,
        THEN  it returns at least one error message.
        """
        sentence = _make_sentence(toxicity_labels=[invalid_label])  # type: ignore[list-item]
        errors = validate_toxicity_labels(sentence)
        assert len(errors) >= 1, (
            f"Expected an error for invalid label {invalid_label!r}, got none"
        )

    @given(
        valid_label=st.sampled_from(_TOXICITY_CATS)
    )
    @settings(max_examples=100)
    def test_duplicate_label_produces_errors(self, valid_label: str) -> None:
        """
        GIVEN an AnnotatedSentence with a duplicated toxicity label,
        WHEN  validate_toxicity_labels is called,
        THEN  it returns at least one error message.
        """
        sentence = _make_sentence(toxicity_labels=[valid_label, valid_label])  # type: ignore[list-item]
        errors = validate_toxicity_labels(sentence)
        assert len(errors) >= 1, (
            f"Expected an error for duplicate label {valid_label!r}, got none"
        )


# ---------------------------------------------------------------------------
# Unit tests — validate_span_text_invariant
# ---------------------------------------------------------------------------


class TestValidateSpanTextInvariantUnit:
    """Unit tests for validate_span_text_invariant (Requirements 6.2, 11.2)."""

    def test_empty_ner_spans_passes(self) -> None:
        """
        GIVEN a sentence with no NER spans,
        WHEN  validate_span_text_invariant is called,
        THEN  it returns an empty error list.
        """
        sentence = _make_sentence(ner_spans=[], ner_annotator_spans=[[], [], []])
        assert validate_span_text_invariant(sentence) == []

    def test_correct_span_passes(self) -> None:
        """
        GIVEN a sentence with a span whose text matches sentence.text[start:end],
        WHEN  validate_span_text_invariant is called,
        THEN  it returns an empty error list.
        """
        text = "Gehlot ne Jaipur"
        span = EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")
        sentence = _make_sentence(text=text, ner_spans=[span])
        assert validate_span_text_invariant(sentence) == []

    def test_incorrect_span_text_fails(self) -> None:
        """
        GIVEN a sentence with a span whose text does NOT match sentence.text[start:end],
        WHEN  validate_span_text_invariant is called,
        THEN  it returns a non-empty error list.
        """
        text = "Gehlot ne Jaipur"
        bad_span = EntitySpan(start=0, end=6, entity_type="PER", text="Vasundhara")
        sentence = _make_sentence(text=text, ner_spans=[bad_span])
        errors = validate_span_text_invariant(sentence)
        assert len(errors) == 1
        assert "Vasundhara" in errors[0]
        assert "Gehlot" in errors[0]

    def test_multiple_spans_one_bad_reports_one_error(self) -> None:
        """
        GIVEN a sentence with two spans where one is correct and one is wrong,
        WHEN  validate_span_text_invariant is called,
        THEN  it returns exactly one error.
        """
        text = "Gehlot ne Jaipur"
        good_span = EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")
        bad_span = EntitySpan(start=10, end=16, entity_type="LOC", text="WRONG")
        sentence = _make_sentence(text=text, ner_spans=[good_span, bad_span])
        errors = validate_span_text_invariant(sentence)
        assert len(errors) == 1

    def test_annotator_spans_are_also_checked(self) -> None:
        """
        GIVEN a sentence where ner_spans is valid but one annotator span is wrong,
        WHEN  validate_span_text_invariant is called,
        THEN  it returns an error for the annotator span.
        """
        text = "Gehlot ne Jaipur"
        good_span = EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")
        bad_annotator_span = EntitySpan(start=0, end=6, entity_type="PER", text="WRONG")
        sentence = _make_sentence(
            text=text,
            ner_spans=[good_span],
            ner_annotator_spans=[[bad_annotator_span], [], []],
        )
        errors = validate_span_text_invariant(sentence)
        assert len(errors) == 1
        assert "ner_annotator_spans[0]" in errors[0]

    def test_error_message_contains_sentence_id(self) -> None:
        """
        GIVEN a sentence with a bad span,
        WHEN  validate_span_text_invariant is called,
        THEN  the error message contains the sentence_id.
        """
        sid = "test-sentence-id-123"
        text = "hello world"
        bad_span = EntitySpan(start=0, end=5, entity_type="PER", text="WRONG")
        sentence = _make_sentence(text=text, ner_spans=[bad_span], sentence_id=sid)
        errors = validate_span_text_invariant(sentence)
        assert any(sid in e for e in errors)

    def test_validate_all_span_text_invariants_empty_list(self) -> None:
        """
        GIVEN an empty list of sentences,
        WHEN  validate_all_span_text_invariants is called,
        THEN  it returns an empty error list.
        """
        assert validate_all_span_text_invariants([]) == []

    def test_validate_all_span_text_invariants_collects_all_errors(self) -> None:
        """
        GIVEN two sentences each with one bad span,
        WHEN  validate_all_span_text_invariants is called,
        THEN  it returns two errors (one per sentence).
        """
        text = "hello world"
        bad_span = EntitySpan(start=0, end=5, entity_type="PER", text="WRONG")
        s1 = _make_sentence(text=text, ner_spans=[bad_span])
        s2 = _make_sentence(text=text, ner_spans=[bad_span])
        errors = validate_all_span_text_invariants([s1, s2])
        assert len(errors) == 2

    def test_validate_all_span_text_invariants_valid_sentences_no_errors(self) -> None:
        """
        GIVEN two sentences with valid spans,
        WHEN  validate_all_span_text_invariants is called,
        THEN  it returns an empty error list.
        """
        text = "Gehlot ne Jaipur"
        good_span = EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")
        s1 = _make_sentence(text=text, ner_spans=[good_span])
        s2 = _make_sentence(text=text, ner_spans=[good_span])
        assert validate_all_span_text_invariants([s1, s2]) == []


# ---------------------------------------------------------------------------
# Unit tests — validate_toxicity_labels
# ---------------------------------------------------------------------------


class TestValidateToxicityLabelsUnit:
    """Unit tests for validate_toxicity_labels (Requirements 7.1, 7.2, 12.2)."""

    def test_empty_toxicity_labels_passes(self) -> None:
        """
        GIVEN a sentence with no toxicity labels (non-toxic),
        WHEN  validate_toxicity_labels is called,
        THEN  it returns an empty error list.
        """
        sentence = _make_sentence(toxicity_labels=[])
        assert validate_toxicity_labels(sentence) == []

    def test_all_valid_categories_pass(self) -> None:
        """
        GIVEN a sentence with all four valid toxicity categories (no duplicates),
        WHEN  validate_toxicity_labels is called,
        THEN  it returns an empty error list.
        """
        sentence = _make_sentence(
            toxicity_labels=["caste_slur", "religious", "gender", "general"]  # type: ignore[list-item]
        )
        assert validate_toxicity_labels(sentence) == []

    def test_single_valid_category_passes(self) -> None:
        """
        GIVEN a sentence with a single valid toxicity category,
        WHEN  validate_toxicity_labels is called,
        THEN  it returns an empty error list.
        """
        for cat in VALID_TOXICITY_CATEGORIES:
            sentence = _make_sentence(toxicity_labels=[cat])  # type: ignore[list-item]
            assert validate_toxicity_labels(sentence) == [], f"Failed for category {cat!r}"

    def test_invalid_category_fails(self) -> None:
        """
        GIVEN a sentence with an invalid toxicity category,
        WHEN  validate_toxicity_labels is called,
        THEN  it returns a non-empty error list.
        """
        sentence = _make_sentence(toxicity_labels=["hate_speech"])  # type: ignore[list-item]
        errors = validate_toxicity_labels(sentence)
        assert len(errors) >= 1
        assert "hate_speech" in errors[0]

    def test_duplicate_label_fails(self) -> None:
        """
        GIVEN a sentence with a duplicated toxicity label,
        WHEN  validate_toxicity_labels is called,
        THEN  it returns a non-empty error list.
        """
        sentence = _make_sentence(toxicity_labels=["religious", "religious"])  # type: ignore[list-item]
        errors = validate_toxicity_labels(sentence)
        assert len(errors) >= 1
        assert "religious" in errors[0]

    def test_invalid_and_duplicate_both_reported(self) -> None:
        """
        GIVEN a sentence with both an invalid category and a duplicate,
        WHEN  validate_toxicity_labels is called,
        THEN  it returns errors for both issues.
        """
        sentence = _make_sentence(
            toxicity_labels=["caste_slur", "caste_slur", "unknown_cat"]  # type: ignore[list-item]
        )
        errors = validate_toxicity_labels(sentence)
        assert len(errors) == 2

    def test_error_message_contains_sentence_id(self) -> None:
        """
        GIVEN a sentence with an invalid toxicity label,
        WHEN  validate_toxicity_labels is called,
        THEN  the error message contains the sentence_id.
        """
        sid = "test-sentence-id-456"
        sentence = _make_sentence(toxicity_labels=["bad_label"], sentence_id=sid)  # type: ignore[list-item]
        errors = validate_toxicity_labels(sentence)
        assert any(sid in e for e in errors)

    def test_validate_all_toxicity_labels_empty_list(self) -> None:
        """
        GIVEN an empty list of sentences,
        WHEN  validate_all_toxicity_labels is called,
        THEN  it returns an empty error list.
        """
        assert validate_all_toxicity_labels([]) == []

    def test_validate_all_toxicity_labels_collects_all_errors(self) -> None:
        """
        GIVEN two sentences each with an invalid toxicity label,
        WHEN  validate_all_toxicity_labels is called,
        THEN  it returns two errors (one per sentence).
        """
        s1 = _make_sentence(toxicity_labels=["bad1"])  # type: ignore[list-item]
        s2 = _make_sentence(toxicity_labels=["bad2"])  # type: ignore[list-item]
        errors = validate_all_toxicity_labels([s1, s2])
        assert len(errors) == 2

    def test_validate_all_toxicity_labels_valid_sentences_no_errors(self) -> None:
        """
        GIVEN two sentences with valid toxicity labels,
        WHEN  validate_all_toxicity_labels is called,
        THEN  it returns an empty error list.
        """
        s1 = _make_sentence(toxicity_labels=["caste_slur"])  # type: ignore[list-item]
        s2 = _make_sentence(toxicity_labels=[])
        assert validate_all_toxicity_labels([s1, s2]) == []
