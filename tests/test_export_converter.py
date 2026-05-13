"""
Tests for annotator_tool.export_converter.

Covers:
- Unit tests for Label Studio JSON export → AnnotatedSentence conversion
- Validation that JSON output matches HuggingFace Datasets schema
- Field type and structure verification

Requirements: 4.5
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from annotator_tool.export_converter import (
    HUGGINGFACE_SCHEMA_FIELDS,
    VALID_ENTITY_TYPES,
    VALID_PLATFORMS,
    VALID_SENTIMENTS,
    VALID_SPLITS,
    VALID_TOXICITY_CATEGORIES,
    annotated_sentence_to_dict,
    convert_ner_export,
    convert_sentiment_export,
    convert_toxicity_export,
    validate_huggingface_schema,
)
from models.data_models import AnnotatedSentence, EntitySpan, TokenLabel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FIXED_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
_ANNOTATED_TS = datetime(2024, 2, 1, 14, 0, 0, tzinfo=timezone.utc)


def _make_sentiment_task(
    sentence_id: str | None = None,
    text: str = "Gehlot ne Jaipur mein bada ailan kiya",
    platform: str = "twitter",
    labels: list[str] | None = None,
) -> dict:
    """Build a minimal Label Studio sentiment task dict."""
    if sentence_id is None:
        sentence_id = str(uuid.uuid4())
    if labels is None:
        labels = ["positive", "positive", "neutral"]

    annotations = [
        {
            "id": i + 1,
            "result": [
                {
                    "type": "choices",
                    "value": {"choices": [label]},
                }
            ],
        }
        for i, label in enumerate(labels)
    ]

    return {
        "id": 1,
        "data": {
            "sentence_id": sentence_id,
            "text": text,
            "platform": platform,
            "source_url": "https://twitter.com/example/status/123",
            "collected_at": "2024-01-15T10:30:00Z",
        },
        "annotations": annotations,
    }


def _make_ner_task(
    sentence_id: str | None = None,
    text: str = "Gehlot ne Jaipur mein bada ailan kiya",
    annotator_spans: list[list[tuple[int, int, str]]] | None = None,
) -> dict:
    """Build a minimal Label Studio NER task dict."""
    if sentence_id is None:
        sentence_id = str(uuid.uuid4())
    if annotator_spans is None:
        annotator_spans = [
            [(0, 6, "PER"), (10, 16, "LOC")],
            [(0, 6, "PER")],
            [(0, 6, "PER")],
        ]

    annotations = []
    for i, spans in enumerate(annotator_spans):
        results = [
            {
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "labels": [entity_type],
                },
            }
            for start, end, entity_type in spans
        ]
        annotations.append({"id": i + 1, "result": results})

    return {
        "id": 2,
        "data": {
            "sentence_id": sentence_id,
            "text": text,
            "platform": "twitter",
            "source_url": "https://twitter.com/example/status/456",
            "collected_at": "2024-01-15T10:30:00Z",
        },
        "annotations": annotations,
    }


def _make_toxicity_task(
    sentence_id: str | None = None,
    text: str = "Yeh bahut bura hai",
    annotator_labels: list[list[str]] | None = None,
) -> dict:
    """Build a minimal Label Studio toxicity task dict."""
    if sentence_id is None:
        sentence_id = str(uuid.uuid4())
    if annotator_labels is None:
        annotator_labels = [
            ["caste_slur", "general"],
            ["caste_slur"],
            ["caste_slur", "religious"],
        ]

    annotations = [
        {
            "id": i + 1,
            "result": [
                {
                    "type": "choices",
                    "value": {"choices": labels},
                }
            ],
        }
        for i, labels in enumerate(annotator_labels)
    ]

    return {
        "id": 3,
        "data": {
            "sentence_id": sentence_id,
            "text": text,
            "platform": "sharechat",
            "source_url": "https://sharechat.com/post/789",
            "collected_at": "2024-01-15T10:30:00Z",
        },
        "annotations": annotations,
    }


def _make_valid_annotated_sentence(
    sentence_id: str | None = None,
) -> AnnotatedSentence:
    """Build a valid AnnotatedSentence for schema validation tests."""
    if sentence_id is None:
        sentence_id = str(uuid.uuid4())
    text = "Gehlot ne Jaipur mein bada ailan kiya"
    return AnnotatedSentence(
        sentence_id=sentence_id,
        text=text,
        platform="twitter",
        split="train",
        sentiment="positive",
        sentiment_annotator_labels=["positive", "positive", "neutral"],
        ner_spans=[EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")],
        ner_annotator_spans=[
            [EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")],
            [EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")],
            [EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")],
        ],
        toxicity_labels=[],
        toxicity_annotator_labels=[[], [], []],
        token_language_labels=[
            TokenLabel(token="Gehlot", label="RAJ", confidence=0.97),
        ],
        source_url="https://twitter.com/example/status/123",
        collected_at=_FIXED_TS,
        annotated_at=_ANNOTATED_TS,
    )


# ---------------------------------------------------------------------------
# Unit tests — sentiment export converter
# ---------------------------------------------------------------------------


class TestConvertSentimentExport:
    """Unit tests for convert_sentiment_export (Requirement 4.5)."""

    def test_converts_single_task_to_annotated_sentence(self):
        """
        GIVEN a single Label Studio sentiment task,
        WHEN  convert_sentiment_export is called,
        THEN  one AnnotatedSentence is returned.
        """
        task = _make_sentiment_task()
        results = convert_sentiment_export([task], split="train")
        assert len(results) == 1
        assert isinstance(results[0], AnnotatedSentence)

    def test_sentence_id_is_preserved(self):
        """
        GIVEN a task with a specific sentence_id,
        WHEN  convert_sentiment_export is called,
        THEN  the sentence_id is preserved in the output.
        """
        sid = str(uuid.uuid4())
        task = _make_sentiment_task(sentence_id=sid)
        results = convert_sentiment_export([task])
        assert results[0].sentence_id == sid

    def test_text_is_preserved(self):
        """
        GIVEN a task with specific text,
        WHEN  convert_sentiment_export is called,
        THEN  the text is preserved (NFC-normalised) in the output.
        """
        text = "Gehlot ne Jaipur mein bada ailan kiya"
        task = _make_sentiment_task(text=text)
        results = convert_sentiment_export([task])
        assert results[0].text == text

    def test_majority_vote_sentiment_is_computed(self):
        """
        GIVEN labels ["positive", "positive", "neutral"] (2-1 split),
        WHEN  convert_sentiment_export is called,
        THEN  the gold sentiment is "positive".
        """
        task = _make_sentiment_task(labels=["positive", "positive", "neutral"])
        results = convert_sentiment_export([task])
        assert results[0].sentiment == "positive"

    def test_annotator_labels_are_stored(self):
        """
        GIVEN 3 annotator labels,
        WHEN  convert_sentiment_export is called,
        THEN  all 3 raw labels are stored in sentiment_annotator_labels.
        """
        labels = ["positive", "positive", "neutral"]
        task = _make_sentiment_task(labels=labels)
        results = convert_sentiment_export([task])
        assert results[0].sentiment_annotator_labels == labels

    def test_split_is_assigned(self):
        """
        GIVEN split="validation",
        WHEN  convert_sentiment_export is called,
        THEN  the output sentence has split="validation".
        """
        task = _make_sentiment_task()
        results = convert_sentiment_export([task], split="validation")
        assert results[0].split == "validation"

    def test_platform_is_preserved(self):
        """
        GIVEN a task with platform="sharechat",
        WHEN  convert_sentiment_export is called,
        THEN  the output sentence has platform="sharechat".
        """
        task = _make_sentiment_task(platform="sharechat")
        results = convert_sentiment_export([task])
        assert results[0].platform == "sharechat"

    def test_empty_task_list_returns_empty(self):
        """
        GIVEN an empty task list,
        WHEN  convert_sentiment_export is called,
        THEN  an empty list is returned.
        """
        results = convert_sentiment_export([])
        assert results == []

    def test_multiple_tasks_converted(self):
        """
        GIVEN 3 tasks,
        WHEN  convert_sentiment_export is called,
        THEN  3 AnnotatedSentence objects are returned.
        """
        tasks = [_make_sentiment_task() for _ in range(3)]
        results = convert_sentiment_export(tasks)
        assert len(results) == 3

    def test_collected_at_is_parsed(self):
        """
        GIVEN a task with collected_at="2024-01-15T10:30:00Z",
        WHEN  convert_sentiment_export is called,
        THEN  collected_at is a timezone-aware datetime.
        """
        task = _make_sentiment_task()
        results = convert_sentiment_export([task])
        assert results[0].collected_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Unit tests — NER export converter
# ---------------------------------------------------------------------------


class TestConvertNERExport:
    """Unit tests for convert_ner_export (Requirement 4.5)."""

    def test_converts_single_task_to_annotated_sentence(self):
        """
        GIVEN a single Label Studio NER task,
        WHEN  convert_ner_export is called,
        THEN  one AnnotatedSentence is returned.
        """
        task = _make_ner_task()
        results = convert_ner_export([task])
        assert len(results) == 1
        assert isinstance(results[0], AnnotatedSentence)

    def test_majority_vote_ner_spans_are_computed(self):
        """
        GIVEN span (0, 6, "PER") agreed by 2 of 3 annotators,
        WHEN  convert_ner_export is called,
        THEN  the gold ner_spans contains that span.
        """
        task = _make_ner_task(
            text="Gehlot ne Jaipur mein bada ailan kiya",
            annotator_spans=[
                [(0, 6, "PER")],
                [(0, 6, "PER")],
                [],
            ],
        )
        results = convert_ner_export([task])
        assert len(results[0].ner_spans) == 1
        assert results[0].ner_spans[0].start == 0
        assert results[0].ner_spans[0].end == 6
        assert results[0].ner_spans[0].entity_type == "PER"

    def test_span_agreed_by_only_one_annotator_excluded(self):
        """
        GIVEN a span marked by only 1 of 3 annotators,
        WHEN  convert_ner_export is called,
        THEN  the gold ner_spans does NOT contain that span.
        """
        task = _make_ner_task(
            annotator_spans=[
                [(0, 6, "PER")],
                [],
                [],
            ],
        )
        results = convert_ner_export([task])
        assert results[0].ner_spans == []

    def test_ner_annotator_spans_are_stored(self):
        """
        GIVEN 3 annotators with different spans,
        WHEN  convert_ner_export is called,
        THEN  all 3 raw span sets are stored in ner_annotator_spans.
        """
        annotator_spans = [
            [(0, 6, "PER")],
            [(0, 6, "PER"), (10, 16, "LOC")],
            [],
        ]
        task = _make_ner_task(annotator_spans=annotator_spans)
        results = convert_ner_export([task])
        assert len(results[0].ner_annotator_spans) == 3

    def test_zero_spans_allowed(self):
        """
        GIVEN all annotators mark zero spans,
        WHEN  convert_ner_export is called,
        THEN  ner_spans is empty (NER allows zero spans — Requirement 4.4).
        """
        task = _make_ner_task(annotator_spans=[[], [], []])
        results = convert_ner_export([task])
        assert results[0].ner_spans == []

    def test_span_text_is_derived_from_sentence_text(self):
        """
        GIVEN a span (0, 6) on text "Gehlot ne Jaipur mein bada ailan kiya",
        WHEN  convert_ner_export is called,
        THEN  the span text is "Gehlot".
        """
        text = "Gehlot ne Jaipur mein bada ailan kiya"
        task = _make_ner_task(
            text=text,
            annotator_spans=[
                [(0, 6, "PER")],
                [(0, 6, "PER")],
                [(0, 6, "PER")],
            ],
        )
        results = convert_ner_export([task])
        assert results[0].ner_spans[0].text == "Gehlot"


# ---------------------------------------------------------------------------
# Unit tests — toxicity export converter
# ---------------------------------------------------------------------------


class TestConvertToxicityExport:
    """Unit tests for convert_toxicity_export (Requirement 4.5)."""

    def test_converts_single_task_to_annotated_sentence(self):
        """
        GIVEN a single Label Studio toxicity task,
        WHEN  convert_toxicity_export is called,
        THEN  one AnnotatedSentence is returned.
        """
        task = _make_toxicity_task()
        results = convert_toxicity_export([task])
        assert len(results) == 1
        assert isinstance(results[0], AnnotatedSentence)

    def test_majority_vote_toxicity_labels_are_computed(self):
        """
        GIVEN "caste_slur" marked by 2 of 3 annotators,
        WHEN  convert_toxicity_export is called,
        THEN  "caste_slur" is in the gold toxicity_labels.
        """
        task = _make_toxicity_task(
            annotator_labels=[
                ["caste_slur"],
                ["caste_slur"],
                [],
            ]
        )
        results = convert_toxicity_export([task])
        assert "caste_slur" in results[0].toxicity_labels

    def test_category_agreed_by_only_one_annotator_excluded(self):
        """
        GIVEN "religious" marked by only 1 of 3 annotators,
        WHEN  convert_toxicity_export is called,
        THEN  "religious" is NOT in the gold toxicity_labels.
        """
        task = _make_toxicity_task(
            annotator_labels=[
                ["religious"],
                [],
                [],
            ]
        )
        results = convert_toxicity_export([task])
        assert "religious" not in results[0].toxicity_labels

    def test_none_sentinel_is_filtered_out(self):
        """
        GIVEN annotators selecting "none" (non-toxic sentinel),
        WHEN  convert_toxicity_export is called,
        THEN  "none" does NOT appear in toxicity_labels.
        """
        task = _make_toxicity_task(
            annotator_labels=[
                ["none"],
                ["none"],
                ["none"],
            ]
        )
        results = convert_toxicity_export([task])
        assert "none" not in results[0].toxicity_labels
        assert results[0].toxicity_labels == []

    def test_non_toxic_sentence_has_empty_toxicity_labels(self):
        """
        GIVEN all annotators select "none" (non-toxic),
        WHEN  convert_toxicity_export is called,
        THEN  toxicity_labels is empty.
        """
        task = _make_toxicity_task(
            annotator_labels=[["none"], ["none"], ["none"]]
        )
        results = convert_toxicity_export([task])
        assert results[0].toxicity_labels == []

    def test_toxicity_annotator_labels_are_stored(self):
        """
        GIVEN 3 annotators with different toxicity labels,
        WHEN  convert_toxicity_export is called,
        THEN  all 3 raw label sets are stored in toxicity_annotator_labels.
        """
        annotator_labels = [
            ["caste_slur", "general"],
            ["caste_slur"],
            ["caste_slur", "religious"],
        ]
        task = _make_toxicity_task(annotator_labels=annotator_labels)
        results = convert_toxicity_export([task])
        assert len(results[0].toxicity_annotator_labels) == 3


# ---------------------------------------------------------------------------
# Unit tests — HuggingFace schema validation
# ---------------------------------------------------------------------------


class TestValidateHuggingFaceSchema:
    """Unit tests for validate_huggingface_schema (Requirement 4.5)."""

    def test_valid_sentence_passes_validation(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  validate_huggingface_schema is called,
        THEN  no errors are returned.
        """
        sentence = _make_valid_annotated_sentence()
        errors = validate_huggingface_schema([sentence])
        assert errors == []

    def test_multiple_valid_sentences_pass_validation(self):
        """
        GIVEN multiple valid AnnotatedSentence objects,
        WHEN  validate_huggingface_schema is called,
        THEN  no errors are returned.
        """
        sentences = [_make_valid_annotated_sentence() for _ in range(5)]
        errors = validate_huggingface_schema(sentences)
        assert errors == []

    def test_invalid_platform_produces_error(self):
        """
        GIVEN a sentence with an invalid platform value,
        WHEN  validate_huggingface_schema is called,
        THEN  an error is returned mentioning 'platform'.
        """
        sentence = _make_valid_annotated_sentence()
        sentence.platform = "facebook"  # type: ignore[assignment]
        errors = validate_huggingface_schema([sentence])
        assert any("platform" in e for e in errors)

    def test_invalid_sentiment_produces_error(self):
        """
        GIVEN a sentence with an invalid sentiment value,
        WHEN  validate_huggingface_schema is called,
        THEN  an error is returned mentioning 'sentiment'.
        """
        sentence = _make_valid_annotated_sentence()
        sentence.sentiment = "angry"  # type: ignore[assignment]
        errors = validate_huggingface_schema([sentence])
        assert any("sentiment" in e for e in errors)

    def test_invalid_toxicity_category_produces_error(self):
        """
        GIVEN a sentence with an invalid toxicity category,
        WHEN  validate_huggingface_schema is called,
        THEN  an error is returned mentioning the invalid category.
        """
        sentence = _make_valid_annotated_sentence()
        sentence.toxicity_labels = ["invalid_category"]  # type: ignore[list-item]
        errors = validate_huggingface_schema([sentence])
        assert any("invalid_category" in e for e in errors)

    def test_invalid_split_produces_error(self):
        """
        GIVEN a sentence with an invalid split value,
        WHEN  validate_huggingface_schema is called,
        THEN  an error is returned mentioning 'split'.
        """
        sentence = _make_valid_annotated_sentence()
        sentence.split = "dev"  # type: ignore[assignment]
        errors = validate_huggingface_schema([sentence])
        assert any("split" in e for e in errors)

    def test_empty_list_passes_validation(self):
        """
        GIVEN an empty list,
        WHEN  validate_huggingface_schema is called,
        THEN  no errors are returned.
        """
        errors = validate_huggingface_schema([])
        assert errors == []


# ---------------------------------------------------------------------------
# Unit tests — annotated_sentence_to_dict
# ---------------------------------------------------------------------------


class TestAnnotatedSentenceToDict:
    """Unit tests for annotated_sentence_to_dict (Requirement 4.5)."""

    def test_output_contains_all_schema_fields(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  the output dict contains all HuggingFace schema fields.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert set(result.keys()) == HUGGINGFACE_SCHEMA_FIELDS

    def test_sentence_id_is_string(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  sentence_id is a string.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["sentence_id"], str)

    def test_text_is_string(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  text is a string.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["text"], str)

    def test_ner_spans_is_list_of_dicts(self):
        """
        GIVEN a sentence with NER spans,
        WHEN  annotated_sentence_to_dict is called,
        THEN  ner_spans is a list of dicts with start, end, entity_type, text.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["ner_spans"], list)
        for span in result["ner_spans"]:
            assert isinstance(span, dict)
            assert "start" in span
            assert "end" in span
            assert "entity_type" in span
            assert "text" in span

    def test_toxicity_labels_is_list(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  toxicity_labels is a list.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["toxicity_labels"], list)

    def test_token_language_labels_is_list_of_dicts(self):
        """
        GIVEN a sentence with token language labels,
        WHEN  annotated_sentence_to_dict is called,
        THEN  token_language_labels is a list of dicts with token, label, confidence.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["token_language_labels"], list)
        for tl in result["token_language_labels"]:
            assert isinstance(tl, dict)
            assert "token" in tl
            assert "label" in tl
            assert "confidence" in tl

    def test_collected_at_is_iso_string(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  collected_at is an ISO 8601 string.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["collected_at"], str)
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(result["collected_at"].replace("Z", "+00:00"))
        assert parsed is not None

    def test_annotated_at_is_iso_string(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  annotated_at is an ISO 8601 string.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["annotated_at"], str)

    def test_platform_value_is_valid(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  platform is one of the valid platform values.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert result["platform"] in VALID_PLATFORMS

    def test_sentiment_value_is_valid(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  sentiment is one of the valid sentiment values.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert result["sentiment"] in VALID_SENTIMENTS

    def test_split_value_is_valid(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  split is one of the valid split values.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert result["split"] in VALID_SPLITS

    def test_sentiment_annotator_labels_is_list_of_strings(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  sentiment_annotator_labels is a list of strings.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["sentiment_annotator_labels"], list)
        for label in result["sentiment_annotator_labels"]:
            assert isinstance(label, str)

    def test_ner_annotator_spans_is_list_of_lists(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  ner_annotator_spans is a list of lists.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["ner_annotator_spans"], list)
        for annotator_spans in result["ner_annotator_spans"]:
            assert isinstance(annotator_spans, list)

    def test_toxicity_annotator_labels_is_list_of_lists(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  toxicity_annotator_labels is a list of lists.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert isinstance(result["toxicity_annotator_labels"], list)
        for ann_labels in result["toxicity_annotator_labels"]:
            assert isinstance(ann_labels, list)

    def test_round_trip_preserves_sentence_id(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  the sentence_id in the dict matches the original.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert result["sentence_id"] == sentence.sentence_id

    def test_round_trip_preserves_text(self):
        """
        GIVEN a valid AnnotatedSentence,
        WHEN  annotated_sentence_to_dict is called,
        THEN  the text in the dict matches the original.
        """
        sentence = _make_valid_annotated_sentence()
        result = annotated_sentence_to_dict(sentence)
        assert result["text"] == sentence.text
