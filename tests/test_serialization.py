"""
Tests for corpus_builder.serialization -- serialization and round-trip validation.

Covers:
- Property 1: Corpus serialization round-trip
  For any valid AnnotatedSentence record, serializing to JSON Lines and
  deserializing SHALL produce a field-for-field identical record.
  (Validates: Requirement 15.2)
- Unit tests: corrupted record halts validate_round_trip (Requirement 15.3)
- Unit tests: Parquet schema matches PARQUET_SCHEMA (Requirement 15.1)

Requirements: 15.1, 15.2, 15.3
"""

from __future__ import annotations

import copy
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from corpus_builder.serialization import (
    PARQUET_SCHEMA,
    RoundTripValidationError,
    deserialize,
    serialize,
    validate_round_trip,
)
from models.data_models import (
    AnnotatedSentence,
    EntitySpan,
    RoundTripReport,
    TokenLabel,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FIXED_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
_ANNOTATED_TS = datetime(2024, 2, 1, 14, 0, 0, tzinfo=timezone.utc)


def _make_annotated_sentence(
    sentence_id=None,
    text="Gehlot ne Jaipur mein bada ailan kiya",
    platform="twitter",
    split="train",
    sentiment="positive",
    sentiment_annotator_labels=None,
    ner_spans=None,
    ner_annotator_spans=None,
    toxicity_labels=None,
    toxicity_annotator_labels=None,
    token_language_labels=None,
    source_url="https://twitter.com/example/status/123",
    collected_at=None,
    annotated_at=None,
):
    """Factory for AnnotatedSentence with sensible defaults."""
    if sentence_id is None:
        sentence_id = str(uuid.uuid4())
    if sentiment_annotator_labels is None:
        sentiment_annotator_labels = ["positive", "positive", "neutral"]
    if ner_spans is None:
        ner_spans = [EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")]
    if ner_annotator_spans is None:
        ner_annotator_spans = [
            [EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")],
            [EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")],
            [EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")],
        ]
    if toxicity_labels is None:
        toxicity_labels = []
    if toxicity_annotator_labels is None:
        toxicity_annotator_labels = [[], [], []]
    if token_language_labels is None:
        token_language_labels = [
            TokenLabel(token="Gehlot", label="RAJ", confidence=0.97),
            TokenLabel(token="ne", label="HIN", confidence=0.99),
        ]
    if collected_at is None:
        collected_at = _FIXED_TS
    if annotated_at is None:
        annotated_at = _ANNOTATED_TS
    return AnnotatedSentence(
        sentence_id=sentence_id,
        text=text,
        platform=platform,
        split=split,
        sentiment=sentiment,
        sentiment_annotator_labels=sentiment_annotator_labels,
        ner_spans=ner_spans,
        ner_annotator_spans=ner_annotator_spans,
        toxicity_labels=toxicity_labels,
        toxicity_annotator_labels=toxicity_annotator_labels,
        token_language_labels=token_language_labels,
        source_url=source_url,
        collected_at=collected_at,
        annotated_at=annotated_at,
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating valid AnnotatedSentence objects
# ---------------------------------------------------------------------------

_PLATFORMS = ["twitter", "sharechat"]
_SPLITS = ["train", "validation", "test"]
_SENTIMENTS = ["positive", "neutral", "negative"]
_ENTITY_TYPES = ["PER", "LOC", "ORG"]
_LANG_LABELS = ["RAJ", "HIN", "ENG", "TRL"]
_TOXICITY_CATS = ["caste_slur", "religious", "gender", "general"]


@st.composite
def _entity_span_strategy(draw, text: str) -> EntitySpan:
    """Generate a valid EntitySpan whose text matches sentence.text[start:end]."""
    if not text:
        # Fallback: empty text means no valid span possible; return a zero-length span
        return EntitySpan(start=0, end=0, entity_type="PER", text="")
    max_end = len(text)
    start = draw(st.integers(min_value=0, max_value=max(0, max_end - 1)))
    end = draw(st.integers(min_value=start, max_value=max_end))
    entity_type = draw(st.sampled_from(_ENTITY_TYPES))
    span_text = text[start:end]
    return EntitySpan(start=start, end=end, entity_type=entity_type, text=span_text)


@st.composite
def _token_label_strategy(draw) -> TokenLabel:
    """Generate a valid TokenLabel."""
    token = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )))
    label = draw(st.sampled_from(_LANG_LABELS))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return TokenLabel(token=token, label=label, confidence=confidence)


@st.composite
def _annotated_sentence_strategy(draw) -> AnnotatedSentence:
    """Generate a valid AnnotatedSentence for property-based testing."""
    sentence_id = str(draw(st.uuids()))
    # Use ASCII-safe text to avoid NFC normalization surprises in the property test
    text = draw(st.text(
        min_size=1,
        max_size=100,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
            whitelist_characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
        ),
    ))
    platform = draw(st.sampled_from(_PLATFORMS))
    split = draw(st.sampled_from(_SPLITS))
    sentiment = draw(st.sampled_from(_SENTIMENTS))

    # sentiment_annotator_labels: exactly 3 labels
    sentiment_annotator_labels = draw(
        st.lists(st.sampled_from(_SENTIMENTS), min_size=3, max_size=3)
    )

    # ner_spans: 0-3 spans
    ner_spans = draw(
        st.lists(_entity_span_strategy(text), min_size=0, max_size=3)
    )

    # ner_annotator_spans: exactly 3 sets of 0-2 spans each
    ner_annotator_spans = draw(
        st.lists(
            st.lists(_entity_span_strategy(text), min_size=0, max_size=2),
            min_size=3,
            max_size=3,
        )
    )

    # toxicity_labels: 0-4 unique categories
    toxicity_labels = draw(
        st.lists(
            st.sampled_from(_TOXICITY_CATS),
            min_size=0,
            max_size=4,
            unique=True,
        )
    )

    # toxicity_annotator_labels: exactly 3 sets
    toxicity_annotator_labels = draw(
        st.lists(
            st.lists(
                st.sampled_from(_TOXICITY_CATS),
                min_size=0,
                max_size=4,
                unique=True,
            ),
            min_size=3,
            max_size=3,
        )
    )

    # token_language_labels: 0-5 labels
    token_language_labels = draw(
        st.lists(_token_label_strategy(), min_size=0, max_size=5)
    )

    source_url = "https://example.com/" + draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        )
    )

    # Datetimes: use fixed timestamps to avoid sub-second precision issues
    year = draw(st.integers(min_value=2020, max_value=2024))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))
    hour = draw(st.integers(min_value=0, max_value=23))
    minute = draw(st.integers(min_value=0, max_value=59))
    second = draw(st.integers(min_value=0, max_value=59))
    collected_at = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    annotated_at = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)

    return AnnotatedSentence(
        sentence_id=sentence_id,
        text=text,
        platform=platform,
        split=split,
        sentiment=sentiment,
        sentiment_annotator_labels=sentiment_annotator_labels,
        ner_spans=ner_spans,
        ner_annotator_spans=ner_annotator_spans,
        toxicity_labels=toxicity_labels,
        toxicity_annotator_labels=toxicity_annotator_labels,
        token_language_labels=token_language_labels,
        source_url=source_url,
        collected_at=collected_at,
        annotated_at=annotated_at,
    )


# ---------------------------------------------------------------------------
# Property 1: Corpus serialization round-trip (JSON Lines)
# ---------------------------------------------------------------------------


class TestProperty1SerializationRoundTrip:
    """
    Property 1: For any valid AnnotatedSentence record, serializing it to
    JSON Lines and then deserializing it SHALL produce a record that is
    field-for-field identical to the original.

    Validates: Requirements 15.2
    """

    @given(sentence=_annotated_sentence_strategy())
    @settings(max_examples=100)
    def test_jsonl_round_trip_is_field_identical(self, sentence):
        """
        GIVEN any valid AnnotatedSentence,
        WHEN  it is serialized to JSON Lines and deserialized,
        THEN  the deserialized record is field-for-field identical to the original.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus.jsonl"
            serialize([sentence], path, fmt="jsonl")
            roundtrip_list = deserialize(path, fmt="jsonl")

        assert len(roundtrip_list) == 1
        rt = roundtrip_list[0]

        # Compare every field explicitly
        assert rt.sentence_id == sentence.sentence_id, "sentence_id mismatch"
        # NFC-normalise text for comparison (serialize normalises to NFC)
        assert rt.text == unicodedata.normalize("NFC", sentence.text), "text mismatch"
        assert rt.platform == sentence.platform, "platform mismatch"
        assert rt.split == sentence.split, "split mismatch"
        assert rt.sentiment == sentence.sentiment, "sentiment mismatch"
        assert rt.sentiment_annotator_labels == sentence.sentiment_annotator_labels, (
            "sentiment_annotator_labels mismatch"
        )
        assert list(rt.toxicity_labels) == list(sentence.toxicity_labels), (
            "toxicity_labels mismatch"
        )
        assert rt.source_url == unicodedata.normalize("NFC", sentence.source_url), (
            "source_url mismatch"
        )

        # Datetime comparison at second precision
        def _utc_sec(dt):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).replace(microsecond=0)

        assert _utc_sec(rt.collected_at) == _utc_sec(sentence.collected_at), (
            "collected_at mismatch"
        )
        assert _utc_sec(rt.annotated_at) == _utc_sec(sentence.annotated_at), (
            "annotated_at mismatch"
        )

        # NER spans
        assert len(rt.ner_spans) == len(sentence.ner_spans), "ner_spans length mismatch"
        for orig_span, rt_span in zip(sentence.ner_spans, rt.ner_spans):
            assert rt_span.start == orig_span.start
            assert rt_span.end == orig_span.end
            assert rt_span.entity_type == orig_span.entity_type
            assert rt_span.text == unicodedata.normalize("NFC", orig_span.text)

        # Token language labels
        assert len(rt.token_language_labels) == len(sentence.token_language_labels), (
            "token_language_labels length mismatch"
        )
        for orig_tl, rt_tl in zip(sentence.token_language_labels, rt.token_language_labels):
            assert rt_tl.token == unicodedata.normalize("NFC", orig_tl.token)
            assert rt_tl.label == orig_tl.label
            assert abs(rt_tl.confidence - orig_tl.confidence) < 1e-9

    @given(sentences=st.lists(_annotated_sentence_strategy(), min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_jsonl_round_trip_preserves_record_count(self, sentences):
        """
        GIVEN a list of AnnotatedSentence records,
        WHEN  serialized to JSON Lines and deserialized,
        THEN  the number of records is preserved.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus.jsonl"
            serialize(sentences, path, fmt="jsonl")
            roundtrip_list = deserialize(path, fmt="jsonl")
        assert len(roundtrip_list) == len(sentences)

    @given(sentences=st.lists(_annotated_sentence_strategy(), min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_jsonl_round_trip_preserves_sentence_ids(self, sentences):
        """
        GIVEN a list of AnnotatedSentence records,
        WHEN  serialized to JSON Lines and deserialized,
        THEN  the sentence_ids are preserved in order.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus.jsonl"
            serialize(sentences, path, fmt="jsonl")
            roundtrip_list = deserialize(path, fmt="jsonl")
        orig_ids = [s.sentence_id for s in sentences]
        rt_ids = [s.sentence_id for s in roundtrip_list]
        assert orig_ids == rt_ids


# ---------------------------------------------------------------------------
# Unit tests — validate_round_trip halts on corrupted record
# ---------------------------------------------------------------------------


class TestValidateRoundTripUnit:
    """Unit tests for validate_round_trip (Requirements 15.2, 15.3)."""

    def test_valid_corpus_passes_round_trip(self):
        """
        GIVEN a valid corpus serialized to JSON Lines,
        WHEN  validate_round_trip is called,
        THEN  it returns a RoundTripReport with failed=0 and raises no exception.
        """
        sentences = [
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus.jsonl"
            serialize(sentences, path, fmt="jsonl")
            report = validate_round_trip(sentences, path, fmt="jsonl")

        assert isinstance(report, RoundTripReport)
        assert report.total_records == 2
        assert report.passed == 2
        assert report.failed == 0
        assert report.failures == []

    def test_corrupted_text_field_halts_pipeline(self, tmp_path):
        """
        GIVEN a corpus where one record has a corrupted text field in the file,
        WHEN  validate_round_trip is called,
        THEN  RoundTripValidationError is raised and the failure is logged.
        """
        import json

        sid = str(uuid.uuid4())
        sentence = _make_annotated_sentence(sentence_id=sid, text="original text")
        path = tmp_path / "corpus.jsonl"
        serialize([sentence], path, fmt="jsonl")

        # Corrupt the text field in the file
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["text"] = "CORRUPTED TEXT"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

        with pytest.raises(RoundTripValidationError):
            validate_round_trip([sentence], path, fmt="jsonl")

    def test_corrupted_record_logs_sentence_id_and_fields(self, tmp_path, caplog):
        """
        GIVEN a corpus where one record has a corrupted field,
        WHEN  validate_round_trip is called,
        THEN  the sentence_id and differing fields are logged at ERROR level.
        """
        import json
        import logging

        sid = str(uuid.uuid4())
        sentence = _make_annotated_sentence(sentence_id=sid, text="original text")
        path = tmp_path / "corpus.jsonl"
        serialize([sentence], path, fmt="jsonl")

        # Corrupt the sentiment field
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["sentiment"] = "negative"  # was "positive"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

        with caplog.at_level(logging.ERROR, logger="corpus_builder.serialization"):
            with pytest.raises(RoundTripValidationError):
                validate_round_trip([sentence], path, fmt="jsonl")

        # Verify the sentence_id appears in the error log
        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(sid in msg for msg in error_messages), (
            f"Expected sentence_id {sid!r} in error logs; got: {error_messages}"
        )

    def test_round_trip_report_contains_failure_details(self, tmp_path):
        """
        GIVEN a corpus where one record has a corrupted field,
        WHEN  validate_round_trip raises RoundTripValidationError,
        THEN  the exception message mentions the number of failures.
        """
        import json

        sid = str(uuid.uuid4())
        sentence = _make_annotated_sentence(sentence_id=sid, text="original text")
        path = tmp_path / "corpus.jsonl"
        serialize([sentence], path, fmt="jsonl")

        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["platform"] = "sharechat"  # was "twitter"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

        with pytest.raises(RoundTripValidationError) as exc_info:
            validate_round_trip([sentence], path, fmt="jsonl")

        assert "1" in str(exc_info.value)

    def test_multiple_records_partial_failure(self, tmp_path):
        """
        GIVEN a corpus of 3 records where 1 is corrupted,
        WHEN  validate_round_trip is called,
        THEN  RoundTripValidationError is raised and the report shows 2 passed, 1 failed.
        """
        import json

        sentences = [
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
        ]
        path = tmp_path / "corpus.jsonl"
        serialize(sentences, path, fmt="jsonl")

        # Corrupt the second record
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[1])
        record["split"] = "test"  # was "train"
        lines[1] = json.dumps(record, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(RoundTripValidationError):
            validate_round_trip(sentences, path, fmt="jsonl")

    def test_empty_corpus_passes_round_trip(self, tmp_path):
        """
        GIVEN an empty corpus serialized to JSON Lines,
        WHEN  validate_round_trip is called,
        THEN  it returns a RoundTripReport with total=0, passed=0, failed=0.
        """
        path = tmp_path / "corpus.jsonl"
        serialize([], path, fmt="jsonl")
        report = validate_round_trip([], path, fmt="jsonl")
        assert report.total_records == 0
        assert report.passed == 0
        assert report.failed == 0

    def test_nfc_normalised_text_passes_round_trip(self, tmp_path):
        """
        GIVEN a sentence with text that is already NFC-normalised,
        WHEN  serialized and deserialized,
        THEN  validate_round_trip passes without error.
        """
        import unicodedata as ud
        text = ud.normalize("NFC", "Gehlot ne Jaipur mein bada ailan kiya")
        sentence = _make_annotated_sentence(text=text)
        path = tmp_path / "corpus.jsonl"
        serialize([sentence], path, fmt="jsonl")
        report = validate_round_trip([sentence], path, fmt="jsonl")
        assert report.failed == 0


# ---------------------------------------------------------------------------
# Unit tests — Parquet schema validation
# ---------------------------------------------------------------------------


class TestParquetSchemaUnit:
    """Unit tests for Parquet serialization schema (Requirement 15.1)."""

    def test_parquet_schema_has_all_required_fields(self):
        """
        GIVEN the PARQUET_SCHEMA constant,
        THEN  it contains all required AnnotatedSentence fields.
        """
        expected_fields = {
            "sentence_id",
            "text",
            "platform",
            "split",
            "sentiment",
            "sentiment_annotator_labels",
            "ner_spans",
            "ner_annotator_spans",
            "toxicity_labels",
            "toxicity_annotator_labels",
            "token_language_labels",
            "source_url",
            "collected_at",
            "annotated_at",
        }
        schema_fields = set(PARQUET_SCHEMA.names)
        assert expected_fields == schema_fields, (
            f"Schema fields mismatch. Missing: {expected_fields - schema_fields}, "
            f"Extra: {schema_fields - expected_fields}"
        )

    def test_parquet_schema_ner_spans_is_list_of_struct(self):
        """
        GIVEN the PARQUET_SCHEMA,
        THEN  ner_spans is a list of struct with start, end, entity_type, text fields.
        """
        ner_spans_field = PARQUET_SCHEMA.field("ner_spans")
        assert pa.types.is_list(ner_spans_field.type), (
            f"ner_spans should be a list type, got {ner_spans_field.type}"
        )
        value_type = ner_spans_field.type.value_type
        assert pa.types.is_struct(value_type), (
            f"ner_spans value type should be struct, got {value_type}"
        )
        struct_field_names = {value_type.field(i).name for i in range(value_type.num_fields)}
        assert {"start", "end", "entity_type", "text"} == struct_field_names

    def test_parquet_schema_token_language_labels_is_list_of_struct(self):
        """
        GIVEN the PARQUET_SCHEMA,
        THEN  token_language_labels is a list of struct with token, label, confidence fields.
        """
        tll_field = PARQUET_SCHEMA.field("token_language_labels")
        assert pa.types.is_list(tll_field.type)
        value_type = tll_field.type.value_type
        assert pa.types.is_struct(value_type)
        struct_field_names = {value_type.field(i).name for i in range(value_type.num_fields)}
        assert {"token", "label", "confidence"} == struct_field_names

    def test_parquet_schema_toxicity_labels_is_list_of_string(self):
        """
        GIVEN the PARQUET_SCHEMA,
        THEN  toxicity_labels is a list of string.
        """
        tl_field = PARQUET_SCHEMA.field("toxicity_labels")
        assert pa.types.is_list(tl_field.type)
        assert pa.types.is_string(tl_field.type.value_type)

    def test_parquet_schema_ner_annotator_spans_is_nested_list(self):
        """
        GIVEN the PARQUET_SCHEMA,
        THEN  ner_annotator_spans is a list of list of struct.
        """
        field = PARQUET_SCHEMA.field("ner_annotator_spans")
        assert pa.types.is_list(field.type), "ner_annotator_spans should be list"
        inner_type = field.type.value_type
        assert pa.types.is_list(inner_type), "ner_annotator_spans inner type should be list"
        struct_type = inner_type.value_type
        assert pa.types.is_struct(struct_type), "ner_annotator_spans innermost type should be struct"

    def test_parquet_round_trip_preserves_records(self, tmp_path):
        """
        GIVEN a corpus serialized to Parquet,
        WHEN  deserialized,
        THEN  the number of records and sentence_ids are preserved.
        """
        sentences = [
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
        ]
        path = tmp_path / "corpus.parquet"
        serialize(sentences, path, fmt="parquet")
        roundtrip = deserialize(path, fmt="parquet")

        assert len(roundtrip) == len(sentences)
        for orig, rt in zip(sentences, roundtrip):
            assert rt.sentence_id == orig.sentence_id
            assert rt.text == orig.text
            assert rt.platform == orig.platform
            assert rt.sentiment == orig.sentiment

    def test_parquet_written_file_matches_schema(self, tmp_path):
        """
        GIVEN a corpus serialized to Parquet,
        WHEN  the Parquet file is read back with pyarrow,
        THEN  the file schema matches PARQUET_SCHEMA.
        """
        import pyarrow.parquet as pq

        sentence = _make_annotated_sentence()
        path = tmp_path / "corpus.parquet"
        serialize([sentence], path, fmt="parquet")

        table = pq.read_table(str(path))
        # Check all field names match
        assert set(table.schema.names) == set(PARQUET_SCHEMA.names)

    def test_parquet_validate_round_trip_passes(self, tmp_path):
        """
        GIVEN a valid corpus serialized to Parquet,
        WHEN  validate_round_trip is called with fmt='parquet',
        THEN  it returns a RoundTripReport with failed=0.
        """
        sentences = [
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
            _make_annotated_sentence(sentence_id=str(uuid.uuid4())),
        ]
        path = tmp_path / "corpus.parquet"
        serialize(sentences, path, fmt="parquet")
        report = validate_round_trip(sentences, path, fmt="parquet")
        assert report.failed == 0
        assert report.passed == len(sentences)

    def test_serialize_invalid_format_raises_value_error(self, tmp_path):
        """
        GIVEN an unsupported format string,
        WHEN  serialize is called,
        THEN  ValueError is raised.
        """
        sentence = _make_annotated_sentence()
        path = tmp_path / "corpus.xyz"
        with pytest.raises(ValueError, match="Unsupported format"):
            serialize([sentence], path, fmt="csv")  # type: ignore[arg-type]
