"""
Corpus_Builder — serialization and round-trip validation for RajNLP-50K.

Provides two public functions:

- ``serialize``: Writes a list of :class:`~models.data_models.AnnotatedSentence`
  objects to disk in either JSON Lines (``"jsonl"``) or Parquet (``"parquet"``)
  format.  All text fields are NFC-normalised before writing.

- ``validate_round_trip``: Deserializes a previously serialized file and
  compares every field of every record against the original
  :class:`~models.data_models.AnnotatedSentence` objects.  On any mismatch the
  method logs the ``sentence_id`` and differing fields, then raises an exception
  to halt the release pipeline.  Returns a :class:`~models.data_models.RoundTripReport`.

Requirements: 15.1, 15.2, 15.3
"""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from models.data_models import (
    AnnotatedSentence,
    EntitySpan,
    RoundTripFailure,
    RoundTripReport,
    TokenLabel,
    ToxicityCategory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyArrow schema definition
# ---------------------------------------------------------------------------

#: PyArrow schema for a single EntitySpan struct.
_ENTITY_SPAN_TYPE = pa.struct(
    [
        pa.field("start", pa.int32(), nullable=False),
        pa.field("end", pa.int32(), nullable=False),
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)

#: PyArrow schema for a single TokenLabel struct.
_TOKEN_LABEL_TYPE = pa.struct(
    [
        pa.field("token", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
    ]
)

#: Full PyArrow schema for an AnnotatedSentence record.
PARQUET_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("sentence_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("platform", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("sentiment", pa.string(), nullable=False),
        pa.field("sentiment_annotator_labels", pa.list_(pa.string()), nullable=False),
        pa.field("ner_spans", pa.list_(_ENTITY_SPAN_TYPE), nullable=False),
        pa.field(
            "ner_annotator_spans",
            pa.list_(pa.list_(_ENTITY_SPAN_TYPE)),
            nullable=False,
        ),
        pa.field("toxicity_labels", pa.list_(pa.string()), nullable=False),
        pa.field(
            "toxicity_annotator_labels",
            pa.list_(pa.list_(pa.string())),
            nullable=False,
        ),
        pa.field("token_language_labels", pa.list_(_TOKEN_LABEL_TYPE), nullable=False),
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("collected_at", pa.string(), nullable=False),
        pa.field("annotated_at", pa.string(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _nfc(text: str) -> str:
    """Return the Unicode NFC normalisation of *text*."""
    return unicodedata.normalize("NFC", text)


def _datetime_to_str(dt: datetime) -> str:
    """Serialise a :class:`datetime` to an ISO-8601 UTC string (``Z`` suffix).

    The datetime is converted to UTC if it carries timezone information, then
    formatted as ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _str_to_datetime(s: str) -> datetime:
    """Parse an ISO-8601 UTC string (``Z`` suffix) back to a UTC-aware :class:`datetime`."""
    # Accept both "Z" suffix and "+00:00" offset
    s_clean = s.rstrip("Z")
    dt = datetime.fromisoformat(s_clean)
    return dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Record → dict conversion (shared by both formats)
# ---------------------------------------------------------------------------


def _entity_span_to_dict(span: EntitySpan) -> dict:
    return {
        "start": span.start,
        "end": span.end,
        "entity_type": span.entity_type,
        "text": _nfc(span.text),
    }


def _token_label_to_dict(tl: TokenLabel) -> dict:
    return {
        "token": _nfc(tl.token),
        "label": tl.label,
        "confidence": tl.confidence,
    }


def _sentence_to_dict(sentence: AnnotatedSentence) -> dict:
    """Convert an :class:`AnnotatedSentence` to a plain dict ready for serialization.

    All text fields are NFC-normalised.  Datetime fields are serialised to
    ISO-8601 UTC strings.
    """
    return {
        "sentence_id": _nfc(sentence.sentence_id),
        "text": _nfc(sentence.text),
        "platform": sentence.platform,
        "split": sentence.split,
        "sentiment": sentence.sentiment,
        "sentiment_annotator_labels": [_nfc(lbl) for lbl in sentence.sentiment_annotator_labels],
        "ner_spans": [_entity_span_to_dict(s) for s in sentence.ner_spans],
        "ner_annotator_spans": [
            [_entity_span_to_dict(s) for s in span_set]
            for span_set in sentence.ner_annotator_spans
        ],
        "toxicity_labels": list(sentence.toxicity_labels),
        "toxicity_annotator_labels": [
            list(label_set) for label_set in sentence.toxicity_annotator_labels
        ],
        "token_language_labels": [_token_label_to_dict(tl) for tl in sentence.token_language_labels],
        "source_url": _nfc(sentence.source_url),
        "collected_at": _datetime_to_str(sentence.collected_at),
        "annotated_at": _datetime_to_str(sentence.annotated_at),
    }


# ---------------------------------------------------------------------------
# dict → AnnotatedSentence reconstruction
# ---------------------------------------------------------------------------


def _dict_to_entity_span(d: dict) -> EntitySpan:
    return EntitySpan(
        start=int(d["start"]),
        end=int(d["end"]),
        entity_type=d["entity_type"],
        text=d["text"],
    )


def _dict_to_token_label(d: dict) -> TokenLabel:
    return TokenLabel(
        token=d["token"],
        label=d["label"],
        confidence=float(d["confidence"]),
    )


def _dict_to_sentence(d: dict) -> AnnotatedSentence:
    """Reconstruct an :class:`AnnotatedSentence` from a plain dict."""
    return AnnotatedSentence(
        sentence_id=d["sentence_id"],
        text=d["text"],
        platform=d["platform"],
        split=d["split"],
        sentiment=d["sentiment"],
        sentiment_annotator_labels=list(d["sentiment_annotator_labels"]),
        ner_spans=[_dict_to_entity_span(s) for s in d["ner_spans"]],
        ner_annotator_spans=[
            [_dict_to_entity_span(s) for s in span_set]
            for span_set in d["ner_annotator_spans"]
        ],
        toxicity_labels=list(d["toxicity_labels"]),
        toxicity_annotator_labels=[
            list(label_set) for label_set in d["toxicity_annotator_labels"]
        ],
        token_language_labels=[_dict_to_token_label(tl) for tl in d["token_language_labels"]],
        source_url=d["source_url"],
        collected_at=_str_to_datetime(d["collected_at"]),
        annotated_at=_str_to_datetime(d["annotated_at"]),
    )


# ---------------------------------------------------------------------------
# JSON Lines serialization
# ---------------------------------------------------------------------------


def _serialize_jsonl(sentences: list[AnnotatedSentence], path: Path) -> None:
    """Write *sentences* to *path* as UTF-8 JSON Lines (one record per line).

    All text fields are NFC-normalised before writing.

    Args:
        sentences: Records to serialize.
        path: Destination file path (created or overwritten).
    """
    with path.open("w", encoding="utf-8") as fh:
        for sentence in sentences:
            record = _sentence_to_dict(sentence)
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    logger.info("serialize (jsonl): wrote %d records to %s", len(sentences), path)


def _deserialize_jsonl(path: Path) -> list[AnnotatedSentence]:
    """Read JSON Lines from *path* and return a list of :class:`AnnotatedSentence`."""
    sentences: list[AnnotatedSentence] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sentences.append(_dict_to_sentence(d))
    logger.info("deserialize (jsonl): read %d records from %s", len(sentences), path)
    return sentences


# ---------------------------------------------------------------------------
# Parquet serialization
# ---------------------------------------------------------------------------


def _sentences_to_arrow_table(sentences: list[AnnotatedSentence]) -> pa.Table:
    """Convert *sentences* to a PyArrow :class:`pa.Table` using :data:`PARQUET_SCHEMA`.

    PyArrow's ``pa.array`` for ``list<struct<...>>`` expects a list of lists of
    dicts (one dict per struct element).  The helper functions below produce
    that shape.
    """
    rows = [_sentence_to_dict(s) for s in sentences]

    # Build column arrays explicitly to match the nested schema.
    # For list<struct> columns, PyArrow expects: list[list[dict]] where each
    # inner dict has keys matching the struct field names.

    sentence_ids = pa.array([r["sentence_id"] for r in rows], type=pa.string())
    texts = pa.array([r["text"] for r in rows], type=pa.string())
    platforms = pa.array([r["platform"] for r in rows], type=pa.string())
    splits = pa.array([r["split"] for r in rows], type=pa.string())
    sentiments = pa.array([r["sentiment"] for r in rows], type=pa.string())
    sentiment_annotator_labels = pa.array(
        [r["sentiment_annotator_labels"] for r in rows],
        type=pa.list_(pa.string()),
    )

    # ner_spans: list<struct<start, end, entity_type, text>>
    # Each row is a list of dicts; each dict is one EntitySpan.
    ner_spans = pa.array(
        [r["ner_spans"] for r in rows],          # list[list[dict]]
        type=pa.list_(_ENTITY_SPAN_TYPE),
    )

    # ner_annotator_spans: list<list<struct<...>>>
    ner_annotator_spans = pa.array(
        [r["ner_annotator_spans"] for r in rows],  # list[list[list[dict]]]
        type=pa.list_(pa.list_(_ENTITY_SPAN_TYPE)),
    )

    toxicity_labels = pa.array(
        [r["toxicity_labels"] for r in rows],
        type=pa.list_(pa.string()),
    )
    toxicity_annotator_labels = pa.array(
        [r["toxicity_annotator_labels"] for r in rows],
        type=pa.list_(pa.list_(pa.string())),
    )

    # token_language_labels: list<struct<token, label, confidence>>
    token_language_labels = pa.array(
        [r["token_language_labels"] for r in rows],  # list[list[dict]]
        type=pa.list_(_TOKEN_LABEL_TYPE),
    )

    source_urls = pa.array([r["source_url"] for r in rows], type=pa.string())
    collected_ats = pa.array([r["collected_at"] for r in rows], type=pa.string())
    annotated_ats = pa.array([r["annotated_at"] for r in rows], type=pa.string())

    return pa.table(
        {
            "sentence_id": sentence_ids,
            "text": texts,
            "platform": platforms,
            "split": splits,
            "sentiment": sentiments,
            "sentiment_annotator_labels": sentiment_annotator_labels,
            "ner_spans": ner_spans,
            "ner_annotator_spans": ner_annotator_spans,
            "toxicity_labels": toxicity_labels,
            "toxicity_annotator_labels": toxicity_annotator_labels,
            "token_language_labels": token_language_labels,
            "source_url": source_urls,
            "collected_at": collected_ats,
            "annotated_at": annotated_ats,
        },
        schema=PARQUET_SCHEMA,
    )


def _serialize_parquet(sentences: list[AnnotatedSentence], path: Path) -> None:
    """Write *sentences* to *path* as a Parquet file using :data:`PARQUET_SCHEMA`.

    Args:
        sentences: Records to serialize.
        path: Destination file path (created or overwritten).
    """
    table = _sentences_to_arrow_table(sentences)
    pq.write_table(table, str(path))
    logger.info("serialize (parquet): wrote %d records to %s", len(sentences), path)


def _deserialize_parquet(path: Path) -> list[AnnotatedSentence]:
    """Read a Parquet file from *path* and return a list of :class:`AnnotatedSentence`."""
    table = pq.read_table(str(path))
    sentences: list[AnnotatedSentence] = []

    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.schema.names}

        # Reconstruct nested structures from PyArrow dicts
        def _row_to_span(s: dict) -> EntitySpan:
            return EntitySpan(
                start=int(s["start"]),
                end=int(s["end"]),
                entity_type=s["entity_type"],
                text=s["text"],
            )

        def _row_to_token_label(tl: dict) -> TokenLabel:
            return TokenLabel(
                token=tl["token"],
                label=tl["label"],
                confidence=float(tl["confidence"]),
            )

        ner_spans = [_row_to_span(s) for s in (row["ner_spans"] or [])]
        ner_annotator_spans = [
            [_row_to_span(s) for s in span_set]
            for span_set in (row["ner_annotator_spans"] or [])
        ]
        token_language_labels = [
            _row_to_token_label(tl) for tl in (row["token_language_labels"] or [])
        ]

        sentences.append(
            AnnotatedSentence(
                sentence_id=row["sentence_id"],
                text=row["text"],
                platform=row["platform"],
                split=row["split"],
                sentiment=row["sentiment"],
                sentiment_annotator_labels=list(row["sentiment_annotator_labels"] or []),
                ner_spans=ner_spans,
                ner_annotator_spans=ner_annotator_spans,
                toxicity_labels=list(row["toxicity_labels"] or []),
                toxicity_annotator_labels=[
                    list(ls) for ls in (row["toxicity_annotator_labels"] or [])
                ],
                token_language_labels=token_language_labels,
                source_url=row["source_url"],
                collected_at=_str_to_datetime(row["collected_at"]),
                annotated_at=_str_to_datetime(row["annotated_at"]),
            )
        )

    logger.info("deserialize (parquet): read %d records from %s", len(sentences), path)
    return sentences


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serialize(
    sentences: list[AnnotatedSentence],
    path: Path,
    fmt: Literal["jsonl", "parquet"],
) -> Path:
    """Serialize *sentences* to *path* in the requested format.

    All text fields are NFC-normalised before writing.  Datetime fields are
    serialised as ISO-8601 UTC strings (``YYYY-MM-DDTHH:MM:SSZ``).

    Args:
        sentences: Records to serialize.
        path: Destination file path.  The file is created or overwritten.
        fmt: Output format — ``"jsonl"`` for JSON Lines or ``"parquet"`` for
            Apache Parquet.

    Returns:
        The resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If *fmt* is not ``"jsonl"`` or ``"parquet"``.
    """
    path = Path(path)
    if fmt == "jsonl":
        _serialize_jsonl(sentences, path)
    elif fmt == "parquet":
        _serialize_parquet(sentences, path)
    else:
        raise ValueError(f"Unsupported format {fmt!r}; expected 'jsonl' or 'parquet'")
    return path


def deserialize(path: Path, fmt: Literal["jsonl", "parquet"]) -> list[AnnotatedSentence]:
    """Deserialize *path* back to a list of :class:`AnnotatedSentence` objects.

    Args:
        path: Source file path.
        fmt: File format — ``"jsonl"`` or ``"parquet"``.

    Returns:
        List of reconstructed :class:`AnnotatedSentence` objects.

    Raises:
        ValueError: If *fmt* is not ``"jsonl"`` or ``"parquet"``.
    """
    path = Path(path)
    if fmt == "jsonl":
        return _deserialize_jsonl(path)
    elif fmt == "parquet":
        return _deserialize_parquet(path)
    else:
        raise ValueError(f"Unsupported format {fmt!r}; expected 'jsonl' or 'parquet'")


# ---------------------------------------------------------------------------
# Field-level comparison helpers
# ---------------------------------------------------------------------------


def _compare_entity_span(a: EntitySpan, b: EntitySpan) -> bool:
    return (
        a.start == b.start
        and a.end == b.end
        and a.entity_type == b.entity_type
        and a.text == b.text
    )


def _compare_token_label(a: TokenLabel, b: TokenLabel) -> bool:
    return a.token == b.token and a.label == b.label and abs(a.confidence - b.confidence) < 1e-9


def _compare_sentences(original: AnnotatedSentence, roundtrip: AnnotatedSentence) -> list[str]:
    """Return a list of field names that differ between *original* and *roundtrip*.

    An empty list means the records are field-for-field identical.
    """
    differing: list[str] = []

    # Scalar fields
    scalar_fields = [
        "sentence_id",
        "text",
        "platform",
        "split",
        "sentiment",
        "source_url",
    ]
    for field_name in scalar_fields:
        if getattr(original, field_name) != getattr(roundtrip, field_name):
            differing.append(field_name)

    # Datetime fields (compare as UTC-aware datetimes)
    for field_name in ("collected_at", "annotated_at"):
        orig_dt: datetime = getattr(original, field_name)
        rt_dt: datetime = getattr(roundtrip, field_name)
        # Normalise both to UTC-aware for comparison
        if orig_dt.tzinfo is None:
            orig_dt = orig_dt.replace(tzinfo=timezone.utc)
        if rt_dt.tzinfo is None:
            rt_dt = rt_dt.replace(tzinfo=timezone.utc)
        orig_dt = orig_dt.astimezone(timezone.utc)
        rt_dt = rt_dt.astimezone(timezone.utc)
        # Compare at second precision (ISO-8601 format has no sub-second resolution)
        if orig_dt.replace(microsecond=0) != rt_dt.replace(microsecond=0):
            differing.append(field_name)

    # List[str] fields
    if original.sentiment_annotator_labels != roundtrip.sentiment_annotator_labels:
        differing.append("sentiment_annotator_labels")
    if list(original.toxicity_labels) != list(roundtrip.toxicity_labels):
        differing.append("toxicity_labels")

    # toxicity_annotator_labels: list[list[str]]
    if len(original.toxicity_annotator_labels) != len(roundtrip.toxicity_annotator_labels):
        differing.append("toxicity_annotator_labels")
    else:
        for i, (orig_set, rt_set) in enumerate(
            zip(original.toxicity_annotator_labels, roundtrip.toxicity_annotator_labels)
        ):
            if list(orig_set) != list(rt_set):
                differing.append("toxicity_annotator_labels")
                break

    # ner_spans: list[EntitySpan]
    if len(original.ner_spans) != len(roundtrip.ner_spans):
        differing.append("ner_spans")
    else:
        for orig_span, rt_span in zip(original.ner_spans, roundtrip.ner_spans):
            if not _compare_entity_span(orig_span, rt_span):
                differing.append("ner_spans")
                break

    # ner_annotator_spans: list[list[EntitySpan]]
    if len(original.ner_annotator_spans) != len(roundtrip.ner_annotator_spans):
        differing.append("ner_annotator_spans")
    else:
        outer_diff = False
        for orig_set, rt_set in zip(original.ner_annotator_spans, roundtrip.ner_annotator_spans):
            if len(orig_set) != len(rt_set):
                outer_diff = True
                break
            for orig_span, rt_span in zip(orig_set, rt_set):
                if not _compare_entity_span(orig_span, rt_span):
                    outer_diff = True
                    break
            if outer_diff:
                break
        if outer_diff:
            differing.append("ner_annotator_spans")

    # token_language_labels: list[TokenLabel]
    if len(original.token_language_labels) != len(roundtrip.token_language_labels):
        differing.append("token_language_labels")
    else:
        for orig_tl, rt_tl in zip(original.token_language_labels, roundtrip.token_language_labels):
            if not _compare_token_label(orig_tl, rt_tl):
                differing.append("token_language_labels")
                break

    return differing


# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------


class RoundTripValidationError(Exception):
    """Raised by :func:`validate_round_trip` when one or more records fail validation.

    Halts the release pipeline as required by Requirement 15.3.
    """


def validate_round_trip(
    originals: list[AnnotatedSentence],
    path: Path,
    fmt: Literal["jsonl", "parquet"],
) -> RoundTripReport:
    """Deserialize *path* and compare every field against *originals*.

    For each record that fails field-for-field comparison:
    - Logs the ``sentence_id`` and the names of differing fields at ERROR level.
    - Records the failure in the returned :class:`~models.data_models.RoundTripReport`.

    After processing all records, if any failures were found, raises
    :class:`RoundTripValidationError` to halt the release pipeline
    (Requirement 15.3).

    Args:
        originals: The original :class:`AnnotatedSentence` objects that were
            serialized.
        path: Path to the serialized file to validate.
        fmt: File format — ``"jsonl"`` or ``"parquet"``.

    Returns:
        A :class:`~models.data_models.RoundTripReport` with counts of passed
        and failed records and detailed failure information.

    Raises:
        RoundTripValidationError: If any record fails round-trip validation.
    """
    roundtrip_sentences = deserialize(path, fmt)

    # Build a lookup by sentence_id for O(1) access
    roundtrip_by_id: dict[str, AnnotatedSentence] = {
        s.sentence_id: s for s in roundtrip_sentences
    }

    failures: list[RoundTripFailure] = []
    passed = 0

    for original in originals:
        sid = original.sentence_id
        if sid not in roundtrip_by_id:
            # Record is missing entirely — treat all fields as differing
            logger.error(
                "validate_round_trip: sentence_id=%s is MISSING from the serialized file",
                sid,
            )
            failures.append(
                RoundTripFailure(
                    sentence_id=sid,
                    differing_fields=["<record missing>"],
                    original_values={"sentence_id": sid},
                    roundtrip_values={},
                )
            )
            continue

        roundtrip = roundtrip_by_id[sid]
        differing = _compare_sentences(original, roundtrip)

        if differing:
            logger.error(
                "validate_round_trip: sentence_id=%s FAILED — differing fields: %s",
                sid,
                differing,
            )
            # Collect original and round-trip values for the differing fields
            original_values: dict = {}
            roundtrip_values: dict = {}
            for field_name in differing:
                original_values[field_name] = getattr(original, field_name, None)
                roundtrip_values[field_name] = getattr(roundtrip, field_name, None)

            failures.append(
                RoundTripFailure(
                    sentence_id=sid,
                    differing_fields=differing,
                    original_values=original_values,
                    roundtrip_values=roundtrip_values,
                )
            )
        else:
            passed += 1

    report = RoundTripReport(
        total_records=len(originals),
        passed=passed,
        failed=len(failures),
        failures=failures,
    )

    logger.info(
        "validate_round_trip: total=%d passed=%d failed=%d",
        report.total_records,
        report.passed,
        report.failed,
    )

    if failures:
        raise RoundTripValidationError(
            f"Round-trip validation failed for {len(failures)} / {len(originals)} records. "
            f"Halting release pipeline. See logs for details."
        )

    return report
