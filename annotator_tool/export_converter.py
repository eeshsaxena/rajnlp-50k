"""
Annotator_Tool — Label Studio export converter.

Converts Label Studio native JSON export format to ``AnnotatedSentence``
objects (the RajNLP-50K schema), and validates that the output matches the
HuggingFace Datasets schema.

Label Studio exports annotations as a list of task objects.  Each task object
contains:
  - ``id``: Label Studio task ID (integer)
  - ``data``: dict with the original task data (e.g., ``text``, ``sentence_id``,
    ``source_url``, ``platform``, ``collected_at``)
  - ``annotations``: list of annotation objects, each with a ``result`` list
    of label/span results from one annotator

This converter handles three export types (one per project):
  - ``sentiment``: Choices annotations → ``sentiment_annotator_labels``
  - ``ner``: Labels (span) annotations → ``ner_annotator_spans``
  - ``toxicity``: Choices annotations → ``toxicity_annotator_labels``

Requirements: 4.5
"""

from __future__ import annotations

import logging
import unicodedata
from datetime import datetime, timezone
from typing import Any, Literal

from models.data_models import AnnotatedSentence, EntitySpan, TokenLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid sentiment labels.
VALID_SENTIMENTS = frozenset({"positive", "neutral", "negative"})

#: Valid toxicity categories (excluding "none" which is a UI-only sentinel).
VALID_TOXICITY_CATEGORIES = frozenset({"caste_slur", "religious", "gender", "general"})

#: Valid NER entity types.
VALID_ENTITY_TYPES = frozenset({"PER", "LOC", "ORG"})

#: Valid platform values.
VALID_PLATFORMS = frozenset({"twitter", "sharechat"})

#: Valid split values.
VALID_SPLITS = frozenset({"train", "validation", "test"})

#: HuggingFace Datasets schema field names for AnnotatedSentence.
HUGGINGFACE_SCHEMA_FIELDS = frozenset({
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
})


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_datetime(value: str | None) -> datetime:
    """Parse an ISO 8601 datetime string to a timezone-aware datetime.

    Args:
        value: ISO 8601 string (e.g., ``"2024-01-15T10:30:00Z"`` or
               ``"2024-01-15T10:30:00+00:00"``).  If ``None`` or empty,
               returns the Unix epoch in UTC.

    Returns:
        A timezone-aware :class:`datetime` in UTC.
    """
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    # Replace trailing 'Z' with '+00:00' for fromisoformat compatibility
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _nfc(text: str) -> str:
    """Return the Unicode NFC normalisation of *text*."""
    return unicodedata.normalize("NFC", text)


def _parse_sentiment_annotation(annotation: dict[str, Any]) -> str | None:
    """Extract the sentiment label from a single Label Studio annotation object.

    Args:
        annotation: A Label Studio annotation dict with a ``result`` list.

    Returns:
        The sentiment label string, or ``None`` if not found.
    """
    for result in annotation.get("result", []):
        if result.get("type") == "choices":
            choices = result.get("value", {}).get("choices", [])
            if choices:
                return choices[0]
    return None


def _parse_ner_annotation(
    annotation: dict[str, Any],
    text: str,
) -> list[EntitySpan]:
    """Extract NER spans from a single Label Studio annotation object.

    Args:
        annotation: A Label Studio annotation dict with a ``result`` list.
        text: The original sentence text (used to derive span text).

    Returns:
        A list of :class:`EntitySpan` objects.
    """
    spans: list[EntitySpan] = []
    for result in annotation.get("result", []):
        if result.get("type") == "labels":
            value = result.get("value", {})
            start = value.get("start", 0)
            end = value.get("end", 0)
            labels = value.get("labels", [])
            if labels and start < end:
                entity_type = labels[0]
                if entity_type in VALID_ENTITY_TYPES:
                    span_text = text[start:end]
                    spans.append(EntitySpan(
                        start=start,
                        end=end,
                        entity_type=entity_type,  # type: ignore[arg-type]
                        text=span_text,
                    ))
    return spans


def _parse_toxicity_annotation(annotation: dict[str, Any]) -> list[str]:
    """Extract toxicity labels from a single Label Studio annotation object.

    The "none" sentinel value (used in the UI to indicate non-toxic content)
    is filtered out — an empty list represents non-toxic content in the schema.

    Args:
        annotation: A Label Studio annotation dict with a ``result`` list.

    Returns:
        A list of toxicity category strings (subset of VALID_TOXICITY_CATEGORIES).
    """
    for result in annotation.get("result", []):
        if result.get("type") == "choices":
            choices = result.get("value", {}).get("choices", [])
            # Filter out the "none" sentinel and invalid categories
            return [c for c in choices if c in VALID_TOXICITY_CATEGORIES]
    return []


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert_sentiment_export(
    tasks: list[dict[str, Any]],
    split: Literal["train", "validation", "test"] = "train",
    annotated_at: datetime | None = None,
) -> list[AnnotatedSentence]:
    """Convert a Label Studio sentiment project export to AnnotatedSentence objects.

    Args:
        tasks: List of Label Studio task dicts from the JSON export.
        split: Dataset split to assign to all converted sentences.
        annotated_at: Annotation completion timestamp.  Defaults to now (UTC).

    Returns:
        A list of :class:`AnnotatedSentence` objects with sentiment annotations
        populated.  NER and toxicity fields are set to empty defaults.
    """
    if annotated_at is None:
        annotated_at = datetime.now(tz=timezone.utc)

    results: list[AnnotatedSentence] = []

    for task in tasks:
        data = task.get("data", {})
        annotations = task.get("annotations", [])

        sentence_id = str(data.get("sentence_id", task.get("id", "")))
        text = _nfc(str(data.get("text", "")))
        platform = data.get("platform", "twitter")
        if platform not in VALID_PLATFORMS:
            logger.warning(
                "Task %s has invalid platform %r; defaulting to 'twitter'",
                sentence_id,
                platform,
            )
            platform = "twitter"

        source_url = str(data.get("source_url", ""))
        collected_at = _parse_datetime(data.get("collected_at"))

        # Extract per-annotator sentiment labels
        sentiment_annotator_labels: list[str] = []
        for ann in annotations:
            label = _parse_sentiment_annotation(ann)
            if label is not None:
                sentiment_annotator_labels.append(label)

        # Compute majority-vote gold label (import here to avoid circular imports)
        from annotator_tool.majority_vote import majority_vote_sentiment
        gold_sentiment = majority_vote_sentiment(sentiment_annotator_labels)
        if gold_sentiment is None:
            gold_sentiment = "neutral"  # fallback for adjudication cases
            logger.warning(
                "No majority sentiment for sentence_id=%s; defaulting to 'neutral'",
                sentence_id,
            )

        sentence = AnnotatedSentence(
            sentence_id=sentence_id,
            text=text,
            platform=platform,  # type: ignore[arg-type]
            split=split,
            sentiment=gold_sentiment,  # type: ignore[arg-type]
            sentiment_annotator_labels=sentiment_annotator_labels,
            ner_spans=[],
            ner_annotator_spans=[],
            toxicity_labels=[],
            toxicity_annotator_labels=[],
            token_language_labels=[],
            source_url=source_url,
            collected_at=collected_at,
            annotated_at=annotated_at,
        )
        results.append(sentence)

    logger.info(
        "convert_sentiment_export: converted %d tasks to AnnotatedSentence objects",
        len(results),
    )
    return results


def convert_ner_export(
    tasks: list[dict[str, Any]],
    split: Literal["train", "validation", "test"] = "train",
    annotated_at: datetime | None = None,
) -> list[AnnotatedSentence]:
    """Convert a Label Studio NER project export to AnnotatedSentence objects.

    Args:
        tasks: List of Label Studio task dicts from the JSON export.
        split: Dataset split to assign to all converted sentences.
        annotated_at: Annotation completion timestamp.  Defaults to now (UTC).

    Returns:
        A list of :class:`AnnotatedSentence` objects with NER annotations
        populated.  Sentiment and toxicity fields are set to empty defaults.
    """
    if annotated_at is None:
        annotated_at = datetime.now(tz=timezone.utc)

    results: list[AnnotatedSentence] = []

    for task in tasks:
        data = task.get("data", {})
        annotations = task.get("annotations", [])

        sentence_id = str(data.get("sentence_id", task.get("id", "")))
        text = _nfc(str(data.get("text", "")))
        platform = data.get("platform", "twitter")
        if platform not in VALID_PLATFORMS:
            platform = "twitter"

        source_url = str(data.get("source_url", ""))
        collected_at = _parse_datetime(data.get("collected_at"))

        # Extract per-annotator NER spans
        ner_annotator_spans: list[list[EntitySpan]] = []
        for ann in annotations:
            spans = _parse_ner_annotation(ann, text)
            ner_annotator_spans.append(spans)

        # Compute majority-vote gold spans
        from annotator_tool.majority_vote import majority_vote_ner_from_entity_spans
        gold_span_tuples = majority_vote_ner_from_entity_spans(ner_annotator_spans)
        gold_spans = [
            EntitySpan(start=s, end=e, entity_type=et, text=text[s:e])  # type: ignore[arg-type]
            for s, e, et in gold_span_tuples
        ]

        sentence = AnnotatedSentence(
            sentence_id=sentence_id,
            text=text,
            platform=platform,  # type: ignore[arg-type]
            split=split,
            sentiment="neutral",  # placeholder — populated from sentiment project
            sentiment_annotator_labels=[],
            ner_spans=gold_spans,
            ner_annotator_spans=ner_annotator_spans,
            toxicity_labels=[],
            toxicity_annotator_labels=[],
            token_language_labels=[],
            source_url=source_url,
            collected_at=collected_at,
            annotated_at=annotated_at,
        )
        results.append(sentence)

    logger.info(
        "convert_ner_export: converted %d tasks to AnnotatedSentence objects",
        len(results),
    )
    return results


def convert_toxicity_export(
    tasks: list[dict[str, Any]],
    split: Literal["train", "validation", "test"] = "train",
    annotated_at: datetime | None = None,
) -> list[AnnotatedSentence]:
    """Convert a Label Studio toxicity project export to AnnotatedSentence objects.

    Args:
        tasks: List of Label Studio task dicts from the JSON export.
        split: Dataset split to assign to all converted sentences.
        annotated_at: Annotation completion timestamp.  Defaults to now (UTC).

    Returns:
        A list of :class:`AnnotatedSentence` objects with toxicity annotations
        populated.  Sentiment and NER fields are set to empty defaults.
    """
    if annotated_at is None:
        annotated_at = datetime.now(tz=timezone.utc)

    results: list[AnnotatedSentence] = []

    for task in tasks:
        data = task.get("data", {})
        annotations = task.get("annotations", [])

        sentence_id = str(data.get("sentence_id", task.get("id", "")))
        text = _nfc(str(data.get("text", "")))
        platform = data.get("platform", "twitter")
        if platform not in VALID_PLATFORMS:
            platform = "twitter"

        source_url = str(data.get("source_url", ""))
        collected_at = _parse_datetime(data.get("collected_at"))

        # Extract per-annotator toxicity labels
        toxicity_annotator_labels: list[list[str]] = []
        for ann in annotations:
            labels = _parse_toxicity_annotation(ann)
            toxicity_annotator_labels.append(labels)

        # Compute majority-vote gold toxicity labels (per-category: include if ≥ 2 annotators marked it)
        from collections import Counter
        category_counts: Counter[str] = Counter()
        for ann_labels in toxicity_annotator_labels:
            for cat in set(ann_labels):
                category_counts[cat] += 1
        gold_toxicity = [
            cat for cat, count in category_counts.items()
            if count >= 2 and cat in VALID_TOXICITY_CATEGORIES
        ]
        gold_toxicity.sort()

        sentence = AnnotatedSentence(
            sentence_id=sentence_id,
            text=text,
            platform=platform,  # type: ignore[arg-type]
            split=split,
            sentiment="neutral",  # placeholder — populated from sentiment project
            sentiment_annotator_labels=[],
            ner_spans=[],
            ner_annotator_spans=[],
            toxicity_labels=gold_toxicity,  # type: ignore[arg-type]
            toxicity_annotator_labels=toxicity_annotator_labels,  # type: ignore[arg-type]
            token_language_labels=[],
            source_url=source_url,
            collected_at=collected_at,
            annotated_at=annotated_at,
        )
        results.append(sentence)

    logger.info(
        "convert_toxicity_export: converted %d tasks to AnnotatedSentence objects",
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_huggingface_schema(sentences: list[AnnotatedSentence]) -> list[str]:
    """Validate that a list of AnnotatedSentence objects matches the HuggingFace schema.

    Checks that all required fields are present and have the correct types.

    Args:
        sentences: List of :class:`AnnotatedSentence` objects to validate.

    Returns:
        A list of validation error messages.  Empty list means all records
        are valid.
    """
    errors: list[str] = []

    for i, sentence in enumerate(sentences):
        prefix = f"Record {i} (sentence_id={sentence.sentence_id!r})"

        # Check required string fields
        if not isinstance(sentence.sentence_id, str):
            errors.append(f"{prefix}: sentence_id must be str")
        if not isinstance(sentence.text, str):
            errors.append(f"{prefix}: text must be str")
        if sentence.platform not in VALID_PLATFORMS:
            errors.append(f"{prefix}: platform must be one of {VALID_PLATFORMS}")
        if sentence.split not in VALID_SPLITS:
            errors.append(f"{prefix}: split must be one of {VALID_SPLITS}")
        if sentence.sentiment not in VALID_SENTIMENTS:
            errors.append(f"{prefix}: sentiment must be one of {VALID_SENTIMENTS}")

        # Check list fields
        if not isinstance(sentence.sentiment_annotator_labels, list):
            errors.append(f"{prefix}: sentiment_annotator_labels must be list")
        if not isinstance(sentence.ner_spans, list):
            errors.append(f"{prefix}: ner_spans must be list")
        if not isinstance(sentence.ner_annotator_spans, list):
            errors.append(f"{prefix}: ner_annotator_spans must be list")
        if not isinstance(sentence.toxicity_labels, list):
            errors.append(f"{prefix}: toxicity_labels must be list")
        if not isinstance(sentence.toxicity_annotator_labels, list):
            errors.append(f"{prefix}: toxicity_annotator_labels must be list")
        if not isinstance(sentence.token_language_labels, list):
            errors.append(f"{prefix}: token_language_labels must be list")

        # Check toxicity labels are valid categories
        for cat in sentence.toxicity_labels:
            if cat not in VALID_TOXICITY_CATEGORIES:
                errors.append(
                    f"{prefix}: toxicity_labels contains invalid category {cat!r}"
                )

        # Check NER spans
        for span in sentence.ner_spans:
            if not isinstance(span, EntitySpan):
                errors.append(f"{prefix}: ner_spans must contain EntitySpan objects")
            elif span.entity_type not in VALID_ENTITY_TYPES:
                errors.append(
                    f"{prefix}: ner_spans contains invalid entity_type {span.entity_type!r}"
                )

        # Check datetime fields
        if not isinstance(sentence.collected_at, datetime):
            errors.append(f"{prefix}: collected_at must be datetime")
        if not isinstance(sentence.annotated_at, datetime):
            errors.append(f"{prefix}: annotated_at must be datetime")

    if errors:
        logger.error(
            "HuggingFace schema validation failed with %d error(s)", len(errors)
        )
    else:
        logger.info(
            "HuggingFace schema validation passed for %d records", len(sentences)
        )

    return errors


def annotated_sentence_to_dict(sentence: AnnotatedSentence) -> dict:
    """Convert an AnnotatedSentence to a plain dict matching the HuggingFace schema.

    Args:
        sentence: The :class:`AnnotatedSentence` to convert.

    Returns:
        A dict with all fields serialized to JSON-compatible types.
    """
    return {
        "sentence_id": sentence.sentence_id,
        "text": sentence.text,
        "platform": sentence.platform,
        "split": sentence.split,
        "sentiment": sentence.sentiment,
        "sentiment_annotator_labels": list(sentence.sentiment_annotator_labels),
        "ner_spans": [
            {
                "start": span.start,
                "end": span.end,
                "entity_type": span.entity_type,
                "text": span.text,
            }
            for span in sentence.ner_spans
        ],
        "ner_annotator_spans": [
            [
                {
                    "start": span.start,
                    "end": span.end,
                    "entity_type": span.entity_type,
                    "text": span.text,
                }
                for span in annotator_spans
            ]
            for annotator_spans in sentence.ner_annotator_spans
        ],
        "toxicity_labels": list(sentence.toxicity_labels),
        "toxicity_annotator_labels": [
            list(ann_labels) for ann_labels in sentence.toxicity_annotator_labels
        ],
        "token_language_labels": [
            {
                "token": tl.token,
                "label": tl.label,
                "confidence": tl.confidence,
            }
            for tl in sentence.token_language_labels
        ],
        "source_url": sentence.source_url,
        "collected_at": sentence.collected_at.isoformat(),
        "annotated_at": sentence.annotated_at.isoformat(),
    }
