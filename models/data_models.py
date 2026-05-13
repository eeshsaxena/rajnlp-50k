"""
Shared data models for the RajNLP-50K project.

All dataclasses used across corpus_builder, annotator_tool, language_id,
models, evaluation, and release packages are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


# ---------------------------------------------------------------------------
# Primitive type aliases
# ---------------------------------------------------------------------------

ToxicityCategory = Literal["caste_slur", "religious", "gender", "general"]


# ---------------------------------------------------------------------------
# Raw collection models
# ---------------------------------------------------------------------------


@dataclass
class RawSentence:
    """A sentence as collected from a social media platform, before annotation."""

    text: str
    """Original text, unmodified (including emoji and non-Latin characters)."""

    source_url: str
    """URL of the source post or page."""

    collected_at: datetime
    """Collection timestamp in UTC."""

    platform: Literal["twitter", "sharechat"]
    """Source platform identifier."""

    sentence_id: str
    """UUID assigned at collection time."""


# ---------------------------------------------------------------------------
# Annotation sub-models
# ---------------------------------------------------------------------------


@dataclass
class EntitySpan:
    """A named entity span within a sentence.

    Invariant: ``sentence.text[span.start:span.end] == span.text`` must hold
    for every span in an ``AnnotatedSentence``.
    """

    start: int
    """Character offset of the span start (inclusive)."""

    end: int
    """Character offset of the span end (exclusive)."""

    entity_type: Literal["PER", "LOC", "ORG"]
    """Named entity type."""

    text: str
    """Span text (derived from the sentence text; stored for validation)."""


@dataclass
class TokenLabel:
    """A token with its language-ID label assigned by the Language_ID_Tagger."""

    token: str
    """The surface form of the token."""

    label: Literal["RAJ", "HIN", "ENG", "TRL"]
    """Language label: Rajasthani, Hindi, English, or Transliterated."""

    confidence: float
    """Model confidence score in [0.0, 1.0]."""


# ---------------------------------------------------------------------------
# Annotated corpus model
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedSentence:
    """A fully annotated sentence in the RajNLP-50K corpus."""

    sentence_id: str
    """UUID matching the original ``RawSentence.sentence_id``."""

    text: str
    """Original sentence text (unmodified)."""

    platform: Literal["twitter", "sharechat"]
    """Source platform."""

    split: Literal["train", "validation", "test"]
    """Dataset partition this sentence belongs to."""

    # --- Sentiment annotation layer ---
    sentiment: Literal["positive", "neutral", "negative"]
    """Gold sentiment label (majority vote of 3 annotators)."""

    sentiment_annotator_labels: list[str]
    """Raw labels from each of the 3 annotators before majority vote."""

    # --- NER annotation layer ---
    ner_spans: list[EntitySpan]
    """Gold NER spans (majority-vote resolved)."""

    ner_annotator_spans: list[list[EntitySpan]]
    """Raw span sets from each of the 3 annotators."""

    # --- Toxicity annotation layer ---
    toxicity_labels: list[ToxicityCategory]
    """Gold toxicity labels (0–4 categories; empty list = non-toxic)."""

    toxicity_annotator_labels: list[list[ToxicityCategory]]
    """Raw toxicity label sets from each of the 3 annotators."""

    # --- Language ID layer ---
    token_language_labels: list[TokenLabel]
    """Per-token language labels assigned by the Language_ID_Tagger."""

    # --- Metadata ---
    source_url: str
    """URL of the source post or page."""

    collected_at: datetime
    """Original collection timestamp (UTC)."""

    annotated_at: datetime
    """Timestamp when annotation was completed (UTC)."""


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------


@dataclass
class DatasetSplit:
    """The RajNLP-50K corpus partitioned into train / validation / test."""

    train: list[AnnotatedSentence] = field(default_factory=list)
    """Training partition — target size 40,000 sentences."""

    validation: list[AnnotatedSentence] = field(default_factory=list)
    """Validation partition — target size 5,000 sentences."""

    test: list[AnnotatedSentence] = field(default_factory=list)
    """Test partition — target size 5,000 sentences."""


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


@dataclass
class ClassificationMetrics:
    """Evaluation metrics for a single-label classification model."""

    macro_f1: float
    """Macro-averaged F1 score across all classes."""

    per_class_precision: dict[str, float]
    """Precision per class label."""

    per_class_recall: dict[str, float]
    """Recall per class label."""

    per_class_f1: dict[str, float]
    """F1 score per class label."""


@dataclass
class MultiLabelMetrics:
    """Evaluation metrics for a multi-label classification model (e.g., Toxicity_Classifier)."""

    macro_f1: float
    """Macro-averaged F1 score across all toxicity categories."""

    per_category_precision: dict[str, float]
    """Precision per toxicity category (caste_slur, religious, gender, general)."""

    per_category_recall: dict[str, float]
    """Recall per toxicity category."""

    per_category_f1: dict[str, float]
    """F1 score per toxicity category."""


@dataclass
class NERMetrics:
    """Span-level evaluation metrics for the NER_Tagger."""

    macro_f1: float
    """Macro-averaged span-level F1 across all entity types."""

    per_type_precision: dict[str, float]
    """Span-level precision per entity type (PER, LOC, ORG)."""

    per_type_recall: dict[str, float]
    """Span-level recall per entity type."""

    per_type_f1: dict[str, float]
    """Span-level F1 per entity type."""


@dataclass
class LangIDMetrics:
    """Evaluation metrics for the Language_ID_Tagger."""

    token_accuracy: float
    """Overall token-level accuracy across all language classes."""

    per_class_f1: dict[str, float]
    """Per-class F1 score for each language label (RAJ, HIN, ENG, TRL)."""


# ---------------------------------------------------------------------------
# Round-trip validation models
# ---------------------------------------------------------------------------


@dataclass
class RoundTripFailure:
    """Details of a single record that failed round-trip validation."""

    sentence_id: str
    """UUID of the failing record."""

    differing_fields: list[str]
    """Names of fields whose values differ between original and round-tripped record."""

    original_values: dict
    """Original field values (keyed by field name)."""

    roundtrip_values: dict
    """Round-tripped field values (keyed by field name)."""


@dataclass
class RoundTripReport:
    """Summary report from ``CorpusBuilder.validate_round_trip``."""

    total_records: int
    """Total number of records validated."""

    passed: int
    """Number of records that passed round-trip validation."""

    failed: int
    """Number of records that failed round-trip validation."""

    failures: list[RoundTripFailure] = field(default_factory=list)
    """Detailed failure information for each failing record."""
