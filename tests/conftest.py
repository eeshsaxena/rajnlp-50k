"""
pytest configuration and shared fixtures for the RajNLP-50K test suite.

This file is automatically loaded by pytest before any tests are collected.
It provides:
- Hypothesis settings profile configuration
- Shared fixtures for building test data (RawSentence, AnnotatedSentence, etc.)
- A fixed random seed constant used across all tests
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

import pytest
from hypothesis import HealthCheck, settings

# ---------------------------------------------------------------------------
# Hypothesis settings profiles
# ---------------------------------------------------------------------------

# Default profile: 100 examples per property test (as specified in the design doc)
settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

# CI profile: fewer examples for faster feedback in continuous integration
settings.register_profile(
    "ci",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)

# Thorough profile: more examples for deeper exploration
settings.register_profile(
    "thorough",
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile("default")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_SEED: int = 42
"""Fixed random seed used across all reproducibility tests (Requirement 17.1)."""

FIXED_TIMESTAMP: datetime = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
"""A fixed UTC timestamp used in test fixtures."""

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_seed() -> int:
    """Return the project-wide fixed random seed."""
    return FIXED_SEED


@pytest.fixture
def fixed_timestamp() -> datetime:
    """Return a fixed UTC datetime for use in test fixtures."""
    return FIXED_TIMESTAMP


@pytest.fixture
def sample_raw_sentence():
    """Return a minimal valid RawSentence for use in tests."""
    from models.data_models import RawSentence

    return RawSentence(
        text="म्हारो राजस्थान बहुत सुंदर है यार",
        source_url="https://twitter.com/example/status/123456789",
        collected_at=FIXED_TIMESTAMP,
        platform="twitter",
        sentence_id="550e8400-e29b-41d4-a716-446655440000",
    )


@pytest.fixture
def sample_annotated_sentence():
    """Return a minimal valid AnnotatedSentence for use in tests."""
    from models.data_models import AnnotatedSentence, EntitySpan, TokenLabel

    text = "Gehlot ने जयपुर में बड़ा ऐलान किया"
    return AnnotatedSentence(
        sentence_id="550e8400-e29b-41d4-a716-446655440001",
        text=text,
        platform="twitter",
        split="train",
        sentiment="positive",
        sentiment_annotator_labels=["positive", "positive", "neutral"],
        ner_spans=[
            EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot"),
            EntitySpan(start=10, end=15, entity_type="LOC", text="जयपुर"),
        ],
        ner_annotator_spans=[
            [EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot")],
            [
                EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot"),
                EntitySpan(start=10, end=15, entity_type="LOC", text="जयपुर"),
            ],
            [
                EntitySpan(start=0, end=6, entity_type="PER", text="Gehlot"),
                EntitySpan(start=10, end=15, entity_type="LOC", text="जयपुर"),
            ],
        ],
        toxicity_labels=[],
        toxicity_annotator_labels=[[], [], []],
        token_language_labels=[
            TokenLabel(token="Gehlot", label="RAJ", confidence=0.97),
            TokenLabel(token="ने", label="HIN", confidence=0.99),
            TokenLabel(token="जयपुर", label="RAJ", confidence=0.95),
            TokenLabel(token="में", label="HIN", confidence=0.99),
            TokenLabel(token="बड़ा", label="HIN", confidence=0.92),
            TokenLabel(token="ऐलान", label="HIN", confidence=0.88),
            TokenLabel(token="किया", label="HIN", confidence=0.98),
        ],
        source_url="https://twitter.com/example/status/987654321",
        collected_at=FIXED_TIMESTAMP,
        annotated_at=datetime(2024, 2, 1, 14, 0, 0, tzinfo=timezone.utc),
    )
