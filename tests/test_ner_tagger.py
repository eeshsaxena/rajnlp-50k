"""
Unit tests for the NERTagger.

Tests:
- test_bio_tags_valid_no_i_without_b
- test_bio_tags_single_token_entity_uses_b_tag
- test_bio_tags_multi_token_entity_uses_b_then_i
- test_tag_returns_entity_spans
- test_tag_known_per_entity
- test_tag_known_loc_entity
- test_evaluate_returns_ner_metrics
- test_evaluate_per_type_metrics_keys
- test_train_calls_set_all_seeds
- test_train_returns_training_log

Requirements: 11.2
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from models.data_models import (
    AnnotatedSentence,
    EntitySpan,
    NERMetrics,
    NERPrediction,
    TrainingLog,
)
from models.ner_tagger import (
    NERTagger,
    spans_to_bio_tags,
    validate_bio_tags,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_sentence(
    text: str,
    ner_spans: list[EntitySpan] | None = None,
    sentence_id: str | None = None,
) -> AnnotatedSentence:
    """Create a minimal AnnotatedSentence for testing."""
    return AnnotatedSentence(
        sentence_id=sentence_id or f"id-{random.randint(0, 10**9)}",
        text=text,
        platform="twitter",
        split="train",
        sentiment="neutral",
        sentiment_annotator_labels=["neutral", "neutral", "neutral"],
        ner_spans=ner_spans or [],
        ner_annotator_spans=[[], [], []],
        toxicity_labels=[],
        toxicity_annotator_labels=[[], [], []],
        token_language_labels=[],
        source_url="https://example.com",
        collected_at=_NOW,
        annotated_at=_NOW,
    )


def _make_span(text: str, sentence: str, entity_type: str) -> EntitySpan:
    """Create an EntitySpan by finding *text* in *sentence*."""
    start = sentence.index(text)
    end = start + len(text)
    return EntitySpan(start=start, end=end, entity_type=entity_type, text=text)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BIO tag validity tests
# ---------------------------------------------------------------------------


class TestBIOTagValidity:
    """Verify BIO tag sequences are valid.

    Requirements: 11.2
    """

    def test_bio_tags_valid_no_i_without_b(self):
        """No I- tag should appear without a preceding B- or I- tag of the same type.

        This is the core BIO invariant: I-X must always follow B-X or I-X.
        """
        tagger = NERTagger()

        # Test with sentences that produce various BIO sequences
        test_sentences = [
            "Modi visited Jaipur today",
            "BJP and Congress are parties",
            "Rahul Gandhi went to Delhi",
            "Some sentence with no entities",
            "Gehlot Vasundhara Modi Rahul Sachin Ashok Pilot",
        ]

        for sentence in test_sentences:
            prediction = tagger.tag_with_bio(sentence)
            bio_tags = prediction.bio_tags
            assert validate_bio_tags(bio_tags), (
                f"Invalid BIO sequence {bio_tags} for sentence: '{sentence}'"
            )

    def test_bio_tags_valid_sequence_o_b_i(self):
        """O B-PER I-PER O is a valid BIO sequence."""
        assert validate_bio_tags(["O", "B-PER", "I-PER", "O"]) is True

    def test_bio_tags_invalid_i_without_b(self):
        """I-PER without preceding B-PER is invalid."""
        assert validate_bio_tags(["O", "I-PER", "O"]) is False

    def test_bio_tags_invalid_type_mismatch(self):
        """I-LOC after B-PER is invalid (type mismatch)."""
        assert validate_bio_tags(["B-PER", "I-LOC"]) is False

    def test_bio_tags_valid_multiple_entities(self):
        """Multiple consecutive entities with correct BIO tags are valid."""
        assert validate_bio_tags(["O", "B-PER", "I-PER", "O", "B-LOC", "O"]) is True

    def test_bio_tags_valid_all_o(self):
        """All-O sequence is valid."""
        assert validate_bio_tags(["O", "O", "O"]) is True

    def test_bio_tags_valid_empty(self):
        """Empty sequence is valid."""
        assert validate_bio_tags([]) is True

    def test_bio_tags_valid_single_b(self):
        """Single B- tag (no following I-) is valid."""
        assert validate_bio_tags(["B-PER"]) is True
        assert validate_bio_tags(["B-LOC"]) is True
        assert validate_bio_tags(["B-ORG"]) is True


class TestBIOTagSingleToken:
    """Verify single-token entities use B- tag.

    Requirements: 11.2
    """

    def test_bio_tags_single_token_entity_uses_b_tag(self):
        """A single-token entity should receive a B- tag, not I-."""
        tagger = NERTagger()
        # "Modi" is a single-token PER entity
        prediction = tagger.tag_with_bio("Modi visited India")
        bio_tags = prediction.bio_tags
        tokens = "Modi visited India".split()

        # Find the index of "Modi"
        modi_idx = tokens.index("Modi")
        assert bio_tags[modi_idx].startswith("B-"), (
            f"Single-token entity 'Modi' should have B- tag, got '{bio_tags[modi_idx]}'"
        )
        assert not bio_tags[modi_idx].startswith("I-"), (
            f"Single-token entity 'Modi' should NOT have I- tag, got '{bio_tags[modi_idx]}'"
        )

    def test_bio_tags_single_token_loc_uses_b_tag(self):
        """A single-token LOC entity should receive a B- tag."""
        tagger = NERTagger()
        prediction = tagger.tag_with_bio("He went to Jaipur yesterday")
        bio_tags = prediction.bio_tags
        tokens = "He went to Jaipur yesterday".split()

        jaipur_idx = tokens.index("Jaipur")
        assert bio_tags[jaipur_idx] == "B-LOC", (
            f"Single-token LOC 'Jaipur' should have B-LOC tag, got '{bio_tags[jaipur_idx]}'"
        )

    def test_bio_tags_single_token_org_uses_b_tag(self):
        """A single-token ORG entity should receive a B- tag."""
        tagger = NERTagger()
        prediction = tagger.tag_with_bio("BJP won the election")
        bio_tags = prediction.bio_tags
        tokens = "BJP won the election".split()

        bjp_idx = tokens.index("BJP")
        assert bio_tags[bjp_idx] == "B-ORG", (
            f"Single-token ORG 'BJP' should have B-ORG tag, got '{bio_tags[bjp_idx]}'"
        )


class TestBIOTagMultiToken:
    """Verify multi-token entities use B- then I- tags.

    Requirements: 11.2
    """

    def test_bio_tags_multi_token_entity_uses_b_then_i(self):
        """A multi-token entity should have B- for first token, I- for rest."""
        tagger = NERTagger()
        # "Rahul Gandhi" is a 2-token PER entity
        sentence = "Rahul Gandhi visited Rajasthan"
        prediction = tagger.tag_with_bio(sentence)
        bio_tags = prediction.bio_tags
        tokens = sentence.split()

        rahul_idx = tokens.index("Rahul")
        gandhi_idx = tokens.index("Gandhi")

        assert bio_tags[rahul_idx] == "B-PER", (
            f"First token of multi-token entity should be B-PER, got '{bio_tags[rahul_idx]}'"
        )
        assert bio_tags[gandhi_idx] == "I-PER", (
            f"Second token of multi-token entity should be I-PER, got '{bio_tags[gandhi_idx]}'"
        )

    def test_bio_tags_multi_token_loc_uses_b_then_i(self):
        """Multi-token LOC entity 'New Delhi' should have B-LOC then I-LOC."""
        tagger = NERTagger()
        sentence = "He traveled to New Delhi for the meeting"
        prediction = tagger.tag_with_bio(sentence)
        bio_tags = prediction.bio_tags
        tokens = sentence.split()

        new_idx = tokens.index("New")
        delhi_idx = tokens.index("Delhi")

        assert bio_tags[new_idx] == "B-LOC", (
            f"'New' in 'New Delhi' should be B-LOC, got '{bio_tags[new_idx]}'"
        )
        assert bio_tags[delhi_idx] == "I-LOC", (
            f"'Delhi' in 'New Delhi' should be I-LOC, got '{bio_tags[delhi_idx]}'"
        )

    def test_bio_tags_multi_token_sequence_is_valid(self):
        """Multi-token entity BIO sequence should pass validate_bio_tags."""
        tagger = NERTagger()
        sentence = "Narendra Modi is the Prime Minister"
        prediction = tagger.tag_with_bio(sentence)
        assert validate_bio_tags(prediction.bio_tags), (
            f"BIO sequence {prediction.bio_tags} should be valid"
        )


# ---------------------------------------------------------------------------
# tag() return type tests
# ---------------------------------------------------------------------------


class TestTagReturnsEntitySpans:
    """Verify tag() returns list of EntitySpan objects.

    Requirements: 11.2
    """

    def test_tag_returns_entity_spans(self):
        """tag() should return a list of EntitySpan objects."""
        tagger = NERTagger()
        result = tagger.tag("Modi visited Jaipur")
        assert isinstance(result, list), "tag() should return a list"
        for span in result:
            assert isinstance(span, EntitySpan), (
                f"Each item should be EntitySpan, got {type(span)}"
            )

    def test_tag_empty_sentence_returns_empty_list(self):
        """tag() on an empty sentence should return an empty list."""
        tagger = NERTagger()
        result = tagger.tag("")
        assert result == []

    def test_tag_no_entities_returns_empty_list(self):
        """tag() on a sentence with no known entities should return empty list."""
        tagger = NERTagger()
        result = tagger.tag("the quick brown fox jumps over the lazy dog")
        assert result == []

    def test_tag_span_text_matches_sentence(self):
        """Each EntitySpan's text should match sentence[start:end]."""
        tagger = NERTagger()
        sentence = "Modi visited Jaipur today"
        spans = tagger.tag(sentence)
        for span in spans:
            assert sentence[span.start:span.end] == span.text, (
                f"span.text '{span.text}' != sentence[{span.start}:{span.end}] "
                f"'{sentence[span.start:span.end]}'"
            )

    def test_tag_span_entity_type_is_valid(self):
        """Each EntitySpan's entity_type should be PER, LOC, or ORG."""
        tagger = NERTagger()
        sentence = "Modi visited Jaipur and BJP won"
        spans = tagger.tag(sentence)
        valid_types = {"PER", "LOC", "ORG"}
        for span in spans:
            assert span.entity_type in valid_types, (
                f"entity_type '{span.entity_type}' is not in {valid_types}"
            )


class TestTagKnownPEREntity:
    """Verify known PER names are tagged as PER.

    Requirements: 11.2
    """

    def test_tag_known_per_entity(self):
        """Known PER name 'Modi' should be tagged as PER."""
        tagger = NERTagger()
        spans = tagger.tag("Modi gave a speech today")
        per_spans = [s for s in spans if s.entity_type == "PER"]
        assert len(per_spans) >= 1, "Expected at least one PER span for 'Modi'"
        per_texts = [s.text for s in per_spans]
        assert any("Modi" in t for t in per_texts), (
            f"Expected 'Modi' in PER spans, got {per_texts}"
        )

    def test_tag_gehlot_is_per(self):
        """'Gehlot' should be tagged as PER."""
        tagger = NERTagger()
        spans = tagger.tag("Gehlot announced the policy")
        per_spans = [s for s in spans if s.entity_type == "PER"]
        assert any("Gehlot" in s.text for s in per_spans), (
            "Expected 'Gehlot' to be tagged as PER"
        )

    def test_tag_vasundhara_is_per(self):
        """'Vasundhara' should be tagged as PER."""
        tagger = NERTagger()
        spans = tagger.tag("Vasundhara spoke at the rally")
        per_spans = [s for s in spans if s.entity_type == "PER"]
        assert any("Vasundhara" in s.text for s in per_spans), (
            "Expected 'Vasundhara' to be tagged as PER"
        )

    def test_tag_rahul_is_per(self):
        """'Rahul' should be tagged as PER."""
        tagger = NERTagger()
        spans = tagger.tag("Rahul visited the state")
        per_spans = [s for s in spans if s.entity_type == "PER"]
        assert any("Rahul" in s.text for s in per_spans), (
            "Expected 'Rahul' to be tagged as PER"
        )


class TestTagKnownLOCEntity:
    """Verify known LOC names are tagged as LOC.

    Requirements: 11.2
    """

    def test_tag_known_loc_entity(self):
        """Known LOC name 'Jaipur' should be tagged as LOC."""
        tagger = NERTagger()
        spans = tagger.tag("The event was held in Jaipur")
        loc_spans = [s for s in spans if s.entity_type == "LOC"]
        assert len(loc_spans) >= 1, "Expected at least one LOC span for 'Jaipur'"
        assert any("Jaipur" in s.text for s in loc_spans), (
            f"Expected 'Jaipur' in LOC spans, got {[s.text for s in loc_spans]}"
        )

    def test_tag_rajasthan_is_loc(self):
        """'Rajasthan' should be tagged as LOC."""
        tagger = NERTagger()
        spans = tagger.tag("Rajasthan is a large state")
        loc_spans = [s for s in spans if s.entity_type == "LOC"]
        assert any("Rajasthan" in s.text for s in loc_spans), (
            "Expected 'Rajasthan' to be tagged as LOC"
        )

    def test_tag_delhi_is_loc(self):
        """'Delhi' should be tagged as LOC."""
        tagger = NERTagger()
        spans = tagger.tag("He went to Delhi for the meeting")
        loc_spans = [s for s in spans if s.entity_type == "LOC"]
        assert any("Delhi" in s.text for s in loc_spans), (
            "Expected 'Delhi' to be tagged as LOC"
        )

    def test_tag_india_is_loc(self):
        """'India' should be tagged as LOC."""
        tagger = NERTagger()
        spans = tagger.tag("India is a diverse country")
        loc_spans = [s for s in spans if s.entity_type == "LOC"]
        assert any("India" in s.text for s in loc_spans), (
            "Expected 'India' to be tagged as LOC"
        )


# ---------------------------------------------------------------------------
# evaluate() tests
# ---------------------------------------------------------------------------


class TestEvaluateReturnsNERMetrics:
    """Verify evaluate() returns NERMetrics with correct structure.

    Requirements: 11.2
    """

    def test_evaluate_returns_ner_metrics(self):
        """evaluate() should return a NERMetrics instance."""
        tagger = NERTagger()
        sentence = "Modi visited Jaipur"
        span = _make_span("Jaipur", sentence, "LOC")
        test_set = [_make_sentence(sentence, [span])]

        metrics = tagger.evaluate(test_set)

        assert isinstance(metrics, NERMetrics), (
            f"evaluate() should return NERMetrics, got {type(metrics)}"
        )
        assert isinstance(metrics.macro_f1, float)
        assert 0.0 <= metrics.macro_f1 <= 1.0

    def test_evaluate_per_type_metrics_keys(self):
        """NERMetrics should have PER, LOC, ORG keys in per-type dicts."""
        tagger = NERTagger()
        sentence = "Modi visited Jaipur and BJP won"
        spans = [
            _make_span("Modi", sentence, "PER"),
            _make_span("Jaipur", sentence, "LOC"),
            _make_span("BJP", sentence, "ORG"),
        ]
        test_set = [_make_sentence(sentence, spans)]

        metrics = tagger.evaluate(test_set)

        expected_keys = {"PER", "LOC", "ORG"}
        assert set(metrics.per_type_precision.keys()) == expected_keys, (
            f"per_type_precision keys should be {expected_keys}, "
            f"got {set(metrics.per_type_precision.keys())}"
        )
        assert set(metrics.per_type_recall.keys()) == expected_keys, (
            f"per_type_recall keys should be {expected_keys}, "
            f"got {set(metrics.per_type_recall.keys())}"
        )
        assert set(metrics.per_type_f1.keys()) == expected_keys, (
            f"per_type_f1 keys should be {expected_keys}, "
            f"got {set(metrics.per_type_f1.keys())}"
        )

    def test_evaluate_empty_test_set_returns_zero_metrics(self):
        """evaluate() on empty test set should return zero metrics."""
        tagger = NERTagger()
        metrics = tagger.evaluate([])
        assert metrics.macro_f1 == 0.0
        assert all(v == 0.0 for v in metrics.per_type_f1.values())

    def test_evaluate_perfect_prediction(self):
        """When predicted spans exactly match gold spans, F1 should be 1.0."""
        tagger = NERTagger()
        # Use a sentence where the tagger will correctly predict the gold spans
        sentence = "BJP won the election"
        gold_span = _make_span("BJP", sentence, "ORG")
        test_set = [_make_sentence(sentence, [gold_span])]

        metrics = tagger.evaluate(test_set)

        # BJP is in the ORG lexicon, so prediction should match gold
        assert metrics.per_type_f1.get("ORG", 0.0) == 1.0, (
            f"Expected ORG F1=1.0 for perfect prediction, got {metrics.per_type_f1}"
        )

    def test_evaluate_metrics_in_valid_range(self):
        """All metric values should be in [0.0, 1.0]."""
        tagger = NERTagger()
        sentence = "Modi visited Jaipur"
        spans = [_make_span("Jaipur", sentence, "LOC")]
        test_set = [_make_sentence(sentence, spans)]

        metrics = tagger.evaluate(test_set)

        assert 0.0 <= metrics.macro_f1 <= 1.0
        for etype in ("PER", "LOC", "ORG"):
            assert 0.0 <= metrics.per_type_precision[etype] <= 1.0
            assert 0.0 <= metrics.per_type_recall[etype] <= 1.0
            assert 0.0 <= metrics.per_type_f1[etype] <= 1.0

    def test_evaluate_seqeval_produces_per_type_metrics(self):
        """seqeval evaluation should produce per-type metrics for each entity type."""
        tagger = NERTagger()
        # Build a test set with all three entity types
        sentences_data = [
            ("Modi is a leader", [_make_span("Modi", "Modi is a leader", "PER")]),
            ("Jaipur is beautiful", [_make_span("Jaipur", "Jaipur is beautiful", "LOC")]),
            ("BJP won seats", [_make_span("BJP", "BJP won seats", "ORG")]),
        ]
        test_set = [
            _make_sentence(text, spans)
            for text, spans in sentences_data
        ]

        metrics = tagger.evaluate(test_set)

        # All three entity types should have metrics
        assert "PER" in metrics.per_type_f1
        assert "LOC" in metrics.per_type_f1
        assert "ORG" in metrics.per_type_f1


# ---------------------------------------------------------------------------
# train() tests
# ---------------------------------------------------------------------------


class TestTrainCallsSetAllSeeds:
    """Verify train() calls set_all_seeds(seed).

    Requirements: 17.1
    """

    def test_train_calls_set_all_seeds(self):
        """train() should call set_all_seeds(seed) at the start."""
        tagger = NERTagger()
        train_set = [_make_sentence("Modi visited Jaipur")]
        val_set = [_make_sentence("BJP won the election")]

        with patch("models.ner_tagger.set_all_seeds") as mock_seeds:
            tagger.train(train_set, val_set, seed=42)
            mock_seeds.assert_called_once_with(42)

    def test_train_calls_set_all_seeds_with_correct_seed(self):
        """train() should pass the exact seed value to set_all_seeds."""
        tagger = NERTagger()
        train_set = [_make_sentence("Modi visited Jaipur")]
        val_set = [_make_sentence("BJP won the election")]

        for seed in [0, 1, 42, 99, 12345]:
            with patch("models.ner_tagger.set_all_seeds") as mock_seeds:
                tagger.train(train_set, val_set, seed=seed)
                mock_seeds.assert_called_once_with(seed)


class TestTrainReturnsTrainingLog:
    """Verify train() returns a TrainingLog.

    Requirements: 17.3
    """

    def test_train_returns_training_log(self):
        """train() should return a TrainingLog dataclass."""
        tagger = NERTagger()
        train_set = [_make_sentence("Modi visited Jaipur")]
        val_set = [_make_sentence("BJP won the election")]

        log = tagger.train(train_set, val_set, seed=42)

        assert isinstance(log, TrainingLog), (
            f"train() should return TrainingLog, got {type(log)}"
        )

    def test_train_log_has_correct_seed(self):
        """TrainingLog should record the seed used."""
        tagger = NERTagger()
        train_set = [_make_sentence("Modi visited Jaipur")]
        val_set = [_make_sentence("BJP won the election")]

        log = tagger.train(train_set, val_set, seed=99)

        assert log.seed == 99, f"Expected seed=99, got {log.seed}"

    def test_train_log_has_valid_epochs(self):
        """TrainingLog should have total_epochs_run >= 1."""
        tagger = NERTagger()
        train_set = [_make_sentence("Modi visited Jaipur")]
        val_set = [_make_sentence("BJP won the election")]

        log = tagger.train(train_set, val_set, seed=42, max_epochs=3)

        assert log.total_epochs_run >= 1
        assert log.total_epochs_run <= 3

    def test_train_log_best_f1_non_negative(self):
        """TrainingLog best_f1 should be non-negative."""
        tagger = NERTagger()
        train_set = [_make_sentence("Modi visited Jaipur")]
        val_set = [_make_sentence("BJP won the election")]

        log = tagger.train(train_set, val_set, seed=42)

        assert log.best_f1 >= 0.0

    def test_train_log_best_epoch_valid(self):
        """TrainingLog best_epoch should be a valid epoch index."""
        tagger = NERTagger()
        train_set = [_make_sentence("Modi visited Jaipur")]
        val_set = [_make_sentence("BJP won the election")]

        log = tagger.train(train_set, val_set, seed=42, max_epochs=5)

        assert 0 <= log.best_epoch < 5

    def test_train_log_class_weights_has_entity_types(self):
        """TrainingLog class_weights should contain PER, LOC, ORG keys."""
        tagger = NERTagger()
        train_set = [_make_sentence("Modi visited Jaipur")]
        val_set = [_make_sentence("BJP won the election")]

        log = tagger.train(train_set, val_set, seed=42)

        assert "PER" in log.class_weights
        assert "LOC" in log.class_weights
        assert "ORG" in log.class_weights

    def test_train_empty_sets(self):
        """train() should handle empty train/val sets gracefully."""
        tagger = NERTagger()
        log = tagger.train([], [], seed=42)
        assert isinstance(log, TrainingLog)
        assert log.seed == 42
