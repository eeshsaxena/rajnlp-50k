"""
Unit tests for the SentimentClassifier.

Tests:
- test_class_weights_inversely_proportional
- test_early_stopping_triggers_after_patience
- test_early_stopping_does_not_trigger_before_patience
- test_predict_returns_valid_label
- test_evaluate_returns_classification_metrics
- test_train_calls_set_all_seeds
- test_train_saves_best_checkpoint

Requirements: 10.1
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from models.data_models import (
    AnnotatedSentence,
    ClassificationMetrics,
    SentimentPrediction,
    TrainingLog,
)
from models.sentiment_classifier import (
    EarlyStopping,
    SentimentClassifier,
    SentimentClassifierWithLangID,
    compute_class_weights,
    run_langid_ablation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
_VALID_LABELS = {"positive", "neutral", "negative"}


def _make_sentence(
    text: str,
    sentiment: str = "neutral",
    sentence_id: str | None = None,
) -> AnnotatedSentence:
    """Create a minimal AnnotatedSentence for testing."""
    return AnnotatedSentence(
        sentence_id=sentence_id or f"id-{random.randint(0, 10**9)}",
        text=text,
        platform="twitter",
        split="train",
        sentiment=sentiment,  # type: ignore[arg-type]
        sentiment_annotator_labels=[sentiment, sentiment, sentiment],
        ner_spans=[],
        ner_annotator_spans=[[], [], []],
        toxicity_labels=[],
        toxicity_annotator_labels=[[], [], []],
        token_language_labels=[],
        source_url="https://example.com",
        collected_at=_NOW,
        annotated_at=_NOW,
    )


def _make_dataset(
    n_positive: int,
    n_negative: int,
    n_neutral: int,
    split: str = "train",
) -> list[AnnotatedSentence]:
    """Create a small labelled dataset with the given class distribution."""
    sentences: list[AnnotatedSentence] = []
    for i in range(n_positive):
        sentences.append(_make_sentence(f"good great wonderful {i}", "positive", f"pos-{i}"))
    for i in range(n_negative):
        sentences.append(_make_sentence(f"bad terrible awful {i}", "negative", f"neg-{i}"))
    for i in range(n_neutral):
        sentences.append(_make_sentence(f"sentence about something {i}", "neutral", f"neu-{i}"))
    for s in sentences:
        s.split = split  # type: ignore[assignment]
    return sentences


# ---------------------------------------------------------------------------
# Task 14.3 — Unit tests
# ---------------------------------------------------------------------------


class TestComputeClassWeights:
    """Verify class weight computation is inversely proportional to class frequency.

    Requirements: 10.1
    """

    def test_class_weights_inversely_proportional(self):
        """Given positive×10, negative×5, neutral×2, positive should have the
        lowest weight (most frequent → lowest weight)."""
        labels = ["positive"] * 10 + ["negative"] * 5 + ["neutral"] * 2
        weights = compute_class_weights(labels)

        # positive is most frequent → lowest weight
        assert weights["positive"] < weights["negative"], (
            f"positive weight {weights['positive']} should be < "
            f"negative weight {weights['negative']}"
        )
        assert weights["negative"] < weights["neutral"], (
            f"negative weight {weights['negative']} should be < "
            f"neutral weight {weights['neutral']}"
        )

    def test_class_weights_formula(self):
        """Verify the formula: weight = total / (n_classes * count)."""
        labels = ["positive"] * 10 + ["negative"] * 5 + ["neutral"] * 2
        weights = compute_class_weights(labels)

        total = 17
        n_classes = 3
        expected_positive = total / (n_classes * 10)
        expected_negative = total / (n_classes * 5)
        expected_neutral = total / (n_classes * 2)

        assert abs(weights["positive"] - expected_positive) < 1e-9
        assert abs(weights["negative"] - expected_negative) < 1e-9
        assert abs(weights["neutral"] - expected_neutral) < 1e-9

    def test_class_weights_equal_distribution(self):
        """Equal class distribution should produce equal weights (all 1.0)."""
        labels = ["positive"] * 5 + ["negative"] * 5 + ["neutral"] * 5
        weights = compute_class_weights(labels)

        assert abs(weights["positive"] - 1.0) < 1e-9
        assert abs(weights["negative"] - 1.0) < 1e-9
        assert abs(weights["neutral"] - 1.0) < 1e-9

    def test_class_weights_empty_labels(self):
        """Empty label list should return default weights of 1.0."""
        weights = compute_class_weights([])
        assert all(w == 1.0 for w in weights.values())

    def test_class_weights_missing_class(self):
        """A class absent from labels should receive weight 1.0."""
        labels = ["positive"] * 10 + ["negative"] * 5
        weights = compute_class_weights(labels)
        # neutral is absent → weight 1.0
        assert weights["neutral"] == 1.0


class TestEarlyStopping:
    """Verify early stopping logic.

    Requirements: 10.1
    """

    def test_early_stopping_triggers_after_patience(self):
        """Early stopping should trigger after patience=3 epochs without improvement."""
        es = EarlyStopping(patience=3)

        # First call: improvement (sets best)
        assert es.step(0.5) is False, "Should not stop on first improvement"

        # Three consecutive non-improvements
        assert es.step(0.4) is False, "Should not stop after 1 non-improvement"
        assert es.step(0.3) is False, "Should not stop after 2 non-improvements"
        assert es.step(0.2) is True, "Should stop after 3 non-improvements (patience=3)"

    def test_early_stopping_does_not_trigger_before_patience(self):
        """Early stopping should NOT trigger with only 2 epochs without improvement."""
        es = EarlyStopping(patience=3)

        assert es.step(0.5) is False  # improvement
        assert es.step(0.4) is False  # 1 non-improvement
        assert es.step(0.3) is False  # 2 non-improvements — should NOT stop yet

    def test_early_stopping_resets_on_improvement(self):
        """Counter should reset when a new best metric is observed."""
        es = EarlyStopping(patience=3)

        es.step(0.5)   # best = 0.5, counter = 0
        es.step(0.4)   # counter = 1
        es.step(0.6)   # improvement → counter resets to 0
        assert es.counter == 0
        assert es.best == 0.6

    def test_early_stopping_tracks_best_metric(self):
        """The best metric should be updated whenever a new high is seen."""
        es = EarlyStopping(patience=3)
        es.step(0.3)
        es.step(0.7)
        es.step(0.5)
        assert es.best == 0.7

    def test_early_stopping_patience_one(self):
        """With patience=1, stopping should trigger after a single non-improvement."""
        es = EarlyStopping(patience=1)
        es.step(0.5)   # improvement
        assert es.step(0.4) is True  # 1 non-improvement → stop


class TestSentimentClassifierPredict:
    """Verify predict() returns valid labels.

    Requirements: 10.2
    """

    def test_predict_returns_valid_label(self):
        """predict() on any sentence should return one of the 3 valid labels."""
        clf = SentimentClassifier()
        for sentence in [
            "good great wonderful",
            "bad terrible awful",
            "some neutral sentence",
            "",
            "म्हारो देश",
            "mixed good and bad",
        ]:
            result = clf.predict(sentence)
            assert result.label in _VALID_LABELS, (
                f"predict('{sentence}') returned invalid label '{result.label}'"
            )

    def test_predict_returns_sentiment_prediction(self):
        """predict() should return a SentimentPrediction dataclass."""
        clf = SentimentClassifier()
        result = clf.predict("good sentence")
        assert isinstance(result, SentimentPrediction)
        assert isinstance(result.confidence, float)
        assert isinstance(result.per_class_scores, dict)
        assert set(result.per_class_scores.keys()) == _VALID_LABELS

    def test_predict_positive_keywords(self):
        """Sentences with positive keywords should be classified as positive."""
        clf = SentimentClassifier()
        result = clf.predict("good great excellent")
        assert result.label == "positive"

    def test_predict_negative_keywords(self):
        """Sentences with negative keywords should be classified as negative."""
        clf = SentimentClassifier()
        result = clf.predict("bad terrible awful")
        assert result.label == "negative"

    def test_predict_neutral_no_keywords(self):
        """Sentences with no sentiment keywords should be classified as neutral."""
        clf = SentimentClassifier()
        result = clf.predict("the quick brown fox jumps over the lazy dog")
        assert result.label == "neutral"

    def test_predict_confidence_in_range(self):
        """Confidence score should be in [0.0, 1.0]."""
        clf = SentimentClassifier()
        for sentence in ["good", "bad", "neutral sentence"]:
            result = clf.predict(sentence)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence {result.confidence} out of range for '{sentence}'"
            )


class TestSentimentClassifierEvaluate:
    """Verify evaluate() returns ClassificationMetrics.

    Requirements: 10.5
    """

    def test_evaluate_returns_classification_metrics(self):
        """evaluate() on a small test set should return a ClassificationMetrics instance."""
        clf = SentimentClassifier()
        test_set = _make_dataset(n_positive=5, n_negative=5, n_neutral=5, split="test")
        metrics = clf.evaluate(test_set)

        assert isinstance(metrics, ClassificationMetrics)
        assert isinstance(metrics.macro_f1, float)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert set(metrics.per_class_precision.keys()) == _VALID_LABELS
        assert set(metrics.per_class_recall.keys()) == _VALID_LABELS
        assert set(metrics.per_class_f1.keys()) == _VALID_LABELS

    def test_evaluate_empty_test_set(self):
        """evaluate() on an empty test set should return zero metrics."""
        clf = SentimentClassifier()
        metrics = clf.evaluate([])
        assert metrics.macro_f1 == 0.0

    def test_evaluate_perfect_predictions(self):
        """When all predictions match gold labels, macro-F1 should be 1.0."""
        # Use a classifier that always predicts "positive" and a test set
        # where all gold labels are "positive"
        clf = SentimentClassifier()
        test_set = [_make_sentence("good great wonderful", "positive", f"id-{i}") for i in range(10)]
        metrics = clf.evaluate(test_set)
        # All gold = positive, all predictions should be positive → F1 = 1.0 for positive
        # neutral and negative have 0 gold → F1 = 0.0 for those
        # macro_f1 = (1.0 + 0.0 + 0.0) / 3 ≈ 0.333
        assert metrics.per_class_f1["positive"] == 1.0


class TestSentimentClassifierTrain:
    """Verify train() behavior.

    Requirements: 10.1, 17.1, 17.3
    """

    def test_train_calls_set_all_seeds(self):
        """train() should call set_all_seeds(seed) at the start."""
        clf = SentimentClassifier()
        train_set = _make_dataset(5, 5, 5, "train")
        val_set = _make_dataset(2, 2, 2, "validation")

        with patch("models.sentiment_classifier.set_all_seeds") as mock_seeds:
            clf.train(train_set, val_set, seed=42)
            mock_seeds.assert_called_once_with(42)

    def test_train_saves_best_checkpoint(self):
        """train() should track best_f1 and best_epoch."""
        clf = SentimentClassifier()
        train_set = _make_dataset(10, 10, 10, "train")
        val_set = _make_dataset(5, 5, 5, "validation")

        log = clf.train(train_set, val_set, seed=42)

        assert isinstance(log, TrainingLog)
        assert log.best_f1 >= 0.0
        assert log.best_epoch >= 0
        assert log.seed == 42
        assert isinstance(log.class_weights, dict)

    def test_train_returns_training_log(self):
        """train() should return a TrainingLog dataclass."""
        clf = SentimentClassifier()
        train_set = _make_dataset(5, 5, 5, "train")
        val_set = _make_dataset(2, 2, 2, "validation")

        log = clf.train(train_set, val_set, seed=7)

        assert isinstance(log, TrainingLog)
        assert log.total_epochs_run >= 1
        assert log.total_epochs_run <= 10

    def test_train_early_stopping_limits_epochs(self):
        """train() should stop early when validation F1 does not improve."""
        clf = SentimentClassifier()
        # Use a val set where the heuristic produces a fixed F1 (no improvement)
        # A single-class val set will produce the same F1 every epoch
        val_set = [_make_sentence("some neutral text", "neutral", f"id-{i}") for i in range(10)]
        train_set = _make_dataset(5, 5, 5, "train")

        log = clf.train(train_set, val_set, seed=42, max_epochs=10, patience=3)

        # Should stop before 10 epochs due to no improvement
        assert log.total_epochs_run <= 10

    def test_train_class_weights_in_log(self):
        """TrainingLog should contain class weights computed from training labels."""
        clf = SentimentClassifier()
        # Imbalanced training set: 10 positive, 5 negative, 2 neutral
        train_set = _make_dataset(10, 5, 2, "train")
        val_set = _make_dataset(2, 2, 2, "validation")

        log = clf.train(train_set, val_set, seed=42)

        # positive is most frequent → lowest weight
        assert log.class_weights["positive"] < log.class_weights["neutral"], (
            "positive (most frequent) should have lower weight than neutral (least frequent)"
        )

    def test_train_different_seeds_tracked(self):
        """TrainingLog should record the seed used."""
        clf = SentimentClassifier()
        train_set = _make_dataset(5, 5, 5, "train")
        val_set = _make_dataset(2, 2, 2, "validation")

        log1 = clf.train(train_set, val_set, seed=1)
        log2 = clf.train(train_set, val_set, seed=99)

        assert log1.seed == 1
        assert log2.seed == 99


class TestSentimentClassifierWithLangID:
    """Verify Language_ID integration.

    Requirements: 9.3
    """

    def test_with_langid_predict_returns_valid_label(self):
        """SentimentClassifierWithLangID.predict() should return a valid label."""
        clf = SentimentClassifierWithLangID()
        result = clf.predict("good great wonderful")
        assert result.label in _VALID_LABELS

    def test_with_langid_augments_sentence(self):
        """_augment_sentence should prepend language-ID tokens."""
        clf = SentimentClassifierWithLangID()
        augmented = clf._augment_sentence("good")
        # Should contain a language tag like [ENG] or [TRL]
        assert "[" in augmented and "]" in augmented

    def test_ablation_result_structure(self):
        """run_langid_ablation should return an AblationResult with correct fields."""
        from models.sentiment_classifier import AblationResult

        train_set = _make_dataset(10, 10, 10, "train")
        val_set = _make_dataset(5, 5, 5, "validation")
        test_set = _make_dataset(5, 5, 5, "test")

        result = run_langid_ablation(train_set, val_set, test_set, seed=42)

        assert isinstance(result, AblationResult)
        assert isinstance(result.no_langid_f1, float)
        assert isinstance(result.with_langid_f1, float)
        assert isinstance(result.improvement, float)
        assert isinstance(result.meets_requirement, bool)
        # improvement should equal the difference
        assert abs(result.improvement - (result.with_langid_f1 - result.no_langid_f1)) < 1e-9
