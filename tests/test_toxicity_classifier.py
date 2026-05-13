"""
Unit tests for the ToxicityClassifier.

Tests:
- test_class_weights_inversely_proportional
- test_oversampling_fallback_triggers_when_f1_below_threshold
- test_oversampling_fallback_does_not_trigger_when_f1_above_threshold
- test_predict_returns_valid_labels
- test_predict_returns_toxicity_prediction
- test_evaluate_returns_multi_label_metrics
- test_evaluate_per_category_keys
- test_train_calls_set_all_seeds
- test_train_returns_training_log
- test_train_activates_oversampling_when_needed

Requirements: 12.3
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from models.data_models import (
    AnnotatedSentence,
    MultiLabelMetrics,
    ToxicityPrediction,
    TrainingLog,
)
from models.toxicity_classifier import (
    TOXICITY_CATEGORIES,
    ToxicityClassifier,
    compute_toxicity_class_weights,
    should_oversample,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
_VALID_CATEGORIES = set(TOXICITY_CATEGORIES)


def _make_sentence(
    text: str,
    toxicity_labels: list[str] | None = None,
    sentence_id: str | None = None,
) -> AnnotatedSentence:
    """Create a minimal AnnotatedSentence for testing."""
    if toxicity_labels is None:
        toxicity_labels = []
    return AnnotatedSentence(
        sentence_id=sentence_id or f"id-{random.randint(0, 10**9)}",
        text=text,
        platform="twitter",
        split="train",
        sentiment="neutral",
        sentiment_annotator_labels=["neutral", "neutral", "neutral"],
        ner_spans=[],
        ner_annotator_spans=[[], [], []],
        toxicity_labels=toxicity_labels,  # type: ignore[arg-type]
        toxicity_annotator_labels=[[], [], []],
        token_language_labels=[],
        source_url="https://example.com",
        collected_at=_NOW,
        annotated_at=_NOW,
    )


def _make_dataset_with_distribution(
    n_caste: int = 0,
    n_religious: int = 0,
    n_gender: int = 0,
    n_general: int = 0,
    n_clean: int = 10,
) -> list[AnnotatedSentence]:
    """Create a dataset with the given per-category positive counts."""
    sentences: list[AnnotatedSentence] = []
    for i in range(n_caste):
        sentences.append(_make_sentence(f"caste text {i}", ["caste_slur"], f"caste-{i}"))
    for i in range(n_religious):
        sentences.append(_make_sentence(f"religious text {i}", ["religious"], f"rel-{i}"))
    for i in range(n_gender):
        sentences.append(_make_sentence(f"gender text {i}", ["gender"], f"gen-{i}"))
    for i in range(n_general):
        sentences.append(_make_sentence(f"general text {i}", ["general"], f"gal-{i}"))
    for i in range(n_clean):
        sentences.append(_make_sentence(f"clean text {i}", [], f"clean-{i}"))
    return sentences


def _make_val_metrics(per_category_f1: dict[str, float]) -> MultiLabelMetrics:
    """Create a MultiLabelMetrics with the given per-category F1 scores."""
    macro_f1 = sum(per_category_f1.values()) / len(per_category_f1)
    return MultiLabelMetrics(
        macro_f1=macro_f1,
        per_category_precision={cat: 0.5 for cat in TOXICITY_CATEGORIES},
        per_category_recall={cat: 0.5 for cat in TOXICITY_CATEGORIES},
        per_category_f1=per_category_f1,
    )


# ---------------------------------------------------------------------------
# Tests for compute_toxicity_class_weights
# ---------------------------------------------------------------------------


class TestComputeToxicityClassWeights:
    """Verify class weight computation is inversely proportional to class frequency.

    Requirements: 12.3
    """

    def test_class_weights_inversely_proportional(self):
        """Minority class should have higher weight than majority class."""
        # caste_slur: 2 positives (minority), general: 10 positives (majority)
        sentences = _make_dataset_with_distribution(
            n_caste=2, n_general=10, n_clean=0
        )
        weights = compute_toxicity_class_weights(sentences)

        assert weights["caste_slur"] > weights["general"], (
            f"caste_slur weight {weights['caste_slur']} should be > "
            f"general weight {weights['general']} (caste_slur is minority)"
        )

    def test_class_weights_formula(self):
        """Verify the formula: weight = total / (2 * count_positive)."""
        sentences = _make_dataset_with_distribution(
            n_caste=4, n_religious=2, n_clean=6
        )
        weights = compute_toxicity_class_weights(sentences)

        total = len(sentences)  # 4 + 2 + 6 = 12
        expected_caste = total / (2 * 4)   # 12 / 8 = 1.5
        expected_religious = total / (2 * 2)  # 12 / 4 = 3.0

        assert abs(weights["caste_slur"] - expected_caste) < 1e-9, (
            f"Expected caste_slur weight {expected_caste}, got {weights['caste_slur']}"
        )
        assert abs(weights["religious"] - expected_religious) < 1e-9, (
            f"Expected religious weight {expected_religious}, got {weights['religious']}"
        )

    def test_class_weights_zero_positives_returns_one(self):
        """A category with no positive examples should receive weight 1.0."""
        sentences = _make_dataset_with_distribution(n_caste=5, n_clean=5)
        weights = compute_toxicity_class_weights(sentences)

        # religious, gender, general have 0 positives → weight 1.0
        assert weights["religious"] == 1.0
        assert weights["gender"] == 1.0
        assert weights["general"] == 1.0

    def test_class_weights_empty_sentences(self):
        """Empty sentence list should return default weights of 1.0."""
        weights = compute_toxicity_class_weights([])
        assert all(w == 1.0 for w in weights.values())
        assert set(weights.keys()) == _VALID_CATEGORIES

    def test_class_weights_all_categories_present(self):
        """Weights dict should contain all 4 toxicity categories."""
        sentences = _make_dataset_with_distribution(
            n_caste=3, n_religious=3, n_gender=3, n_general=3, n_clean=3
        )
        weights = compute_toxicity_class_weights(sentences)
        assert set(weights.keys()) == _VALID_CATEGORIES

    def test_class_weights_equal_distribution(self):
        """Equal positive counts across categories should produce equal weights."""
        sentences = _make_dataset_with_distribution(
            n_caste=5, n_religious=5, n_gender=5, n_general=5, n_clean=0
        )
        weights = compute_toxicity_class_weights(sentences)

        # All categories have 5 positives out of 20 total → same weight
        expected = len(sentences) / (2 * 5)
        for cat in TOXICITY_CATEGORIES:
            assert abs(weights[cat] - expected) < 1e-9, (
                f"Expected equal weight {expected} for {cat}, got {weights[cat]}"
            )


# ---------------------------------------------------------------------------
# Tests for should_oversample
# ---------------------------------------------------------------------------


class TestShouldOversample:
    """Verify oversampling fallback logic.

    Requirements: 12.3
    """

    def test_oversampling_fallback_triggers_when_f1_below_threshold(self):
        """should_oversample returns True when any per-category F1 < 0.60."""
        # One category below threshold
        metrics = _make_val_metrics({
            "caste_slur": 0.55,  # below 0.60
            "religious": 0.70,
            "gender": 0.65,
            "general": 0.80,
        })
        assert should_oversample(metrics) is True

    def test_oversampling_fallback_does_not_trigger_when_f1_above_threshold(self):
        """should_oversample returns False when all per-category F1 >= 0.60."""
        metrics = _make_val_metrics({
            "caste_slur": 0.60,  # exactly at threshold
            "religious": 0.70,
            "gender": 0.65,
            "general": 0.80,
        })
        assert should_oversample(metrics) is False

    def test_oversampling_triggers_when_all_below_threshold(self):
        """should_oversample returns True when all categories are below threshold."""
        metrics = _make_val_metrics({
            "caste_slur": 0.10,
            "religious": 0.20,
            "gender": 0.30,
            "general": 0.40,
        })
        assert should_oversample(metrics) is True

    def test_oversampling_does_not_trigger_when_all_above_threshold(self):
        """should_oversample returns False when all categories are well above threshold."""
        metrics = _make_val_metrics({
            "caste_slur": 0.75,
            "religious": 0.80,
            "gender": 0.85,
            "general": 0.90,
        })
        assert should_oversample(metrics) is False

    def test_oversampling_triggers_at_boundary(self):
        """should_oversample returns True when a category is just below 0.60."""
        metrics = _make_val_metrics({
            "caste_slur": 0.599,  # just below threshold
            "religious": 0.70,
            "gender": 0.65,
            "general": 0.80,
        })
        assert should_oversample(metrics) is True

    def test_oversampling_custom_threshold(self):
        """should_oversample respects a custom threshold."""
        metrics = _make_val_metrics({
            "caste_slur": 0.65,
            "religious": 0.70,
            "gender": 0.75,
            "general": 0.80,
        })
        # With threshold=0.70, caste_slur (0.65) is below → True
        assert should_oversample(metrics, threshold=0.70) is True
        # With threshold=0.60, all are above → False
        assert should_oversample(metrics, threshold=0.60) is False


# ---------------------------------------------------------------------------
# Tests for ToxicityClassifier.predict
# ---------------------------------------------------------------------------


class TestToxicityClassifierPredict:
    """Verify predict() returns valid labels and correct types.

    Requirements: 12.2
    """

    def test_predict_returns_valid_labels(self):
        """predict() should return only valid toxicity category labels."""
        clf = ToxicityClassifier()
        for sentence in [
            "chamar bhangi neech",
            "kafir jihad",
            "randi besharmi",
            "gali bakwaas chutiya",
            "clean neutral sentence",
            "",
        ]:
            result = clf.predict(sentence)
            for label in result.labels:
                assert label in _VALID_CATEGORIES, (
                    f"predict('{sentence}') returned invalid label '{label}'"
                )

    def test_predict_returns_toxicity_prediction(self):
        """predict() should return a ToxicityPrediction dataclass."""
        clf = ToxicityClassifier()
        result = clf.predict("some sentence")
        assert isinstance(result, ToxicityPrediction)
        assert isinstance(result.labels, list)
        assert isinstance(result.per_category_scores, dict)

    def test_predict_per_category_scores_keys(self):
        """per_category_scores should contain all 4 toxicity categories."""
        clf = ToxicityClassifier()
        result = clf.predict("any sentence")
        assert set(result.per_category_scores.keys()) == _VALID_CATEGORIES

    def test_predict_scores_in_range(self):
        """All per-category scores should be in [0.0, 1.0]."""
        clf = ToxicityClassifier()
        for sentence in ["chamar kafir randi gali", "clean text", ""]:
            result = clf.predict(sentence)
            for cat, score in result.per_category_scores.items():
                assert 0.0 <= score <= 1.0, (
                    f"Score {score} for category '{cat}' out of range [0, 1]"
                )

    def test_predict_labels_subset_of_scores(self):
        """Predicted labels should be a subset of per_category_scores keys."""
        clf = ToxicityClassifier()
        result = clf.predict("chamar kafir")
        assert set(result.labels).issubset(set(result.per_category_scores.keys()))

    def test_predict_caste_slur_keywords(self):
        """Sentences with caste slur keywords should trigger caste_slur label."""
        clf = ToxicityClassifier()
        # Use multiple keywords to exceed threshold
        result = clf.predict("chamar bhangi neech jaat dalit_slur chamari bhangin neechi")
        assert "caste_slur" in result.labels

    def test_predict_clean_sentence_no_labels(self):
        """A clean sentence with no toxic keywords should return empty labels."""
        clf = ToxicityClassifier()
        result = clf.predict("the weather is nice today")
        assert result.labels == []


# ---------------------------------------------------------------------------
# Tests for ToxicityClassifier.evaluate
# ---------------------------------------------------------------------------


class TestToxicityClassifierEvaluate:
    """Verify evaluate() returns MultiLabelMetrics with correct structure.

    Requirements: 12.6
    """

    def test_evaluate_returns_multi_label_metrics(self):
        """evaluate() should return a MultiLabelMetrics instance."""
        clf = ToxicityClassifier()
        test_set = _make_dataset_with_distribution(
            n_caste=3, n_religious=3, n_gender=3, n_general=3, n_clean=5
        )
        metrics = clf.evaluate(test_set)
        assert isinstance(metrics, MultiLabelMetrics)

    def test_evaluate_per_category_keys(self):
        """MultiLabelMetrics should have all 4 category keys."""
        clf = ToxicityClassifier()
        test_set = _make_dataset_with_distribution(n_clean=5)
        metrics = clf.evaluate(test_set)

        assert set(metrics.per_category_precision.keys()) == _VALID_CATEGORIES
        assert set(metrics.per_category_recall.keys()) == _VALID_CATEGORIES
        assert set(metrics.per_category_f1.keys()) == _VALID_CATEGORIES

    def test_evaluate_macro_f1_in_range(self):
        """macro_f1 should be in [0.0, 1.0]."""
        clf = ToxicityClassifier()
        test_set = _make_dataset_with_distribution(
            n_caste=2, n_religious=2, n_clean=5
        )
        metrics = clf.evaluate(test_set)
        assert 0.0 <= metrics.macro_f1 <= 1.0

    def test_evaluate_empty_test_set(self):
        """evaluate() on an empty test set should return zero metrics."""
        clf = ToxicityClassifier()
        metrics = clf.evaluate([])
        assert metrics.macro_f1 == 0.0
        for cat in TOXICITY_CATEGORIES:
            assert metrics.per_category_f1[cat] == 0.0

    def test_evaluate_per_category_f1_in_range(self):
        """All per-category F1 scores should be in [0.0, 1.0]."""
        clf = ToxicityClassifier()
        test_set = _make_dataset_with_distribution(
            n_caste=3, n_religious=3, n_gender=3, n_general=3, n_clean=5
        )
        metrics = clf.evaluate(test_set)
        for cat, f1 in metrics.per_category_f1.items():
            assert 0.0 <= f1 <= 1.0, f"F1 {f1} for category '{cat}' out of range"


# ---------------------------------------------------------------------------
# Tests for ToxicityClassifier.train
# ---------------------------------------------------------------------------


class TestToxicityClassifierTrain:
    """Verify train() behavior.

    Requirements: 12.3, 17.1, 17.3
    """

    def test_train_calls_set_all_seeds(self):
        """train() should call set_all_seeds(seed) at the start."""
        clf = ToxicityClassifier()
        train_set = _make_dataset_with_distribution(n_caste=3, n_clean=5)
        val_set = _make_dataset_with_distribution(n_caste=2, n_clean=3)

        with patch("models.toxicity_classifier.set_all_seeds") as mock_seeds:
            clf.train(train_set, val_set, seed=42)
            mock_seeds.assert_called_once_with(42)

    def test_train_returns_training_log(self):
        """train() should return a TrainingLog dataclass."""
        clf = ToxicityClassifier()
        train_set = _make_dataset_with_distribution(n_caste=3, n_clean=5)
        val_set = _make_dataset_with_distribution(n_caste=2, n_clean=3)

        log = clf.train(train_set, val_set, seed=42)

        assert isinstance(log, TrainingLog)
        assert log.seed == 42
        assert log.total_epochs_run >= 1
        assert isinstance(log.class_weights, dict)
        assert isinstance(log.best_f1, float)
        assert isinstance(log.best_epoch, int)

    def test_train_class_weights_in_log(self):
        """TrainingLog should contain per-category class weights."""
        clf = ToxicityClassifier()
        train_set = _make_dataset_with_distribution(
            n_caste=2, n_religious=10, n_clean=0
        )
        val_set = _make_dataset_with_distribution(n_clean=3)

        log = clf.train(train_set, val_set, seed=42)

        # caste_slur is minority (2) → higher weight than religious (10)
        assert log.class_weights["caste_slur"] > log.class_weights["religious"], (
            "caste_slur (minority) should have higher weight than religious (majority)"
        )

    def test_train_activates_oversampling_when_needed(self):
        """Oversampling flag should be set when per-category F1 < 0.60 after 5 epochs.

        We use a val set where the heuristic produces low F1 (no keyword matches
        for the gold-labeled categories), triggering the oversampling fallback.
        """
        clf = ToxicityClassifier()
        train_set = _make_dataset_with_distribution(n_caste=3, n_clean=5)

        # Val set: gold labels say "caste_slur" but text has no caste keywords
        # → heuristic predicts no labels → F1 = 0.0 for caste_slur → triggers fallback
        val_set = [
            _make_sentence("clean text without keywords", ["caste_slur"], f"v-{i}")
            for i in range(5)
        ]

        clf.train(train_set, val_set, seed=42, max_epochs=10, oversample_check_epoch=5)

        assert clf._oversampling_active is True, (
            "Oversampling should be activated when per-category F1 < 0.60 after 5 epochs"
        )

    def test_train_does_not_activate_oversampling_when_not_needed(self):
        """Oversampling flag should NOT be set when all per-category F1 >= 0.60.

        We use a val set where all gold labels are empty (non-toxic) and the
        heuristic also predicts no labels → perfect precision/recall for all
        categories (no positives in gold or predictions → F1 = 0.0 by convention,
        but we need a scenario where F1 is high).

        Use a custom classifier with no keywords so it always predicts empty,
        and a val set that is all non-toxic → no false positives, no false negatives
        for any category → F1 = 0.0 (no positives). This is a degenerate case.

        Instead, use a val set where the heuristic correctly predicts all labels.
        """
        # Use a classifier with custom keywords that match the val set text
        custom_keywords = {
            "caste_slur": frozenset({"toxic_caste_word_a", "toxic_caste_word_b",
                                     "toxic_caste_word_c", "toxic_caste_word_d",
                                     "toxic_caste_word_e"}),
            "religious": frozenset({"toxic_rel_word_a", "toxic_rel_word_b",
                                    "toxic_rel_word_c", "toxic_rel_word_d",
                                    "toxic_rel_word_e"}),
            "gender": frozenset({"toxic_gen_word_a", "toxic_gen_word_b",
                                 "toxic_gen_word_c", "toxic_gen_word_d",
                                 "toxic_gen_word_e"}),
            "general": frozenset({"toxic_gal_word_a", "toxic_gal_word_b",
                                  "toxic_gal_word_c", "toxic_gal_word_d",
                                  "toxic_gal_word_e"}),
        }
        clf = ToxicityClassifier(category_keywords=custom_keywords)

        # Val set: all non-toxic, no keywords → heuristic predicts empty → no FP/FN
        # F1 = 0.0 for all categories (no positives) → should_oversample returns True
        # To avoid oversampling, we need F1 >= 0.60 for all categories.
        # Create a val set where gold labels match predictions perfectly.
        val_set = [
            _make_sentence(
                "toxic_caste_word_a toxic_caste_word_b toxic_caste_word_c "
                "toxic_caste_word_d toxic_caste_word_e",
                ["caste_slur"],
                f"v-caste-{i}",
            )
            for i in range(5)
        ] + [
            _make_sentence(
                "toxic_rel_word_a toxic_rel_word_b toxic_rel_word_c "
                "toxic_rel_word_d toxic_rel_word_e",
                ["religious"],
                f"v-rel-{i}",
            )
            for i in range(5)
        ] + [
            _make_sentence(
                "toxic_gen_word_a toxic_gen_word_b toxic_gen_word_c "
                "toxic_gen_word_d toxic_gen_word_e",
                ["gender"],
                f"v-gen-{i}",
            )
            for i in range(5)
        ] + [
            _make_sentence(
                "toxic_gal_word_a toxic_gal_word_b toxic_gal_word_c "
                "toxic_gal_word_d toxic_gal_word_e",
                ["general"],
                f"v-gal-{i}",
            )
            for i in range(5)
        ]

        train_set = _make_dataset_with_distribution(n_caste=3, n_clean=5)
        clf.train(train_set, val_set, seed=42, max_epochs=10, oversample_check_epoch=5)

        assert clf._oversampling_active is False, (
            "Oversampling should NOT be activated when all per-category F1 >= 0.60"
        )

    def test_train_records_seed(self):
        """TrainingLog should record the seed used."""
        clf = ToxicityClassifier()
        train_set = _make_dataset_with_distribution(n_caste=3, n_clean=5)
        val_set = _make_dataset_with_distribution(n_clean=3)

        log = clf.train(train_set, val_set, seed=99)
        assert log.seed == 99

    def test_train_total_epochs_run(self):
        """TrainingLog.total_epochs_run should equal max_epochs."""
        clf = ToxicityClassifier()
        train_set = _make_dataset_with_distribution(n_caste=3, n_clean=5)
        val_set = _make_dataset_with_distribution(n_clean=3)

        log = clf.train(train_set, val_set, seed=42, max_epochs=7)
        assert log.total_epochs_run == 7
