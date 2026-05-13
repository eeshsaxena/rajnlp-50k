"""
Integration tests for the RajNLP-50K evaluation pipeline.

Tests:
- test_mbert_zero_shot_produces_valid_classification_metrics
- test_muril_zero_shot_produces_valid_classification_metrics
- test_gpt4o_5shot_produces_valid_classification_metrics
- test_comparison_table_includes_all_required_model_rows
- test_comparison_table_has_all_three_tasks
- test_platform_split_evaluation_returns_results_for_both_directions

Requirements: 13.1, 13.4
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone

import pytest

from models.data_models import (
    AnnotatedSentence,
    ClassificationMetrics,
    DatasetSplit,
    EntitySpan,
    MultiLabelMetrics,
    NERMetrics,
    TokenLabel,
)
from models.ner_tagger import NERTagger
from models.sentiment_classifier import SentimentClassifier
from models.toxicity_classifier import ToxicityClassifier
from evaluation.baselines import (
    BaselineResult,
    GPT4o5ShotEvaluator,
    ZeroShotMBERTEvaluator,
    ZeroShotMuRILEvaluator,
)
from evaluation.comparison_table import (
    REQUIRED_MODEL_NAMES,
    ComparisonRow,
    format_comparison_table,
    generate_comparison_table,
)
from evaluation.platform_split import (
    PlatformSplitResult,
    run_platform_split_evaluation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_sentence(
    text: str = "Gehlot ने जयपुर में बड़ा ऐलान किया",
    sentiment: str = "neutral",
    platform: str = "twitter",
    sentence_id: str | None = None,
) -> AnnotatedSentence:
    """Create a minimal valid AnnotatedSentence for testing."""
    sid = sentence_id or f"id-{random.randint(0, 10**9)}"
    return AnnotatedSentence(
        sentence_id=sid,
        text=text,
        platform=platform,  # type: ignore[arg-type]
        split="test",
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


def _make_test_set(n: int = 10, platform: str = "twitter") -> list[AnnotatedSentence]:
    """Create a small test set with mixed sentiments."""
    sentences = []
    sentiments = ["positive", "neutral", "negative"]
    for i in range(n):
        sentences.append(_make_sentence(
            text=f"test sentence {i}",
            sentiment=sentiments[i % 3],
            platform=platform,
            sentence_id=f"test-{platform}-{i}",
        ))
    return sentences


def _make_dataset_split(
    n_twitter: int = 10,
    n_sharechat: int = 10,
) -> DatasetSplit:
    """Create a DatasetSplit with sentences from both platforms."""
    twitter_sentences = _make_test_set(n_twitter, platform="twitter")
    sharechat_sentences = _make_test_set(n_sharechat, platform="sharechat")

    # Put half in train, rest in test
    all_sentences = twitter_sentences + sharechat_sentences
    mid = len(all_sentences) // 2
    return DatasetSplit(
        train=all_sentences[:mid],
        validation=[],
        test=all_sentences[mid:],
    )


def _make_all_baseline_results() -> list[BaselineResult]:
    """Create a complete set of BaselineResult objects for all required models."""
    results: list[BaselineResult] = []

    # Baseline models
    mbert = ZeroShotMBERTEvaluator()
    muril = ZeroShotMuRILEvaluator()
    gpt4o = GPT4o5ShotEvaluator()
    test_set = _make_test_set(5)

    results.extend(mbert.run_all(test_set))
    results.extend(muril.run_all(test_set))
    results.extend(gpt4o.run_all(test_set))

    # Fine-tuned model stubs (fixed plausible F1 scores)
    results.append(BaselineResult(
        model_name="SentimentClassifier-finetuned",
        task="sentiment",
        macro_f1=0.87,
        metrics=ClassificationMetrics(
            macro_f1=0.87,
            per_class_precision={"positive": 0.87, "neutral": 0.87, "negative": 0.87},
            per_class_recall={"positive": 0.87, "neutral": 0.87, "negative": 0.87},
            per_class_f1={"positive": 0.87, "neutral": 0.87, "negative": 0.87},
        ),
    ))
    results.append(BaselineResult(
        model_name="NERTagger-finetuned",
        task="ner",
        macro_f1=0.83,
        metrics=NERMetrics(
            macro_f1=0.83,
            per_type_precision={"PER": 0.83, "LOC": 0.83, "ORG": 0.83},
            per_type_recall={"PER": 0.83, "LOC": 0.83, "ORG": 0.83},
            per_type_f1={"PER": 0.83, "LOC": 0.83, "ORG": 0.83},
        ),
    ))
    results.append(BaselineResult(
        model_name="ToxicityClassifier-finetuned",
        task="toxicity",
        macro_f1=0.80,
        metrics=MultiLabelMetrics(
            macro_f1=0.80,
            per_category_precision={
                "caste_slur": 0.80, "religious": 0.80,
                "gender": 0.80, "general": 0.80,
            },
            per_category_recall={
                "caste_slur": 0.80, "religious": 0.80,
                "gender": 0.80, "general": 0.80,
            },
            per_category_f1={
                "caste_slur": 0.80, "religious": 0.80,
                "gender": 0.80, "general": 0.80,
            },
        ),
    ))
    return results


# ---------------------------------------------------------------------------
# 18.1 / 18.2 — Baseline evaluator tests
# ---------------------------------------------------------------------------


class TestZeroShotMBERTEvaluator:
    """Verify mBERT zero-shot baseline produces valid metric outputs.

    Requirements: 13.1, 13.2, 13.3
    """

    def test_mbert_zero_shot_produces_valid_classification_metrics(self):
        """evaluate_sentiment() should return ClassificationMetrics with valid macro_f1."""
        evaluator = ZeroShotMBERTEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_sentiment(test_set)

        assert isinstance(metrics, ClassificationMetrics), (
            "evaluate_sentiment() should return ClassificationMetrics"
        )
        assert isinstance(metrics.macro_f1, float), "macro_f1 should be a float"
        assert 0.0 <= metrics.macro_f1 <= 1.0, (
            f"macro_f1={metrics.macro_f1} should be in [0, 1]"
        )
        assert metrics.macro_f1 == pytest.approx(0.45), (
            f"mBERT zero-shot sentiment F1 should be 0.45, got {metrics.macro_f1}"
        )

    def test_mbert_zero_shot_ner_produces_valid_ner_metrics(self):
        """evaluate_ner() should return NERMetrics with valid macro_f1."""
        evaluator = ZeroShotMBERTEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_ner(test_set)

        assert isinstance(metrics, NERMetrics)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.macro_f1 == pytest.approx(0.38)

    def test_mbert_zero_shot_toxicity_produces_valid_multilabel_metrics(self):
        """evaluate_toxicity() should return MultiLabelMetrics with valid macro_f1."""
        evaluator = ZeroShotMBERTEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_toxicity(test_set)

        assert isinstance(metrics, MultiLabelMetrics)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.macro_f1 == pytest.approx(0.32)

    def test_mbert_run_all_returns_three_results(self):
        """run_all() should return exactly 3 BaselineResult objects."""
        evaluator = ZeroShotMBERTEvaluator()
        test_set = _make_test_set(5)
        results = evaluator.run_all(test_set)

        assert len(results) == 3
        tasks = {r.task for r in results}
        assert tasks == {"sentiment", "ner", "toxicity"}
        for r in results:
            assert r.model_name == "mBERT-zero-shot"
            assert 0.0 <= r.macro_f1 <= 1.0


class TestZeroShotMuRILEvaluator:
    """Verify MuRIL zero-shot baseline produces valid metric outputs.

    Requirements: 13.1, 13.2, 13.3
    """

    def test_muril_zero_shot_produces_valid_classification_metrics(self):
        """evaluate_sentiment() should return ClassificationMetrics with valid macro_f1."""
        evaluator = ZeroShotMuRILEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_sentiment(test_set)

        assert isinstance(metrics, ClassificationMetrics)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.macro_f1 == pytest.approx(0.52)

    def test_muril_zero_shot_ner_produces_valid_ner_metrics(self):
        """evaluate_ner() should return NERMetrics with valid macro_f1."""
        evaluator = ZeroShotMuRILEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_ner(test_set)

        assert isinstance(metrics, NERMetrics)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.macro_f1 == pytest.approx(0.44)

    def test_muril_zero_shot_toxicity_produces_valid_multilabel_metrics(self):
        """evaluate_toxicity() should return MultiLabelMetrics with valid macro_f1."""
        evaluator = ZeroShotMuRILEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_toxicity(test_set)

        assert isinstance(metrics, MultiLabelMetrics)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.macro_f1 == pytest.approx(0.38)

    def test_muril_run_all_returns_three_results(self):
        """run_all() should return exactly 3 BaselineResult objects."""
        evaluator = ZeroShotMuRILEvaluator()
        test_set = _make_test_set(5)
        results = evaluator.run_all(test_set)

        assert len(results) == 3
        tasks = {r.task for r in results}
        assert tasks == {"sentiment", "ner", "toxicity"}
        for r in results:
            assert r.model_name == "MuRIL-zero-shot"


class TestGPT4o5ShotEvaluator:
    """Verify GPT-4o 5-shot baseline produces valid metric outputs.

    Requirements: 13.1, 13.2, 13.3
    """

    def test_gpt4o_5shot_produces_valid_classification_metrics(self):
        """evaluate_sentiment() should return ClassificationMetrics with valid macro_f1."""
        evaluator = GPT4o5ShotEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_sentiment(test_set)

        assert isinstance(metrics, ClassificationMetrics)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.macro_f1 == pytest.approx(0.62)

    def test_gpt4o_5shot_ner_produces_valid_ner_metrics(self):
        """evaluate_ner() should return NERMetrics with valid macro_f1."""
        evaluator = GPT4o5ShotEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_ner(test_set)

        assert isinstance(metrics, NERMetrics)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.macro_f1 == pytest.approx(0.58)

    def test_gpt4o_5shot_toxicity_produces_valid_multilabel_metrics(self):
        """evaluate_toxicity() should return MultiLabelMetrics with valid macro_f1."""
        evaluator = GPT4o5ShotEvaluator()
        test_set = _make_test_set(10)
        metrics = evaluator.evaluate_toxicity(test_set)

        assert isinstance(metrics, MultiLabelMetrics)
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.macro_f1 == pytest.approx(0.51)

    def test_gpt4o_run_all_returns_three_results(self):
        """run_all() should return exactly 3 BaselineResult objects."""
        evaluator = GPT4o5ShotEvaluator()
        test_set = _make_test_set(5)
        results = evaluator.run_all(test_set)

        assert len(results) == 3
        tasks = {r.task for r in results}
        assert tasks == {"sentiment", "ner", "toxicity"}
        for r in results:
            assert r.model_name == "GPT-4o-5-shot"

    def test_build_sentiment_prompt_contains_examples(self):
        """build_sentiment_prompt() should include few-shot examples and target sentence."""
        evaluator = GPT4o5ShotEvaluator()
        examples = _make_test_set(5)
        target = "म्हारो राजस्थान"
        prompt = evaluator.build_sentiment_prompt(examples, target)

        assert target in prompt, "Target sentence should appear in the prompt"
        assert "Sentiment:" in prompt, "Prompt should ask for Sentiment"
        # Should contain at least one example
        assert "Example 1:" in prompt

    def test_build_sentiment_prompt_uses_at_most_5_examples(self):
        """build_sentiment_prompt() should use at most 5 few-shot examples."""
        evaluator = GPT4o5ShotEvaluator()
        examples = _make_test_set(10)  # More than 5
        prompt = evaluator.build_sentiment_prompt(examples, "test sentence")

        # Should have Example 5 but not Example 6
        assert "Example 5:" in prompt
        assert "Example 6:" not in prompt

    def test_build_ner_prompt_contains_entities_label(self):
        """build_ner_prompt() should include 'Entities:' in the prompt."""
        evaluator = GPT4o5ShotEvaluator()
        examples = _make_test_set(3)
        prompt = evaluator.build_ner_prompt(examples, "Gehlot ने जयपुर में")

        assert "Entities:" in prompt

    def test_build_toxicity_prompt_contains_categories(self):
        """build_toxicity_prompt() should mention toxicity categories."""
        evaluator = GPT4o5ShotEvaluator()
        examples = _make_test_set(3)
        prompt = evaluator.build_toxicity_prompt(examples, "test sentence")

        assert "caste_slur" in prompt
        assert "Toxicity:" in prompt


# ---------------------------------------------------------------------------
# 18.3 — Comparison table tests
# ---------------------------------------------------------------------------


class TestComparisonTable:
    """Verify comparison table generator.

    Requirements: 13.4
    """

    def test_comparison_table_includes_all_required_model_rows(self):
        """generate_comparison_table() should include all 6 required model rows.

        Requirements: 13.4
        """
        results = _make_all_baseline_results()
        rows = generate_comparison_table(results)

        model_names = {row.model_name for row in rows}
        for required_name in REQUIRED_MODEL_NAMES:
            assert required_name in model_names, (
                f"Required model '{required_name}' missing from comparison table. "
                f"Found: {sorted(model_names)}"
            )

    def test_comparison_table_has_all_three_tasks(self):
        """Each row in the comparison table should have sentiment, NER, and toxicity F1.

        Requirements: 13.4
        """
        results = _make_all_baseline_results()
        rows = generate_comparison_table(results)

        for row in rows:
            assert isinstance(row, ComparisonRow), (
                f"Row for '{row.model_name}' is not a ComparisonRow"
            )
            assert hasattr(row, "sentiment_f1"), (
                f"Row for '{row.model_name}' missing sentiment_f1"
            )
            assert hasattr(row, "ner_f1"), (
                f"Row for '{row.model_name}' missing ner_f1"
            )
            assert hasattr(row, "toxicity_f1"), (
                f"Row for '{row.model_name}' missing toxicity_f1"
            )

    def test_comparison_table_f1_values_are_floats(self):
        """All F1 values in the comparison table should be floats."""
        results = _make_all_baseline_results()
        rows = generate_comparison_table(results)

        for row in rows:
            assert isinstance(row.sentiment_f1, float), (
                f"sentiment_f1 for '{row.model_name}' is not a float"
            )
            assert isinstance(row.ner_f1, float), (
                f"ner_f1 for '{row.model_name}' is not a float"
            )
            assert isinstance(row.toxicity_f1, float), (
                f"toxicity_f1 for '{row.model_name}' is not a float"
            )

    def test_comparison_table_baseline_f1_values_correct(self):
        """Baseline model F1 values should match the stub values."""
        results = _make_all_baseline_results()
        rows = generate_comparison_table(results)

        row_by_name = {row.model_name: row for row in rows}

        mbert_row = row_by_name["mBERT-zero-shot"]
        assert mbert_row.sentiment_f1 == pytest.approx(0.45)
        assert mbert_row.ner_f1 == pytest.approx(0.38)
        assert mbert_row.toxicity_f1 == pytest.approx(0.32)

        muril_row = row_by_name["MuRIL-zero-shot"]
        assert muril_row.sentiment_f1 == pytest.approx(0.52)
        assert muril_row.ner_f1 == pytest.approx(0.44)
        assert muril_row.toxicity_f1 == pytest.approx(0.38)

        gpt4o_row = row_by_name["GPT-4o-5-shot"]
        assert gpt4o_row.sentiment_f1 == pytest.approx(0.62)
        assert gpt4o_row.ner_f1 == pytest.approx(0.58)
        assert gpt4o_row.toxicity_f1 == pytest.approx(0.51)

    def test_comparison_table_empty_results(self):
        """generate_comparison_table() with empty results should still include required rows."""
        rows = generate_comparison_table([])
        model_names = {row.model_name for row in rows}
        for required_name in REQUIRED_MODEL_NAMES:
            assert required_name in model_names

    def test_format_comparison_table_returns_string(self):
        """format_comparison_table() should return a non-empty string."""
        results = _make_all_baseline_results()
        rows = generate_comparison_table(results)
        table_str = format_comparison_table(rows)

        assert isinstance(table_str, str)
        assert len(table_str) > 0

    def test_format_comparison_table_contains_model_names(self):
        """Formatted table should contain all required model names."""
        results = _make_all_baseline_results()
        rows = generate_comparison_table(results)
        table_str = format_comparison_table(rows)

        for name in REQUIRED_MODEL_NAMES:
            assert name in table_str, (
                f"Model name '{name}' not found in formatted table"
            )

    def test_format_comparison_table_empty_rows(self):
        """format_comparison_table() with empty rows should return a placeholder string."""
        table_str = format_comparison_table([])
        assert isinstance(table_str, str)
        assert len(table_str) > 0

    def test_comparison_table_has_exactly_six_required_rows(self):
        """The table should have exactly 6 required model rows."""
        results = _make_all_baseline_results()
        rows = generate_comparison_table(results)

        required_rows = [r for r in rows if r.model_name in REQUIRED_MODEL_NAMES]
        assert len(required_rows) == 6, (
            f"Expected 6 required model rows, got {len(required_rows)}"
        )


# ---------------------------------------------------------------------------
# 18.4 — Platform split evaluation tests
# ---------------------------------------------------------------------------


class TestPlatformSplitEvaluation:
    """Verify platform-split evaluation returns results for both directions.

    Requirements: 13.5
    """

    def test_platform_split_evaluation_returns_results_for_both_directions(self):
        """run_platform_split_evaluation() should return results for both
        twitter→sharechat and sharechat→twitter directions.

        Requirements: 13.5
        """
        dataset = _make_dataset_split(n_twitter=10, n_sharechat=10)
        sentiment_clf = SentimentClassifier()
        ner_tagger = NERTagger()
        toxicity_clf = ToxicityClassifier()

        results = run_platform_split_evaluation(
            dataset=dataset,
            sentiment_clf=sentiment_clf,
            ner_tagger=ner_tagger,
            toxicity_clf=toxicity_clf,
            seed=42,
        )

        assert isinstance(results, list)
        assert len(results) == 6, (
            f"Expected 6 platform-split results (3 tasks × 2 directions), got {len(results)}"
        )

        # Check both directions are present for each task
        for task in ("sentiment", "ner", "toxicity"):
            task_results = [r for r in results if r.task == task]
            assert len(task_results) == 2, (
                f"Expected 2 results for task '{task}', got {len(task_results)}"
            )
            directions = {(r.train_platform, r.eval_platform) for r in task_results}
            assert ("twitter", "sharechat") in directions, (
                f"Missing twitter→sharechat direction for task '{task}'"
            )
            assert ("sharechat", "twitter") in directions, (
                f"Missing sharechat→twitter direction for task '{task}'"
            )

    def test_platform_split_results_have_valid_macro_f1(self):
        """All platform-split results should have macro_f1 that is a float.

        NER macro_f1 may be NaN when the evaluation set contains no gold
        entity spans (seqeval returns NaN for all-O sequences), so we only
        check that the value is a float rather than asserting it is in [0, 1].
        Sentiment and toxicity results should always be in [0, 1].
        """
        dataset = _make_dataset_split(n_twitter=8, n_sharechat=8)
        sentiment_clf = SentimentClassifier()
        ner_tagger = NERTagger()
        toxicity_clf = ToxicityClassifier()

        results = run_platform_split_evaluation(
            dataset=dataset,
            sentiment_clf=sentiment_clf,
            ner_tagger=ner_tagger,
            toxicity_clf=toxicity_clf,
            seed=42,
        )

        for result in results:
            assert isinstance(result, PlatformSplitResult)
            assert isinstance(result.macro_f1, float), (
                f"macro_f1 for {result.task} {result.train_platform}→{result.eval_platform} "
                f"is not a float"
            )
            # Sentiment and toxicity should always be in [0, 1]
            if result.task in ("sentiment", "toxicity"):
                assert 0.0 <= result.macro_f1 <= 1.0, (
                    f"macro_f1={result.macro_f1} out of range for "
                    f"{result.task} {result.train_platform}→{result.eval_platform}"
                )

    def test_platform_split_results_cover_all_tasks(self):
        """Platform-split results should cover all three tasks."""
        dataset = _make_dataset_split(n_twitter=6, n_sharechat=6)
        sentiment_clf = SentimentClassifier()
        ner_tagger = NERTagger()
        toxicity_clf = ToxicityClassifier()

        results = run_platform_split_evaluation(
            dataset=dataset,
            sentiment_clf=sentiment_clf,
            ner_tagger=ner_tagger,
            toxicity_clf=toxicity_clf,
            seed=42,
        )

        tasks_covered = {r.task for r in results}
        assert "sentiment" in tasks_covered
        assert "ner" in tasks_covered
        assert "toxicity" in tasks_covered

    def test_platform_split_result_platforms_are_valid(self):
        """train_platform and eval_platform should be 'twitter' or 'sharechat'."""
        dataset = _make_dataset_split(n_twitter=6, n_sharechat=6)
        sentiment_clf = SentimentClassifier()
        ner_tagger = NERTagger()
        toxicity_clf = ToxicityClassifier()

        results = run_platform_split_evaluation(
            dataset=dataset,
            sentiment_clf=sentiment_clf,
            ner_tagger=ner_tagger,
            toxicity_clf=toxicity_clf,
            seed=42,
        )

        valid_platforms = {"twitter", "sharechat"}
        for result in results:
            assert result.train_platform in valid_platforms, (
                f"Invalid train_platform: {result.train_platform}"
            )
            assert result.eval_platform in valid_platforms, (
                f"Invalid eval_platform: {result.eval_platform}"
            )
            assert result.train_platform != result.eval_platform, (
                f"train_platform and eval_platform should differ, "
                f"got both = '{result.train_platform}'"
            )

