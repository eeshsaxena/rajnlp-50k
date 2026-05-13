"""
Unit tests for the HuggingFace release pipeline.

Tests:
- test_dataset_card_includes_source_platforms
- test_dataset_card_includes_compensation_rates
- test_dataset_card_includes_content_warning_procedures
- test_dataset_card_includes_known_demographic_biases
- test_dataset_card_includes_annotator_demographics
- test_model_card_includes_random_seed
- test_model_card_includes_hardware_config
- test_model_card_includes_training_duration
- test_upload_retry_fires_on_failure
- test_upload_retry_succeeds_after_transient_failure
- test_model_not_published_below_target_f1
- test_model_published_above_target_f1

Requirements: 14.4, 14.5, 17.4
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from models.data_models import DatasetSplit
from release.huggingface_publisher import (
    TARGET_F1,
    DatasetCard,
    HuggingFacePublisher,
    ModelCard,
    _upload_with_retry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def publisher() -> HuggingFacePublisher:
    """Return a publisher configured for successful uploads."""
    return HuggingFacePublisher()


@pytest.fixture
def failing_publisher() -> HuggingFacePublisher:
    """Return a publisher configured to always fail uploads."""
    return HuggingFacePublisher(simulate_upload_failure=True)


@pytest.fixture
def transient_publisher() -> HuggingFacePublisher:
    """Return a publisher that fails once then succeeds."""
    return HuggingFacePublisher(failure_count=1)


@pytest.fixture
def empty_dataset_split() -> DatasetSplit:
    """Return an empty DatasetSplit for testing."""
    return DatasetSplit(train=[], validation=[], test=[])


@pytest.fixture
def sample_model_card(publisher: HuggingFacePublisher) -> ModelCard:
    """Return a sample ModelCard for SentimentClassifier."""
    return publisher.generate_model_card(
        model_name="SentimentClassifier",
        repo_id="org/rajnlp-sentiment",
        random_seed=42,
        hardware_config="1× NVIDIA A100 80GB",
        training_duration="4h 32m",
        evaluation_metrics={"macro_f1": 0.87, "precision": 0.86, "recall": 0.88},
    )


# ---------------------------------------------------------------------------
# Dataset card content tests (Requirements 14.4, 14.5)
# ---------------------------------------------------------------------------


class TestDatasetCardSourcePlatforms:
    """Verify dataset card includes source platform information.

    Requirements: 14.4, 14.5
    """

    def test_dataset_card_includes_source_platforms(self, publisher: HuggingFacePublisher) -> None:
        """Dataset card must mention both Twitter/X and ShareChat as source platforms."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")

        assert isinstance(card, DatasetCard)
        assert "Twitter" in card.content, (
            "Dataset card must mention 'Twitter' as a source platform"
        )
        assert "ShareChat" in card.content, (
            "Dataset card must mention 'ShareChat' as a source platform"
        )

    def test_dataset_card_mentions_twitter_x(self, publisher: HuggingFacePublisher) -> None:
        """Dataset card must explicitly reference Twitter/X."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        # Accept either "Twitter/X" or "Twitter" as valid
        assert "Twitter" in card.content

    def test_dataset_card_mentions_sharechat(self, publisher: HuggingFacePublisher) -> None:
        """Dataset card must explicitly reference ShareChat."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        assert "ShareChat" in card.content


class TestDatasetCardCompensationRates:
    """Verify dataset card includes annotator compensation information.

    Requirements: 14.5
    """

    def test_dataset_card_includes_compensation_rates(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must mention annotator compensation rates."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")

        # The card should mention compensation in some form
        content_lower = card.content.lower()
        assert any(
            keyword in content_lower
            for keyword in ("compensat", "payment", "paid", "rate", "salary", "wage")
        ), (
            "Dataset card must include annotator compensation rate information"
        )

    def test_dataset_card_compensation_section_present(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must have a compensation section or mention."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        assert "Compensation" in card.content or "compensation" in card.content


class TestDatasetCardContentWarning:
    """Verify dataset card includes content warning procedures.

    Requirements: 14.5
    """

    def test_dataset_card_includes_content_warning_procedures(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must describe content warning procedures."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")

        content_lower = card.content.lower()
        assert any(
            keyword in content_lower
            for keyword in ("content warning", "warning", "opt-out", "opt out", "toxic")
        ), (
            "Dataset card must include content warning procedure information"
        )

    def test_dataset_card_content_warning_mentions_opt_out(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card content warning section should mention opt-out mechanism."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        content_lower = card.content.lower()
        assert "opt-out" in content_lower or "opt out" in content_lower or "withdraw" in content_lower


class TestDatasetCardDemographicBiases:
    """Verify dataset card includes known demographic biases.

    Requirements: 14.5
    """

    def test_dataset_card_includes_known_demographic_biases(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must describe known demographic biases."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")

        content_lower = card.content.lower()
        assert any(
            keyword in content_lower
            for keyword in ("bias", "demographic", "under-represent", "skew")
        ), (
            "Dataset card must include known demographic bias information"
        )

    def test_dataset_card_demographic_biases_section_present(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must have a demographic biases section."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        assert "Demographic" in card.content or "demographic" in card.content or "bias" in card.content.lower()


class TestDatasetCardAnnotatorDemographics:
    """Verify dataset card includes annotator demographics.

    Requirements: 14.4
    """

    def test_dataset_card_includes_annotator_demographics(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must describe annotator demographics."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")

        content_lower = card.content.lower()
        assert any(
            keyword in content_lower
            for keyword in ("annotator", "demographics", "native", "speaker")
        ), (
            "Dataset card must include annotator demographics information"
        )

    def test_dataset_card_annotator_section_present(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must have an annotator demographics section."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        assert "Annotator" in card.content or "annotator" in card.content

    def test_dataset_card_collection_period_present(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must include the collection period."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        assert "Collection Period" in card.content or "collection period" in card.content.lower() or "2024" in card.content


# ---------------------------------------------------------------------------
# Model card content tests (Requirement 17.4)
# ---------------------------------------------------------------------------


class TestModelCardRandomSeed:
    """Verify model cards include Random_Seed.

    Requirements: 17.4
    """

    def test_model_card_includes_random_seed(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Model card must mention the Random_Seed used for training."""
        card = publisher.generate_model_card(
            model_name="SentimentClassifier",
            repo_id="org/rajnlp-sentiment",
            random_seed=42,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration="4h 32m",
            evaluation_metrics={"macro_f1": 0.87},
        )

        content_lower = card.content.lower()
        assert "random_seed" in content_lower or "random seed" in content_lower, (
            "Model card must include 'Random_Seed' or 'random seed'"
        )
        # The actual seed value should appear in the card
        assert "42" in card.content, (
            "Model card must include the actual seed value"
        )

    def test_model_card_seed_value_matches(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """The seed value in the model card must match the provided seed."""
        for seed in (0, 42, 12345, 999):
            card = publisher.generate_model_card(
                model_name="NERTagger",
                repo_id="org/rajnlp-ner",
                random_seed=seed,
                hardware_config="2× NVIDIA V100 32GB",
                training_duration="6h 15m",
                evaluation_metrics={"macro_f1": 0.84},
            )
            assert str(seed) in card.content, (
                f"Model card must contain seed value {seed}"
            )

    def test_model_card_random_seed_field_populated(
        self, sample_model_card: ModelCard
    ) -> None:
        """ModelCard dataclass must have random_seed field set correctly."""
        assert sample_model_card.random_seed == 42


class TestModelCardHardwareConfig:
    """Verify model cards include hardware configuration.

    Requirements: 17.4
    """

    def test_model_card_includes_hardware_config(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Model card must mention hardware configuration (GPU type and count)."""
        card = publisher.generate_model_card(
            model_name="ToxicityClassifier",
            repo_id="org/rajnlp-toxicity",
            random_seed=7,
            hardware_config="4× NVIDIA A100 80GB",
            training_duration="8h 00m",
            evaluation_metrics={"macro_f1": 0.81},
        )

        content_lower = card.content.lower()
        assert any(
            keyword in content_lower
            for keyword in ("hardware", "gpu", "nvidia", "a100", "v100", "cuda")
        ), (
            "Model card must include hardware configuration information"
        )
        # The specific hardware config string should appear
        assert "4× NVIDIA A100 80GB" in card.content or "NVIDIA" in card.content

    def test_model_card_hardware_config_field_populated(
        self, sample_model_card: ModelCard
    ) -> None:
        """ModelCard dataclass must have hardware_config field set correctly."""
        assert "NVIDIA" in sample_model_card.hardware_config or "GPU" in sample_model_card.hardware_config.upper()

    def test_model_card_hardware_config_in_content(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """The hardware config string must appear verbatim in the card content."""
        hw = "2× NVIDIA V100 32GB"
        card = publisher.generate_model_card(
            model_name="NERTagger",
            repo_id="org/rajnlp-ner",
            random_seed=99,
            hardware_config=hw,
            training_duration="3h 45m",
            evaluation_metrics={"macro_f1": 0.83},
        )
        assert hw in card.content, (
            f"Hardware config '{hw}' must appear in model card content"
        )


class TestModelCardTrainingDuration:
    """Verify model cards include training duration.

    Requirements: 17.4
    """

    def test_model_card_includes_training_duration(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Model card must mention training duration."""
        card = publisher.generate_model_card(
            model_name="SentimentClassifier",
            repo_id="org/rajnlp-sentiment",
            random_seed=42,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration="4h 32m",
            evaluation_metrics={"macro_f1": 0.87},
        )

        content_lower = card.content.lower()
        assert any(
            keyword in content_lower
            for keyword in ("duration", "training duration", "time", "hours", "minutes")
        ), (
            "Model card must include training duration information"
        )
        assert "4h 32m" in card.content, (
            "Model card must include the actual training duration value"
        )

    def test_model_card_training_duration_field_populated(
        self, sample_model_card: ModelCard
    ) -> None:
        """ModelCard dataclass must have training_duration field set correctly."""
        assert sample_model_card.training_duration == "4h 32m"

    def test_model_card_training_duration_in_content(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """The training duration string must appear verbatim in the card content."""
        duration = "12h 05m"
        card = publisher.generate_model_card(
            model_name="ToxicityClassifier",
            repo_id="org/rajnlp-toxicity",
            random_seed=1,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration=duration,
            evaluation_metrics={"macro_f1": 0.80},
        )
        assert duration in card.content, (
            f"Training duration '{duration}' must appear in model card content"
        )


# ---------------------------------------------------------------------------
# Retry logic tests (Requirements 14.4, 17.4)
# ---------------------------------------------------------------------------


class TestUploadRetryLogic:
    """Verify upload retry logic fires on simulated upload failure.

    Requirements: 14.4, 17.4
    """

    def test_upload_retry_fires_on_failure(self) -> None:
        """Retry logic must call the upload function multiple times on failure."""
        call_count = 0

        def always_failing_upload() -> None:
            nonlocal call_count
            call_count += 1
            raise IOError("Simulated failure")

        with pytest.raises(IOError):
            _upload_with_retry(always_failing_upload, max_attempts=3, base_delay=0.0)

        assert call_count == 3, (
            f"Expected 3 upload attempts, got {call_count}"
        )

    def test_upload_retry_exhausts_all_attempts(self) -> None:
        """Retry logic must exhaust all max_attempts before raising."""
        attempts = []

        def failing_upload() -> None:
            attempts.append(1)
            raise IOError("Always fails")

        with pytest.raises(IOError):
            _upload_with_retry(failing_upload, max_attempts=5, base_delay=0.0)

        assert len(attempts) == 5

    def test_upload_retry_raises_last_exception(self) -> None:
        """Retry logic must re-raise the last exception after exhausting attempts."""
        def failing_upload() -> None:
            raise ValueError("Specific error message")

        with pytest.raises(ValueError, match="Specific error message"):
            _upload_with_retry(failing_upload, max_attempts=2, base_delay=0.0)

    def test_upload_retry_succeeds_after_transient_failure(self) -> None:
        """Retry logic must succeed if the upload succeeds on a later attempt."""
        call_count = 0

        def transient_upload() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IOError("Transient failure")
            # Succeeds on attempt 2

        # Should not raise
        _upload_with_retry(transient_upload, max_attempts=3, base_delay=0.0)
        assert call_count == 2, (
            f"Expected 2 upload attempts (1 failure + 1 success), got {call_count}"
        )

    def test_upload_retry_succeeds_on_first_attempt(self) -> None:
        """Retry logic must succeed immediately if the first attempt succeeds."""
        call_count = 0

        def successful_upload() -> None:
            nonlocal call_count
            call_count += 1

        _upload_with_retry(successful_upload, max_attempts=3, base_delay=0.0)
        assert call_count == 1, (
            f"Expected 1 upload attempt, got {call_count}"
        )

    def test_publisher_retry_fires_on_dataset_upload_failure(
        self, failing_publisher: HuggingFacePublisher, empty_dataset_split: DatasetSplit
    ) -> None:
        """Publisher must retry dataset upload on failure."""
        result = failing_publisher.publish_dataset(
            empty_dataset_split,
            "org/rajnlp-50k",
            max_attempts=3,
            base_delay=0.0,
        )
        assert result.success is False
        assert result.attempts == 3, (
            f"Expected 3 upload attempts, got {result.attempts}"
        )

    def test_publisher_retry_fires_on_model_upload_failure(
        self,
        failing_publisher: HuggingFacePublisher,
        sample_model_card: ModelCard,
    ) -> None:
        """Publisher must retry model upload on failure."""
        # Reset call count for the failing publisher
        failing_publisher._call_count = 0
        result = failing_publisher.publish_model(
            sample_model_card,
            target_f1=0.85,
            actual_f1=0.90,
            max_attempts=3,
            base_delay=0.0,
        )
        assert result is False
        assert failing_publisher._call_count == 3, (
            f"Expected 3 upload attempts, got {failing_publisher._call_count}"
        )

    def test_publisher_succeeds_after_transient_failure(
        self,
        transient_publisher: HuggingFacePublisher,
        empty_dataset_split: DatasetSplit,
    ) -> None:
        """Publisher must succeed after a single transient failure."""
        result = transient_publisher.publish_dataset(
            empty_dataset_split,
            "org/rajnlp-50k",
            max_attempts=3,
            base_delay=0.0,
        )
        assert result.success is True
        assert result.attempts == 2, (
            f"Expected 2 upload attempts (1 failure + 1 success), got {result.attempts}"
        )


# ---------------------------------------------------------------------------
# F1 threshold gating tests (Requirements 14.3, 17.3)
# ---------------------------------------------------------------------------


class TestModelPublishF1Threshold:
    """Verify models are only published when they meet their F1 target.

    Requirements: 14.3, 17.3
    """

    def test_model_not_published_below_target_f1(
        self, publisher: HuggingFacePublisher, sample_model_card: ModelCard
    ) -> None:
        """Model must NOT be published if actual F1 < target F1."""
        result = publisher.publish_model(
            sample_model_card,
            target_f1=0.85,
            actual_f1=0.84,  # Below target
        )
        assert result is False, (
            "Model should not be published when actual F1 (0.84) < target F1 (0.85)"
        )

    def test_model_published_above_target_f1(
        self, publisher: HuggingFacePublisher, sample_model_card: ModelCard
    ) -> None:
        """Model MUST be published if actual F1 >= target F1."""
        result = publisher.publish_model(
            sample_model_card,
            target_f1=0.85,
            actual_f1=0.87,  # Above target
        )
        assert result is True, (
            "Model should be published when actual F1 (0.87) >= target F1 (0.85)"
        )

    def test_model_published_at_exact_target_f1(
        self, publisher: HuggingFacePublisher, sample_model_card: ModelCard
    ) -> None:
        """Model must be published when actual F1 exactly equals target F1."""
        result = publisher.publish_model(
            sample_model_card,
            target_f1=0.85,
            actual_f1=0.85,  # Exactly at target
        )
        assert result is True, (
            "Model should be published when actual F1 (0.85) == target F1 (0.85)"
        )

    def test_model_not_published_well_below_target(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Model must not be published when F1 is significantly below target."""
        card = publisher.generate_model_card(
            model_name="NERTagger",
            repo_id="org/rajnlp-ner",
            random_seed=42,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration="6h 00m",
            evaluation_metrics={"macro_f1": 0.50},
        )
        result = publisher.publish_model(card, target_f1=0.82, actual_f1=0.50)
        assert result is False

    def test_target_f1_constants_are_correct(self) -> None:
        """TARGET_F1 constants must match the requirements."""
        assert TARGET_F1["SentimentClassifier"] == 0.85, (
            "SentimentClassifier target F1 must be 0.85 (Requirement 10.3)"
        )
        assert TARGET_F1["NERTagger"] == 0.82, (
            "NERTagger target F1 must be 0.82 (Requirement 11.3)"
        )
        assert TARGET_F1["ToxicityClassifier"] == 0.79, (
            "ToxicityClassifier target F1 must be 0.79 (Requirement 12.4)"
        )

    def test_all_three_models_publish_above_target(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """All three fine-tuned models should publish when they exceed their targets."""
        model_configs = [
            ("SentimentClassifier", "org/rajnlp-sentiment", 0.87),
            ("NERTagger", "org/rajnlp-ner", 0.84),
            ("ToxicityClassifier", "org/rajnlp-toxicity", 0.81),
        ]
        for model_name, repo_id, actual_f1 in model_configs:
            card = publisher.generate_model_card(
                model_name=model_name,
                repo_id=repo_id,
                random_seed=42,
                hardware_config="1× NVIDIA A100 80GB",
                training_duration="5h 00m",
                evaluation_metrics={"macro_f1": actual_f1},
            )
            target = TARGET_F1[model_name]
            result = publisher.publish_model(card, target_f1=target, actual_f1=actual_f1)
            assert result is True, (
                f"{model_name} should publish with F1={actual_f1} >= target={target}"
            )

    def test_all_three_models_blocked_below_target(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """All three fine-tuned models should be blocked when below their targets."""
        model_configs = [
            ("SentimentClassifier", "org/rajnlp-sentiment", 0.80),
            ("NERTagger", "org/rajnlp-ner", 0.75),
            ("ToxicityClassifier", "org/rajnlp-toxicity", 0.70),
        ]
        for model_name, repo_id, actual_f1 in model_configs:
            card = publisher.generate_model_card(
                model_name=model_name,
                repo_id=repo_id,
                random_seed=42,
                hardware_config="1× NVIDIA A100 80GB",
                training_duration="5h 00m",
                evaluation_metrics={"macro_f1": actual_f1},
            )
            target = TARGET_F1[model_name]
            result = publisher.publish_model(card, target_f1=target, actual_f1=actual_f1)
            assert result is False, (
                f"{model_name} should be blocked with F1={actual_f1} < target={target}"
            )


# ---------------------------------------------------------------------------
# Dataset card structure tests
# ---------------------------------------------------------------------------


class TestDatasetCardStructure:
    """Verify dataset card has correct structure and repo_id."""

    def test_dataset_card_repo_id_matches(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """DatasetCard.repo_id must match the provided repo_id."""
        repo_id = "myorg/rajnlp-50k-test"
        card = publisher.generate_dataset_card(repo_id)
        assert card.repo_id == repo_id

    def test_dataset_card_content_is_markdown(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card content must be non-empty markdown."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        assert len(card.content) > 100, "Dataset card content must be non-trivial"
        assert "#" in card.content, "Dataset card must contain markdown headers"

    def test_dataset_card_has_yaml_frontmatter(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Dataset card must have YAML frontmatter (HuggingFace convention)."""
        card = publisher.generate_dataset_card("org/rajnlp-50k")
        assert card.content.startswith("---"), (
            "Dataset card must start with YAML frontmatter (---)"
        )


class TestModelCardStructure:
    """Verify model card has correct structure."""

    def test_model_card_repo_id_matches(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """ModelCard.repo_id must match the provided repo_id."""
        repo_id = "myorg/rajnlp-sentiment-test"
        card = publisher.generate_model_card(
            model_name="SentimentClassifier",
            repo_id=repo_id,
            random_seed=42,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration="4h 32m",
            evaluation_metrics={"macro_f1": 0.87},
        )
        assert card.repo_id == repo_id

    def test_model_card_model_name_matches(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """ModelCard.model_name must match the provided model_name."""
        card = publisher.generate_model_card(
            model_name="NERTagger",
            repo_id="org/rajnlp-ner",
            random_seed=42,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration="6h 00m",
            evaluation_metrics={"macro_f1": 0.84},
        )
        assert card.model_name == "NERTagger"

    def test_model_card_evaluation_metrics_in_content(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Model card must include evaluation metrics."""
        card = publisher.generate_model_card(
            model_name="ToxicityClassifier",
            repo_id="org/rajnlp-toxicity",
            random_seed=7,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration="8h 00m",
            evaluation_metrics={"macro_f1": 0.81, "precision": 0.80},
        )
        assert "macro_f1" in card.content or "Evaluation" in card.content
        assert "0.8100" in card.content or "0.81" in card.content

    def test_model_card_has_yaml_frontmatter(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Model card must have YAML frontmatter (HuggingFace convention)."""
        card = publisher.generate_model_card(
            model_name="SentimentClassifier",
            repo_id="org/rajnlp-sentiment",
            random_seed=42,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration="4h 32m",
            evaluation_metrics={"macro_f1": 0.87},
        )
        assert card.content.startswith("---"), (
            "Model card must start with YAML frontmatter (---)"
        )

    def test_model_card_includes_intended_use(
        self, publisher: HuggingFacePublisher
    ) -> None:
        """Model card must include intended use section."""
        card = publisher.generate_model_card(
            model_name="SentimentClassifier",
            repo_id="org/rajnlp-sentiment",
            random_seed=42,
            hardware_config="1× NVIDIA A100 80GB",
            training_duration="4h 32m",
            evaluation_metrics={"macro_f1": 0.87},
        )
        assert "Intended Use" in card.content or "intended use" in card.content.lower()
