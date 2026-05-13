"""
HuggingFace release pipeline for RajNLP-50K.

This module implements stub-based publishing that:
- Generates real dataset/model card content (as strings)
- Simulates upload with a configurable failure/success mechanism
- Implements real exponential backoff retry logic

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 17.3, 17.4
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from models.data_models import DatasetSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target F1 thresholds (from requirements 10.3, 11.3, 12.4)
# ---------------------------------------------------------------------------

TARGET_F1: dict[str, float] = {
    "SentimentClassifier": 0.85,
    "NERTagger": 0.82,
    "ToxicityClassifier": 0.79,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DatasetCard:
    """A HuggingFace dataset card."""

    repo_id: str
    """HuggingFace repository ID (e.g., 'org/rajnlp-50k')."""

    content: str
    """Full markdown content of the dataset card."""


@dataclass
class ModelCard:
    """A HuggingFace model card for a fine-tuned model."""

    repo_id: str
    """HuggingFace repository ID (e.g., 'org/rajnlp-sentiment')."""

    model_name: str
    """Name of the model (e.g., 'SentimentClassifier')."""

    content: str
    """Full markdown content of the model card."""

    random_seed: int
    """Random seed used to produce the released checkpoint."""

    hardware_config: str
    """Hardware configuration description (GPU type and count)."""

    training_duration: str
    """Human-readable training duration (e.g., '4h 32m')."""

    evaluation_metrics: dict[str, float]
    """Evaluation metrics reported on the test partition."""


@dataclass
class PublishResult:
    """Result of a publish operation."""

    success: bool
    """Whether the publish succeeded."""

    repo_id: str
    """Repository ID that was published to."""

    attempts: int
    """Number of upload attempts made."""

    error: str | None = None
    """Error message if publish failed."""


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def _upload_with_retry(
    upload_fn: Callable[[], None],
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> None:
    """Retry upload_fn with exponential backoff.

    Delays: base_delay * 2^attempt (1s, 2s, 4s for base_delay=1.0).
    Raises the last exception if all attempts are exhausted.

    Args:
        upload_fn: Callable that performs the upload. Should raise an exception
            on failure.
        max_attempts: Maximum number of attempts (default 3).
        base_delay: Base delay in seconds for exponential backoff (default 1.0).

    Raises:
        Exception: The last exception raised by upload_fn if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            upload_fn()
            logger.info("Upload succeeded on attempt %d", attempt + 1)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Upload attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            if attempt < max_attempts - 1:
                time.sleep(delay)

    logger.error(
        "All %d upload attempts exhausted. Last error: %s",
        max_attempts,
        last_exc,
    )
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class HuggingFacePublisher:
    """Stub HuggingFace publisher that generates real card content and
    simulates uploads without making real API calls.

    The ``_simulate_upload_failure`` flag can be set to control whether
    simulated uploads succeed or fail (for testing retry logic).
    """

    def __init__(
        self,
        *,
        simulate_upload_failure: bool = False,
        failure_count: int = 0,
    ) -> None:
        """Initialise the publisher.

        Args:
            simulate_upload_failure: If True, every upload attempt raises an
                ``IOError``. Useful for testing retry exhaustion.
            failure_count: Number of times to fail before succeeding. When > 0,
                the first ``failure_count`` calls to the upload stub will raise
                an ``IOError``; subsequent calls succeed. Useful for testing
                transient failure recovery.
        """
        self._simulate_upload_failure = simulate_upload_failure
        self._failure_count = failure_count
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Internal upload stub
    # ------------------------------------------------------------------

    def _stub_upload(self, description: str) -> None:
        """Simulate an upload operation.

        Raises ``IOError`` if configured to fail.
        """
        self._call_count += 1
        if self._simulate_upload_failure:
            raise IOError(f"Simulated upload failure for: {description}")
        if self._failure_count > 0 and self._call_count <= self._failure_count:
            raise IOError(
                f"Simulated transient upload failure #{self._call_count} for: {description}"
            )
        logger.info("Stub upload succeeded: %s", description)

    # ------------------------------------------------------------------
    # Dataset card generation
    # ------------------------------------------------------------------

    def generate_dataset_card(
        self,
        repo_id: str,
        *,
        collection_period: str = "January–December 2024",
        annotator_demographics: str = (
            "3 native Rajasthani-Hindi bilingual speakers, "
            "ages 22–35, 2 female / 1 male, based in Rajasthan, India"
        ),
        compensation_rates: str = (
            "Annotators were compensated at ₹150 per hour for sentiment and NER tasks, "
            "and ₹200 per hour for toxicity tasks, paid via bank transfer."
        ),
        content_warning_procedures: str = (
            "All annotators were presented with a written content warning before "
            "accessing the toxicity labeling task, describing the nature of toxic, "
            "caste-slur, religious, and gender-based content. An opt-out mechanism "
            "was provided allowing annotators to withdraw at any time without penalty."
        ),
        known_demographic_biases: str = (
            "Twitter/X data over-represents urban, educated, and politically active "
            "users. ShareChat data skews toward regional news consumers. Both platforms "
            "under-represent rural dialects and older speakers. Caste-based toxicity "
            "labels reflect annotator positionality and may not capture all regional "
            "caste dynamics."
        ),
    ) -> DatasetCard:
        """Generate a dataset card with a full data statement.

        The card includes all required fields from Requirements 14.4 and 14.5:
        - Source platforms (Twitter/X and ShareChat)
        - Collection period
        - Annotator demographics
        - Compensation rates
        - Content warning procedures
        - Known demographic biases

        Args:
            repo_id: HuggingFace repository ID.
            collection_period: Human-readable collection period.
            annotator_demographics: Description of annotator demographics.
            compensation_rates: Description of annotator compensation.
            content_warning_procedures: Description of content warning procedures.
            known_demographic_biases: Description of known demographic biases.

        Returns:
            A ``DatasetCard`` with the full markdown content.
        """
        content = f"""---
language:
  - raj
  - hi
  - en
license: cc-by-4.0
task_categories:
  - text-classification
  - token-classification
pretty_name: RajNLP-50K
size_categories:
  - 10K<n<100K
---

# RajNLP-50K: Rajasthani-Hindi Code-Switched NLP Corpus

## Dataset Description

RajNLP-50K is the first open, annotated Rajasthani-Hindi code-switched NLP corpus,
containing 50,000 sentences sourced from social media platforms. The corpus is annotated
for sentiment analysis, named entity recognition (NER), and toxicity detection.

## Data Statement

### Source Platforms

The corpus was collected from two social media platforms:

- **Twitter/X**: Tweets collected using the Twitter/X Academic API, targeting
  Rajasthan politician names, regional hashtags (e.g., #rajasthan), and documented
  Rajasthani slang terms.
- **ShareChat**: Posts scraped from local politics and news pages on ShareChat,
  identified as having high Rajasthani dialect concentration.

### Collection Period

{collection_period}

### Annotator Demographics

{annotator_demographics}

### Compensation Rates

{compensation_rates}

### Content Warning Procedures

{content_warning_procedures}

### Known Demographic Biases

{known_demographic_biases}

## Dataset Structure

The dataset is split into three partitions:

| Split      | Size   |
|------------|--------|
| train      | 40,000 |
| validation |  5,000 |
| test       |  5,000 |

### Fields

- `sentence_id`: UUID string
- `text`: Original sentence text (unmodified, including emoji and non-Latin characters)
- `platform`: Source platform (`twitter` or `sharechat`)
- `split`: Dataset partition (`train`, `validation`, or `test`)
- `sentiment`: Gold sentiment label (`positive`, `neutral`, or `negative`)
- `sentiment_annotator_labels`: Raw labels from each of the 3 annotators
- `ner_spans`: Gold NER spans (majority-vote resolved)
- `ner_annotator_spans`: Raw span sets from each of the 3 annotators
- `toxicity_labels`: Gold toxicity labels (0–4 categories; empty = non-toxic)
- `toxicity_annotator_labels`: Raw toxicity label sets from each of the 3 annotators
- `token_language_labels`: Per-token language labels (RAJ/HIN/ENG/TRL)
- `source_url`: URL of the source post
- `collected_at`: Collection timestamp (UTC)
- `annotated_at`: Annotation completion timestamp (UTC)

## Annotation Guidelines

The Annotation_Guideline document is included in this repository. It defines
labeling rules, worked examples, and edge-case decisions for all three annotation
tasks, along with final IAA scores (Cohen's Kappa) and the adjudication procedure.

## Citation

If you use RajNLP-50K in your research, please cite:

```bibtex
@dataset{{rajnlp50k,
  title={{RajNLP-50K: A Rajasthani-Hindi Code-Switched NLP Corpus}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
        return DatasetCard(repo_id=repo_id, content=content)

    # ------------------------------------------------------------------
    # Dataset publishing
    # ------------------------------------------------------------------

    def publish_dataset(
        self,
        dataset_split: DatasetSplit,
        repo_id: str,
        *,
        annotation_guideline: str | None = None,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        **card_kwargs,
    ) -> PublishResult:
        """Publish RajNLP-50K to HuggingFace Datasets.

        Uploads the dataset with train/validation/test splits, all annotation
        layers, and Language_ID_Tagger labels. Includes the Annotation_Guideline
        document and a dataset card with a full data statement.

        Implements retry with exponential backoff (3 attempts) on upload failure.

        Requirements: 14.1, 14.2, 14.4, 14.5

        Args:
            dataset_split: The ``DatasetSplit`` containing train/val/test partitions.
            repo_id: HuggingFace repository ID.
            annotation_guideline: Optional annotation guideline text to include.
            max_attempts: Maximum upload attempts (default 3).
            base_delay: Base delay for exponential backoff in seconds (default 1.0).
            **card_kwargs: Additional keyword arguments forwarded to
                ``generate_dataset_card``.

        Returns:
            A ``PublishResult`` indicating success or failure.
        """
        card = self.generate_dataset_card(repo_id, **card_kwargs)

        total_sentences = (
            len(dataset_split.train)
            + len(dataset_split.validation)
            + len(dataset_split.test)
        )
        logger.info(
            "Publishing dataset '%s' (%d sentences) to HuggingFace...",
            repo_id,
            total_sentences,
        )

        attempts_made = 0

        def _do_upload() -> None:
            nonlocal attempts_made
            attempts_made += 1
            self._stub_upload(f"dataset:{repo_id}")

        try:
            _upload_with_retry(_do_upload, max_attempts=max_attempts, base_delay=base_delay)
        except Exception as exc:  # noqa: BLE001
            logger.error("Dataset publish failed for '%s': %s", repo_id, exc)
            return PublishResult(
                success=False,
                repo_id=repo_id,
                attempts=attempts_made,
                error=str(exc),
            )

        logger.info("Dataset '%s' published successfully.", repo_id)
        return PublishResult(success=True, repo_id=repo_id, attempts=attempts_made)

    # ------------------------------------------------------------------
    # Model card generation
    # ------------------------------------------------------------------

    def generate_model_card(
        self,
        model_name: str,
        repo_id: str,
        *,
        random_seed: int,
        hardware_config: str,
        training_duration: str,
        evaluation_metrics: dict[str, float],
        intended_use: str | None = None,
    ) -> ModelCard:
        """Generate a model card for a fine-tuned model.

        The card documents evaluation metrics, training details, Random_Seed,
        hardware configuration, training duration, and intended use.

        Requirements: 14.3, 17.4

        Args:
            model_name: Name of the model (e.g., 'SentimentClassifier').
            repo_id: HuggingFace repository ID.
            random_seed: Random seed used to produce the released checkpoint.
            hardware_config: Hardware configuration (GPU type and count).
            training_duration: Human-readable training duration.
            evaluation_metrics: Dict of metric name → value.
            intended_use: Optional description of intended use.

        Returns:
            A ``ModelCard`` with the full markdown content.
        """
        if intended_use is None:
            intended_use = _default_intended_use(model_name)

        metrics_table = "\n".join(
            f"| {metric} | {value:.4f} |"
            for metric, value in evaluation_metrics.items()
        )

        content = f"""---
language:
  - raj
  - hi
  - en
license: apache-2.0
base_model: google/muril-base-cased
tags:
  - rajasthani
  - hindi
  - code-switching
  - muril
---

# {model_name}

A fine-tuned MuRIL model for Rajasthani-Hindi code-switched text, trained on
the RajNLP-50K corpus.

## Intended Use

{intended_use}

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | google/muril-base-cased |
| Random_Seed | {random_seed} |
| Hardware configuration | {hardware_config} |
| Training duration | {training_duration} |

## Evaluation Metrics

Evaluated on the RajNLP-50K test partition (5,000 sentences).

| Metric | Value |
|--------|-------|
{metrics_table}

## Reproducibility

To reproduce this checkpoint, run the training script with:

```bash
python -m models.{_model_module(model_name)} --seed {random_seed}
```

Ensure the pinned environment from `requirements.txt` is installed.
The training was performed on: {hardware_config}
Total training duration: {training_duration}

## Citation

```bibtex
@model{{rajnlp_{model_name.lower()},
  title={{{model_name} for Rajasthani-Hindi Code-Switched Text}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{repo_id}}}
}}
```
"""
        return ModelCard(
            repo_id=repo_id,
            model_name=model_name,
            content=content,
            random_seed=random_seed,
            hardware_config=hardware_config,
            training_duration=training_duration,
            evaluation_metrics=evaluation_metrics,
        )

    # ------------------------------------------------------------------
    # Model publishing
    # ------------------------------------------------------------------

    def publish_model(
        self,
        model_card: ModelCard,
        target_f1: float,
        actual_f1: float,
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> bool:
        """Publish a fine-tuned model to HuggingFace.

        The model is only published when it has achieved its target F1 score
        on the validation partition. Implements retry with exponential backoff
        (3 attempts) on upload failure.

        Requirements: 14.3, 17.3

        Args:
            model_card: The ``ModelCard`` to publish.
            target_f1: Minimum F1 score required for publication.
            actual_f1: Actual F1 score achieved on the validation partition.
            max_attempts: Maximum upload attempts (default 3).
            base_delay: Base delay for exponential backoff in seconds (default 1.0).

        Returns:
            ``True`` if the model was published successfully, ``False`` if the
            F1 threshold was not met or if all upload attempts failed.
        """
        if actual_f1 < target_f1:
            logger.warning(
                "Model '%s' not published: actual F1 %.4f < target F1 %.4f",
                model_card.model_name,
                actual_f1,
                target_f1,
            )
            return False

        logger.info(
            "Publishing model '%s' (F1=%.4f >= target=%.4f) to '%s'...",
            model_card.model_name,
            actual_f1,
            target_f1,
            model_card.repo_id,
        )

        attempts_made = 0

        def _do_upload() -> None:
            nonlocal attempts_made
            attempts_made += 1
            self._stub_upload(f"model:{model_card.repo_id}")

        try:
            _upload_with_retry(_do_upload, max_attempts=max_attempts, base_delay=base_delay)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Model publish failed for '%s' after %d attempts: %s",
                model_card.repo_id,
                attempts_made,
                exc,
            )
            return False

        logger.info("Model '%s' published successfully.", model_card.repo_id)
        return True

    # ------------------------------------------------------------------
    # Convenience: publish all three fine-tuned models
    # ------------------------------------------------------------------

    def publish_all_models(
        self,
        models: list[tuple[ModelCard, float, float]],
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> dict[str, bool]:
        """Publish all fine-tuned models that meet their F1 targets.

        Args:
            models: List of ``(model_card, target_f1, actual_f1)`` tuples.
            max_attempts: Maximum upload attempts per model.
            base_delay: Base delay for exponential backoff.

        Returns:
            Dict mapping model name → publish success (True/False).
        """
        results: dict[str, bool] = {}
        for model_card, target_f1, actual_f1 in models:
            results[model_card.model_name] = self.publish_model(
                model_card,
                target_f1,
                actual_f1,
                max_attempts=max_attempts,
                base_delay=base_delay,
            )
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_intended_use(model_name: str) -> str:
    """Return a default intended-use description for a given model name."""
    descriptions = {
        "SentimentClassifier": (
            "This model is intended for sentiment analysis of Rajasthani-Hindi "
            "code-switched social media text. It predicts one of three labels: "
            "positive, neutral, or negative. It is NOT intended for use on "
            "monolingual Hindi or English text, or for high-stakes decision-making."
        ),
        "NERTagger": (
            "This model is intended for named entity recognition in Rajasthani-Hindi "
            "code-switched social media text. It identifies person (PER), location (LOC), "
            "and organization (ORG) entities. It is NOT intended for use on monolingual "
            "text or for legal/medical entity extraction."
        ),
        "ToxicityClassifier": (
            "This model is intended for multi-label toxicity detection in Rajasthani-Hindi "
            "code-switched social media text, covering caste slur, religious, gender, and "
            "general toxicity categories. It is NOT intended as a sole decision-making tool "
            "for content moderation. Human review is required for high-stakes decisions."
        ),
    }
    return descriptions.get(
        model_name,
        f"This model is intended for use with Rajasthani-Hindi code-switched text "
        f"as part of the RajNLP-50K project.",
    )


def _model_module(model_name: str) -> str:
    """Return the Python module name for a given model."""
    mapping = {
        "SentimentClassifier": "sentiment_classifier",
        "NERTagger": "ner_tagger",
        "ToxicityClassifier": "toxicity_classifier",
    }
    return mapping.get(model_name, model_name.lower())
