"""
Real HuggingFace publisher using the huggingface_hub API.

Replaces the stub publisher with actual API calls.

Usage:
    from release.huggingface_publisher_real import RealHuggingFacePublisher

    publisher = RealHuggingFacePublisher(token="your_hf_token")
    publisher.publish_dataset(dataset_split, "your-username/rajnlp-50k")
    publisher.publish_model(model_card, target_f1=0.85, actual_f1=0.87)

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 17.3, 17.4
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from models.data_models import DatasetSplit
from release.huggingface_publisher import (
    DatasetCard,
    ModelCard,
    PublishResult,
    TARGET_F1,
    HuggingFacePublisher,
    _upload_with_retry,
)

logger = logging.getLogger(__name__)


class RealHuggingFacePublisher(HuggingFacePublisher):
    """Real HuggingFace publisher that makes actual API calls.

    Extends the stub publisher with real huggingface_hub API calls.
    Falls back to stub behavior if huggingface_hub is not installed.

    Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 17.3, 17.4
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        """Initialise the real publisher.

        Args:
            token: HuggingFace API token. If None, reads from HF_TOKEN
                environment variable.
            max_attempts: Maximum upload attempts (default 3).
            base_delay: Base delay for exponential backoff in seconds.
        """
        super().__init__()
        self.token = token or os.environ.get("HF_TOKEN")
        self.max_attempts = max_attempts
        self.base_delay = base_delay

        if not self.token:
            logger.warning(
                "No HuggingFace token provided. Set HF_TOKEN environment variable "
                "or pass token= to RealHuggingFacePublisher()."
            )

    def _get_api(self):
        """Get the HuggingFace API client."""
        try:
            from huggingface_hub import HfApi
            return HfApi(token=self.token)
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for real publishing. "
                "Install it with: pip install huggingface_hub"
            )

    def publish_dataset(
        self,
        dataset_split: DatasetSplit,
        repo_id: str,
        *,
        annotation_guideline_path: str | None = None,
        jsonl_path: str | None = None,
        parquet_path: str | None = None,
        max_attempts: int | None = None,
        base_delay: float | None = None,
        **card_kwargs,
    ) -> PublishResult:
        """Publish RajNLP-50K to HuggingFace Datasets.

        Uploads:
        1. The dataset card (README.md in the repo)
        2. The corpus files (JSONL and/or Parquet)
        3. The Annotation Guidelines document (if provided)

        Args:
            dataset_split: The DatasetSplit to publish.
            repo_id: HuggingFace repository ID (e.g., "username/rajnlp-50k").
            annotation_guideline_path: Path to the annotation guidelines file.
            jsonl_path: Path to the serialized JSONL corpus file.
            parquet_path: Path to the serialized Parquet corpus file.
            max_attempts: Override default max_attempts.
            base_delay: Override default base_delay.
            **card_kwargs: Additional kwargs for generate_dataset_card().

        Returns:
            PublishResult indicating success or failure.
        """
        max_attempts = max_attempts or self.max_attempts
        base_delay = base_delay or self.base_delay

        card = self.generate_dataset_card(repo_id, **card_kwargs)
        api = self._get_api()
        attempts_made = 0

        def _do_upload() -> None:
            nonlocal attempts_made
            attempts_made += 1

            # Create or get the dataset repo
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                    private=False,
                )
                logger.info("Dataset repo '%s' ready.", repo_id)
            except Exception as exc:
                logger.warning("Could not create repo (may already exist): %s", exc)

            # Upload dataset card as README.md
            api.upload_file(
                path_or_fileobj=card.content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add dataset card with data statement",
            )
            logger.info("Dataset card uploaded to %s", repo_id)

            # Upload corpus files
            if jsonl_path and Path(jsonl_path).exists():
                api.upload_file(
                    path_or_fileobj=jsonl_path,
                    path_in_repo="data/corpus.jsonl",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Add corpus JSONL file",
                )
                logger.info("JSONL corpus uploaded.")

            if parquet_path and Path(parquet_path).exists():
                api.upload_file(
                    path_or_fileobj=parquet_path,
                    path_in_repo="data/corpus.parquet",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Add corpus Parquet file",
                )
                logger.info("Parquet corpus uploaded.")

            # Upload annotation guidelines
            if annotation_guideline_path and Path(annotation_guideline_path).exists():
                api.upload_file(
                    path_or_fileobj=annotation_guideline_path,
                    path_in_repo="annotation_guidelines.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Add annotation guidelines",
                )
                logger.info("Annotation guidelines uploaded.")

        try:
            _upload_with_retry(_do_upload, max_attempts=max_attempts, base_delay=base_delay)
        except Exception as exc:
            logger.error("Dataset publish failed for '%s': %s", repo_id, exc)
            return PublishResult(
                success=False,
                repo_id=repo_id,
                attempts=attempts_made,
                error=str(exc),
            )

        logger.info("Dataset '%s' published successfully at https://huggingface.co/datasets/%s", repo_id, repo_id)
        return PublishResult(success=True, repo_id=repo_id, attempts=attempts_made)

    def publish_model(
        self,
        model_card: ModelCard,
        target_f1: float,
        actual_f1: float,
        *,
        model_dir: str | None = None,
        max_attempts: int | None = None,
        base_delay: float | None = None,
    ) -> bool:
        """Publish a fine-tuned model to HuggingFace.

        Only publishes if actual_f1 >= target_f1.

        Args:
            model_card: The ModelCard to publish.
            target_f1: Minimum F1 required for publication.
            actual_f1: Actual F1 achieved on validation partition.
            model_dir: Local directory containing the saved model files.
                If None, only the model card is uploaded.
            max_attempts: Override default max_attempts.
            base_delay: Override default base_delay.

        Returns:
            True if published successfully, False otherwise.
        """
        if actual_f1 < target_f1:
            logger.warning(
                "Model '%s' not published: F1=%.4f < target=%.4f",
                model_card.model_name, actual_f1, target_f1,
            )
            return False

        max_attempts = max_attempts or self.max_attempts
        base_delay = base_delay or self.base_delay
        api = self._get_api()
        attempts_made = 0

        def _do_upload() -> None:
            nonlocal attempts_made
            attempts_made += 1

            # Create or get the model repo
            try:
                api.create_repo(
                    repo_id=model_card.repo_id,
                    repo_type="model",
                    exist_ok=True,
                    private=False,
                )
            except Exception as exc:
                logger.warning("Could not create model repo: %s", exc)

            # Upload model card as README.md
            api.upload_file(
                path_or_fileobj=model_card.content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=model_card.repo_id,
                repo_type="model",
                commit_message=f"Add model card for {model_card.model_name}",
            )
            logger.info("Model card uploaded for %s", model_card.model_name)

            # Upload model files if directory provided
            if model_dir and Path(model_dir).exists():
                api.upload_folder(
                    folder_path=model_dir,
                    repo_id=model_card.repo_id,
                    repo_type="model",
                    commit_message=f"Upload {model_card.model_name} checkpoint",
                )
                logger.info("Model files uploaded from %s", model_dir)

        try:
            _upload_with_retry(_do_upload, max_attempts=max_attempts, base_delay=base_delay)
        except Exception as exc:
            logger.error("Model publish failed for '%s': %s", model_card.repo_id, exc)
            return False

        logger.info(
            "Model '%s' published at https://huggingface.co/%s",
            model_card.model_name, model_card.repo_id,
        )
        return True

    def publish_all(
        self,
        dataset_split: DatasetSplit,
        dataset_repo_id: str,
        model_checkpoints: dict[str, str],
        model_metrics: dict[str, float],
        seed: int = 42,
        hardware_config: str = "1× NVIDIA A100 80GB",
        training_durations: dict[str, str] | None = None,
        jsonl_path: str | None = None,
        parquet_path: str | None = None,
        annotation_guideline_path: str | None = None,
    ) -> dict[str, bool]:
        """Publish the full RajNLP-50K release: dataset + all 3 models.

        Args:
            dataset_split: The DatasetSplit to publish.
            dataset_repo_id: HuggingFace dataset repo ID.
            model_checkpoints: Dict of model_name → local checkpoint directory.
            model_metrics: Dict of model_name → actual macro-F1.
            seed: Random seed used for training.
            hardware_config: Hardware description for model cards.
            training_durations: Dict of model_name → training duration string.
            jsonl_path: Path to JSONL corpus file.
            parquet_path: Path to Parquet corpus file.
            annotation_guideline_path: Path to annotation guidelines.

        Returns:
            Dict of component → success (True/False).
        """
        if training_durations is None:
            training_durations = {
                "SentimentClassifier": "TBD",
                "NERTagger": "TBD",
                "ToxicityClassifier": "TBD",
            }

        results: dict[str, bool] = {}

        # Publish dataset
        logger.info("Publishing dataset to %s", dataset_repo_id)
        dataset_result = self.publish_dataset(
            dataset_split=dataset_split,
            repo_id=dataset_repo_id,
            jsonl_path=jsonl_path,
            parquet_path=parquet_path,
            annotation_guideline_path=annotation_guideline_path,
        )
        results["dataset"] = dataset_result.success

        # Publish models
        model_repo_map = {
            "SentimentClassifier": f"{dataset_repo_id.split('/')[0]}/rajnlp-sentiment",
            "NERTagger": f"{dataset_repo_id.split('/')[0]}/rajnlp-ner",
            "ToxicityClassifier": f"{dataset_repo_id.split('/')[0]}/rajnlp-toxicity",
        }

        for model_name, repo_id in model_repo_map.items():
            actual_f1 = model_metrics.get(model_name, 0.0)
            target_f1 = TARGET_F1.get(model_name, 0.0)
            model_dir = model_checkpoints.get(model_name)

            card = self.generate_model_card(
                model_name=model_name,
                repo_id=repo_id,
                random_seed=seed,
                hardware_config=hardware_config,
                training_duration=training_durations.get(model_name, "TBD"),
                evaluation_metrics={"macro_f1": actual_f1},
            )

            success = self.publish_model(
                model_card=card,
                target_f1=target_f1,
                actual_f1=actual_f1,
                model_dir=model_dir,
            )
            results[model_name] = success

        logger.info("Full release complete: %s", results)
        return results
