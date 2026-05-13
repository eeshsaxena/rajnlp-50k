"""
Platform_Split_Evaluation for the RajNLP-50K evaluation pipeline.

Implements cross-platform generalization evaluation:
- Train on the Twitter/X partition; evaluate on the ShareChat partition.
- Train on the ShareChat partition; evaluate on the Twitter/X partition.

Reports platform-split macro-averaged F1 for each task and each direction.

Requirements: 13.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from models.data_models import (
    AnnotatedSentence,
    DatasetSplit,
)
from models.ner_tagger import NERTagger
from models.sentiment_classifier import SentimentClassifier
from models.toxicity_classifier import ToxicityClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PlatformSplitResult
# ---------------------------------------------------------------------------


@dataclass
class PlatformSplitResult:
    """Result from a single platform-split evaluation run.

    Attributes:
        task: Task name — one of "sentiment", "ner", "toxicity".
        train_platform: Platform used for training — "twitter" or "sharechat".
        eval_platform: Platform used for evaluation — "twitter" or "sharechat".
        macro_f1: Macro-averaged F1 on the evaluation platform's test sentences.
    """

    task: str
    train_platform: str
    eval_platform: str
    macro_f1: float


# ---------------------------------------------------------------------------
# Helpers: partition sentences by platform
# ---------------------------------------------------------------------------


def _filter_by_platform(
    sentences: list[AnnotatedSentence],
    platform: str,
) -> list[AnnotatedSentence]:
    """Return only sentences from the given platform.

    Args:
        sentences: List of annotated sentences.
        platform: Platform identifier — "twitter" or "sharechat".

    Returns:
        Filtered list containing only sentences from *platform*.
    """
    return [s for s in sentences if s.platform == platform]


def _split_by_platform(
    dataset: DatasetSplit,
) -> tuple[list[AnnotatedSentence], list[AnnotatedSentence]]:
    """Split the full dataset into Twitter and ShareChat sentence pools.

    Combines train + validation + test partitions for each platform so that
    the platform-split evaluation can define its own train/eval boundaries.

    Args:
        dataset: The full DatasetSplit (train + validation + test).

    Returns:
        Tuple of (twitter_sentences, sharechat_sentences).
    """
    all_sentences = dataset.train + dataset.validation + dataset.test
    twitter = _filter_by_platform(all_sentences, "twitter")
    sharechat = _filter_by_platform(all_sentences, "sharechat")
    return twitter, sharechat


# ---------------------------------------------------------------------------
# run_platform_split_evaluation
# ---------------------------------------------------------------------------


def run_platform_split_evaluation(
    dataset: DatasetSplit,
    sentiment_clf: SentimentClassifier,
    ner_tagger: NERTagger,
    toxicity_clf: ToxicityClassifier,
    seed: int = 42,
) -> list[PlatformSplitResult]:
    """Train on one platform, evaluate on the other, for all three tasks.

    Performs four training+evaluation runs:
    1. Sentiment: train on Twitter, evaluate on ShareChat.
    2. Sentiment: train on ShareChat, evaluate on Twitter.
    3. NER: train on Twitter, evaluate on ShareChat.
    4. NER: train on ShareChat, evaluate on Twitter.
    5. Toxicity: train on Twitter, evaluate on ShareChat.
    6. Toxicity: train on ShareChat, evaluate on Twitter.

    Each model is trained using its ``train()`` method with the platform
    training sentences (using the ``train`` split field as a proxy — all
    sentences from the platform are used as training data, and the other
    platform's sentences are used for evaluation).

    Args:
        dataset: The full DatasetSplit containing sentences from both platforms.
        sentiment_clf: SentimentClassifier instance to train and evaluate.
        ner_tagger: NERTagger instance to train and evaluate.
        toxicity_clf: ToxicityClassifier instance to train and evaluate.
        seed: Random seed for all training runs (default 42).

    Returns:
        List of PlatformSplitResult objects — one per (task, direction) pair,
        yielding 6 results total (3 tasks × 2 directions).

    Requirements: 13.5
    """
    twitter_sentences, sharechat_sentences = _split_by_platform(dataset)

    logger.info(
        "run_platform_split_evaluation: twitter=%d sentences, sharechat=%d sentences",
        len(twitter_sentences), len(sharechat_sentences),
    )

    results: list[PlatformSplitResult] = []

    # ------------------------------------------------------------------
    # Sentiment: twitter → sharechat
    # ------------------------------------------------------------------
    results.append(_run_sentiment_split(
        sentiment_clf=sentiment_clf,
        train_sentences=twitter_sentences,
        eval_sentences=sharechat_sentences,
        train_platform="twitter",
        eval_platform="sharechat",
        seed=seed,
    ))

    # ------------------------------------------------------------------
    # Sentiment: sharechat → twitter
    # ------------------------------------------------------------------
    results.append(_run_sentiment_split(
        sentiment_clf=sentiment_clf,
        train_sentences=sharechat_sentences,
        eval_sentences=twitter_sentences,
        train_platform="sharechat",
        eval_platform="twitter",
        seed=seed,
    ))

    # ------------------------------------------------------------------
    # NER: twitter → sharechat
    # ------------------------------------------------------------------
    results.append(_run_ner_split(
        ner_tagger=ner_tagger,
        train_sentences=twitter_sentences,
        eval_sentences=sharechat_sentences,
        train_platform="twitter",
        eval_platform="sharechat",
        seed=seed,
    ))

    # ------------------------------------------------------------------
    # NER: sharechat → twitter
    # ------------------------------------------------------------------
    results.append(_run_ner_split(
        ner_tagger=ner_tagger,
        train_sentences=sharechat_sentences,
        eval_sentences=twitter_sentences,
        train_platform="sharechat",
        eval_platform="twitter",
        seed=seed,
    ))

    # ------------------------------------------------------------------
    # Toxicity: twitter → sharechat
    # ------------------------------------------------------------------
    results.append(_run_toxicity_split(
        toxicity_clf=toxicity_clf,
        train_sentences=twitter_sentences,
        eval_sentences=sharechat_sentences,
        train_platform="twitter",
        eval_platform="sharechat",
        seed=seed,
    ))

    # ------------------------------------------------------------------
    # Toxicity: sharechat → twitter
    # ------------------------------------------------------------------
    results.append(_run_toxicity_split(
        toxicity_clf=toxicity_clf,
        train_sentences=sharechat_sentences,
        eval_sentences=twitter_sentences,
        train_platform="sharechat",
        eval_platform="twitter",
        seed=seed,
    ))

    logger.info(
        "run_platform_split_evaluation: completed %d split evaluations",
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Private helpers: per-task split runners
# ---------------------------------------------------------------------------


def _run_sentiment_split(
    sentiment_clf: SentimentClassifier,
    train_sentences: list[AnnotatedSentence],
    eval_sentences: list[AnnotatedSentence],
    train_platform: str,
    eval_platform: str,
    seed: int,
) -> PlatformSplitResult:
    """Train sentiment classifier on train_sentences, evaluate on eval_sentences."""
    logger.info(
        "Sentiment split: train_platform=%s (%d), eval_platform=%s (%d)",
        train_platform, len(train_sentences),
        eval_platform, len(eval_sentences),
    )

    # Use a fraction of train sentences as validation (last 10%)
    n_val = max(1, len(train_sentences) // 10)
    val_sentences = train_sentences[-n_val:] if train_sentences else []
    actual_train = train_sentences[:-n_val] if len(train_sentences) > n_val else train_sentences

    sentiment_clf.train(actual_train, val_sentences, seed=seed)

    if eval_sentences:
        metrics = sentiment_clf.evaluate(eval_sentences)
        macro_f1 = metrics.macro_f1
    else:
        logger.warning(
            "No eval sentences for platform=%s; returning macro_f1=0.0",
            eval_platform,
        )
        macro_f1 = 0.0

    logger.info(
        "Sentiment split %s→%s: macro_f1=%.4f",
        train_platform, eval_platform, macro_f1,
    )
    return PlatformSplitResult(
        task="sentiment",
        train_platform=train_platform,
        eval_platform=eval_platform,
        macro_f1=macro_f1,
    )


def _run_ner_split(
    ner_tagger: NERTagger,
    train_sentences: list[AnnotatedSentence],
    eval_sentences: list[AnnotatedSentence],
    train_platform: str,
    eval_platform: str,
    seed: int,
) -> PlatformSplitResult:
    """Train NER tagger on train_sentences, evaluate on eval_sentences."""
    logger.info(
        "NER split: train_platform=%s (%d), eval_platform=%s (%d)",
        train_platform, len(train_sentences),
        eval_platform, len(eval_sentences),
    )

    n_val = max(1, len(train_sentences) // 10)
    val_sentences = train_sentences[-n_val:] if train_sentences else []
    actual_train = train_sentences[:-n_val] if len(train_sentences) > n_val else train_sentences

    ner_tagger.train(actual_train, val_sentences, seed=seed)

    if eval_sentences:
        metrics = ner_tagger.evaluate(eval_sentences)
        macro_f1 = metrics.macro_f1
    else:
        logger.warning(
            "No eval sentences for platform=%s; returning macro_f1=0.0",
            eval_platform,
        )
        macro_f1 = 0.0

    logger.info(
        "NER split %s→%s: macro_f1=%.4f",
        train_platform, eval_platform, macro_f1,
    )
    return PlatformSplitResult(
        task="ner",
        train_platform=train_platform,
        eval_platform=eval_platform,
        macro_f1=macro_f1,
    )


def _run_toxicity_split(
    toxicity_clf: ToxicityClassifier,
    train_sentences: list[AnnotatedSentence],
    eval_sentences: list[AnnotatedSentence],
    train_platform: str,
    eval_platform: str,
    seed: int,
) -> PlatformSplitResult:
    """Train toxicity classifier on train_sentences, evaluate on eval_sentences."""
    logger.info(
        "Toxicity split: train_platform=%s (%d), eval_platform=%s (%d)",
        train_platform, len(train_sentences),
        eval_platform, len(eval_sentences),
    )

    n_val = max(1, len(train_sentences) // 10)
    val_sentences = train_sentences[-n_val:] if train_sentences else []
    actual_train = train_sentences[:-n_val] if len(train_sentences) > n_val else train_sentences

    toxicity_clf.train(actual_train, val_sentences, seed=seed)

    if eval_sentences:
        metrics = toxicity_clf.evaluate(eval_sentences)
        macro_f1 = metrics.macro_f1
    else:
        logger.warning(
            "No eval sentences for platform=%s; returning macro_f1=0.0",
            eval_platform,
        )
        macro_f1 = 0.0

    logger.info(
        "Toxicity split %s→%s: macro_f1=%.4f",
        train_platform, eval_platform, macro_f1,
    )
    return PlatformSplitResult(
        task="toxicity",
        train_platform=train_platform,
        eval_platform=eval_platform,
        macro_f1=macro_f1,
    )
