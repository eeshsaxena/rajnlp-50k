"""
Sentiment_Classifier — 3-class sentence-level sentiment model.

Architecture (production): MuRIL + linear classification head
  (positive / neutral / negative).

For testability without GPU/model downloads, this module implements a
**stub/heuristic classifier** with the same public interface as the real
MuRIL-based classifier.

Heuristic logic
---------------
- ``predict(sentence)`` uses simple keyword matching:
    * positive keywords → "positive"
    * negative keywords → "negative"
    * otherwise → "neutral"
- ``train(train_set, val_set, seed)`` simulates training:
    * calls ``set_all_seeds(seed)``
    * computes class weights from training labels
    * iterates up to 10 epochs, computing mock validation F1
    * applies early stopping (patience=3) on validation macro-F1
    * tracks best_f1 and best_epoch
- ``evaluate(test_set)`` computes real metrics against gold labels using
  the heuristic ``predict``.

Language_ID integration (Task 14.2)
------------------------------------
``SentimentClassifierWithLangID`` extends ``SentimentClassifier`` by
prepending language-ID tag tokens to the input sentence before prediction.
An ablation helper ``run_langid_ablation`` trains both variants and verifies
the ≥ +0.04 macro-F1 improvement requirement.

Requirements: 9.3, 10.1, 10.2, 10.5, 17.1, 17.3
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from models.data_models import (
    AnnotatedSentence,
    ClassificationMetrics,
    SentimentPrediction,
    TrainingLog,
)
from models.reproducibility import set_all_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment keyword lexicons
# ---------------------------------------------------------------------------

_POSITIVE_KEYWORDS: frozenset[str] = frozenset({
    # English
    "good", "great", "excellent", "wonderful", "amazing", "fantastic",
    "happy", "joy", "love", "best", "beautiful", "awesome", "nice",
    "positive", "success", "win", "victory", "proud", "hope", "bright",
    "celebrate", "congratulations", "thanks", "thank", "helpful",
    # Hindi/Rajasthani (Devanagari)
    "अच्छा", "अच्छी", "अच्छे", "बढ़िया", "शानदार", "खुशी", "प्यार",
    "जीत", "सफलता", "धन्यवाद", "सुंदर", "बेहतरीन", "उत्कृष्ट",
    "खुश", "प्रसन्न", "आनंद", "शुभ", "मंगल",
    # Rajasthani-specific
    "घणो", "घणी", "राम-राम",
})

_NEGATIVE_KEYWORDS: frozenset[str] = frozenset({
    # English
    "bad", "terrible", "awful", "horrible", "worst", "hate", "angry",
    "sad", "poor", "fail", "failure", "loss", "corrupt", "wrong",
    "problem", "issue", "crisis", "danger", "threat", "violence",
    "crime", "attack", "kill", "death", "disaster", "shame",
    # Hindi/Rajasthani (Devanagari)
    "बुरा", "बुरी", "बुरे", "खराब", "नफरत", "गुस्सा", "दुख",
    "हार", "भ्रष्ट", "गलत", "समस्या", "संकट", "खतरा", "हिंसा",
    "अपराध", "मौत", "शर्म", "बेकार", "निराश", "परेशान",
    # Rajasthani-specific
    "कोनी",  # negation
})

# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

SENTIMENT_CLASSES: tuple[str, ...] = ("positive", "neutral", "negative")


def compute_class_weights(labels: list[str]) -> dict[str, float]:
    """Compute class weights inversely proportional to class frequency.

    Formula: ``weight[class] = total_samples / (n_classes * count[class])``

    This ensures that minority classes receive higher loss penalties during
    training, counteracting class imbalance.

    Args:
        labels: List of class label strings.

    Returns:
        Dictionary mapping each class label to its weight.  Classes not
        present in *labels* receive a weight of 1.0 (neutral).

    Requirements: 10.1
    """
    if not labels:
        return {cls: 1.0 for cls in SENTIMENT_CLASSES}

    total = len(labels)
    counts = Counter(labels)
    n_classes = len(counts)

    weights: dict[str, float] = {}
    for cls in SENTIMENT_CLASSES:
        count = counts.get(cls, 0)
        if count == 0:
            weights[cls] = 1.0  # default weight for unseen class
        else:
            weights[cls] = total / (n_classes * count)

    return weights


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Monitor a validation metric and signal when training should stop.

    Stops training when the metric has not improved for ``patience``
    consecutive epochs.

    Args:
        patience: Number of epochs with no improvement before stopping.

    Requirements: 10.1
    """

    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self._best: float = float("-inf")
        self._counter: int = 0

    @property
    def best(self) -> float:
        """Best metric value seen so far."""
        return self._best

    @property
    def counter(self) -> int:
        """Number of consecutive epochs without improvement."""
        return self._counter

    def step(self, metric: float) -> bool:
        """Update the monitor with the latest metric value.

        Args:
            metric: The current epoch's validation metric (higher is better).

        Returns:
            ``True`` if training should stop (patience exhausted),
            ``False`` otherwise.
        """
        if metric > self._best:
            self._best = metric
            self._counter = 0
            return False
        else:
            self._counter += 1
            return self._counter >= self.patience


# ---------------------------------------------------------------------------
# SentimentClassifier
# ---------------------------------------------------------------------------


class SentimentClassifier:
    """Heuristic stub for the MuRIL-based 3-class sentiment classifier.

    Public interface mirrors the production MuRIL fine-tuned model so that
    all downstream code and tests work without GPU/model downloads.

    Requirements: 10.1, 10.2, 10.5, 17.1, 17.3
    """

    def __init__(
        self,
        positive_keywords: frozenset[str] | None = None,
        negative_keywords: frozenset[str] | None = None,
    ) -> None:
        """Initialise the classifier.

        Args:
            positive_keywords: Custom set of positive-sentiment keywords.
                Defaults to the built-in ``_POSITIVE_KEYWORDS``.
            negative_keywords: Custom set of negative-sentiment keywords.
                Defaults to the built-in ``_NEGATIVE_KEYWORDS``.
        """
        self._pos_kw = positive_keywords if positive_keywords is not None else _POSITIVE_KEYWORDS
        self._neg_kw = negative_keywords if negative_keywords is not None else _NEGATIVE_KEYWORDS

        # State set during training
        self._best_f1: float = 0.0
        self._best_epoch: int = 0
        self._class_weights: dict[str, float] = {cls: 1.0 for cls in SENTIMENT_CLASSES}
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sentence: str) -> SentimentPrediction:
        """Predict the sentiment of *sentence*.

        Uses keyword heuristics:
        - If any positive keyword is found → "positive"
        - If any negative keyword is found → "negative"
        - Otherwise → "neutral"
        When both positive and negative keywords are present, the class with
        more keyword matches wins; ties go to "neutral".

        Args:
            sentence: Input sentence string.

        Returns:
            A :class:`~models.data_models.SentimentPrediction` with the
            predicted label, confidence, and per-class scores.

        Requirements: 10.2
        """
        tokens = set(sentence.lower().split())
        # Also check original-case tokens for Devanagari
        tokens_orig = set(sentence.split())
        all_tokens = tokens | tokens_orig

        pos_hits = sum(1 for kw in self._pos_kw if kw in all_tokens)
        neg_hits = sum(1 for kw in self._neg_kw if kw in all_tokens)

        total_hits = pos_hits + neg_hits

        if total_hits == 0:
            label: Literal["positive", "neutral", "negative"] = "neutral"
            scores = {"positive": 0.2, "neutral": 0.6, "negative": 0.2}
        elif pos_hits > neg_hits:
            label = "positive"
            pos_score = 0.5 + 0.4 * (pos_hits / (total_hits + 1))
            scores = {
                "positive": pos_score,
                "neutral": (1.0 - pos_score) / 2,
                "negative": (1.0 - pos_score) / 2,
            }
        elif neg_hits > pos_hits:
            label = "negative"
            neg_score = 0.5 + 0.4 * (neg_hits / (total_hits + 1))
            scores = {
                "positive": (1.0 - neg_score) / 2,
                "neutral": (1.0 - neg_score) / 2,
                "negative": neg_score,
            }
        else:
            # Tie → neutral
            label = "neutral"
            scores = {"positive": 0.25, "neutral": 0.5, "negative": 0.25}

        confidence = scores[label]
        return SentimentPrediction(
            label=label,
            confidence=confidence,
            per_class_scores=scores,
        )

    # ------------------------------------------------------------------
    # Training simulation
    # ------------------------------------------------------------------

    def train(
        self,
        train_set: list[AnnotatedSentence],
        val_set: list[AnnotatedSentence],
        seed: int,
        max_epochs: int = 10,
        patience: int = 3,
        lr: float = 2e-5,
        batch_size: int = 32,
    ) -> TrainingLog:
        """Simulate training the sentiment classifier.

        Steps:
        1. Call ``set_all_seeds(seed)`` to fix all random number generators.
        2. Compute class weights from training labels.
        3. Iterate up to ``max_epochs`` epochs:
           a. Compute mock validation macro-F1 (using heuristic predict on val_set).
           b. Check early stopping (patience=3).
           c. Save best checkpoint (track best_f1 and best_epoch).
        4. Return a :class:`~models.data_models.TrainingLog`.

        Args:
            train_set: Training partition of annotated sentences.
            val_set: Validation partition of annotated sentences.
            seed: Random seed for reproducibility.
            max_epochs: Maximum number of training epochs (default 10).
            patience: Early stopping patience (default 3).
            lr: Learning rate (logged; not used in stub).
            batch_size: Batch size (logged; not used in stub).

        Returns:
            A :class:`~models.data_models.TrainingLog` with training details.

        Requirements: 10.1, 17.1, 17.3
        """
        # Step 1: Fix all random seeds
        set_all_seeds(seed)
        logger.info(
            "SentimentClassifier.train: seed=%d, max_epochs=%d, patience=%d, "
            "lr=%g, batch_size=%d, train_size=%d, val_size=%d",
            seed, max_epochs, patience, lr, batch_size,
            len(train_set), len(val_set),
        )

        # Step 2: Compute class weights from training labels
        train_labels = [s.sentiment for s in train_set]
        self._class_weights = compute_class_weights(train_labels)
        logger.info("Class weights: %s", self._class_weights)

        # Step 3: Simulate training epochs
        early_stopper = EarlyStopping(patience=patience)
        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(max_epochs):
            # Compute real validation macro-F1 using heuristic predict
            val_f1 = self._compute_val_f1(val_set)
            logger.info("Epoch %d/%d — val macro-F1: %.4f", epoch + 1, max_epochs, val_f1)

            # Track best checkpoint
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                logger.info("New best checkpoint at epoch %d (F1=%.4f)", epoch + 1, val_f1)

            # Check early stopping
            if early_stopper.step(val_f1):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d exhausted)",
                    epoch + 1, patience,
                )
                total_epochs = epoch + 1
                break
        else:
            total_epochs = max_epochs

        self._best_f1 = best_f1
        self._best_epoch = best_epoch
        self._trained = True

        return TrainingLog(
            best_epoch=best_epoch,
            best_f1=best_f1,
            total_epochs_run=total_epochs,
            seed=seed,
            class_weights=self._class_weights,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_set: list[AnnotatedSentence]) -> ClassificationMetrics:
        """Evaluate the classifier on *test_set*.

        Computes macro-F1 and per-class precision, recall, and F1 by
        comparing heuristic predictions against gold sentiment labels.

        Args:
            test_set: List of annotated sentences with gold ``sentiment`` labels.

        Returns:
            A :class:`~models.data_models.ClassificationMetrics` instance.

        Requirements: 10.5
        """
        if not test_set:
            logger.warning("evaluate() called with empty test set; returning zero metrics")
            return ClassificationMetrics(
                macro_f1=0.0,
                per_class_precision={cls: 0.0 for cls in SENTIMENT_CLASSES},
                per_class_recall={cls: 0.0 for cls in SENTIMENT_CLASSES},
                per_class_f1={cls: 0.0 for cls in SENTIMENT_CLASSES},
            )

        gold_labels = [s.sentiment for s in test_set]
        pred_labels = [self.predict(s.text).label for s in test_set]

        metrics = _compute_classification_metrics(gold_labels, pred_labels)
        logger.info(
            "SentimentClassifier.evaluate: macro_f1=%.4f, per_class_f1=%s",
            metrics.macro_f1, metrics.per_class_f1,
        )
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_val_f1(self, val_set: list[AnnotatedSentence]) -> float:
        """Compute macro-F1 on the validation set using heuristic predict."""
        if not val_set:
            return 0.0
        metrics = self.evaluate(val_set)
        return metrics.macro_f1


# ---------------------------------------------------------------------------
# SentimentClassifierWithLangID (Task 14.2)
# ---------------------------------------------------------------------------


class SentimentClassifierWithLangID(SentimentClassifier):
    """Sentiment classifier that incorporates Language_ID_Tagger features.

    Integration strategy: prepend language-ID tag tokens as special tokens
    to the input sentence before prediction.  For example:
        "[RAJ] म्हारो [HIN] देश [ENG] India [TRL] mharo"

    This gives the classifier explicit language-boundary information.

    Requirements: 9.3
    """

    def __init__(
        self,
        lang_id_tagger=None,
        positive_keywords: frozenset[str] | None = None,
        negative_keywords: frozenset[str] | None = None,
    ) -> None:
        """Initialise with an optional Language_ID_Tagger.

        Args:
            lang_id_tagger: A :class:`~language_id.tagger.LanguageIDTagger`
                instance.  If ``None``, a default tagger is created.
            positive_keywords: Custom positive keyword set.
            negative_keywords: Custom negative keyword set.
        """
        super().__init__(
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords,
        )
        if lang_id_tagger is None:
            from language_id.tagger import LanguageIDTagger
            lang_id_tagger = LanguageIDTagger()
        self._lang_id_tagger = lang_id_tagger

    def _augment_sentence(self, sentence: str) -> str:
        """Prepend language-ID special tokens to each token in *sentence*.

        Example: "म्हारो India" → "[RAJ] म्हारो [ENG] India"
        """
        token_labels = self._lang_id_tagger.tag(sentence)
        augmented_tokens = []
        for tl in token_labels:
            augmented_tokens.append(f"[{tl.label}]")
            augmented_tokens.append(tl.token)
        return " ".join(augmented_tokens)

    def predict(self, sentence: str) -> SentimentPrediction:
        """Predict sentiment using language-ID-augmented sentence."""
        augmented = self._augment_sentence(sentence)
        return super().predict(augmented)


# ---------------------------------------------------------------------------
# Ablation helper (Task 14.2)
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Results from the Language_ID ablation study."""

    no_langid_f1: float
    """Macro-F1 of the baseline model (no Language_ID features)."""

    with_langid_f1: float
    """Macro-F1 of the model with Language_ID features."""

    improvement: float
    """Absolute macro-F1 improvement (with_langid_f1 - no_langid_f1)."""

    meets_requirement: bool
    """True if improvement >= 0.04 (Requirement 9.3)."""


def run_langid_ablation(
    train_set: list[AnnotatedSentence],
    val_set: list[AnnotatedSentence],
    test_set: list[AnnotatedSentence],
    seed: int = 42,
    lang_id_tagger=None,
) -> AblationResult:
    """Train and evaluate both ablation variants.

    Trains:
    1. ``SentimentClassifier`` (no Language_ID features)
    2. ``SentimentClassifierWithLangID`` (with Language_ID features)

    Evaluates both on *test_set* and verifies the ≥ +0.04 macro-F1
    improvement requirement (Requirement 9.3).

    Args:
        train_set: Training partition.
        val_set: Validation partition.
        test_set: Test partition.
        seed: Random seed for both training runs.
        lang_id_tagger: Optional pre-built Language_ID_Tagger.

    Returns:
        An :class:`AblationResult` with both F1 scores and the improvement.
    """
    # Baseline: no LangID
    baseline = SentimentClassifier()
    baseline.train(train_set, val_set, seed=seed)
    baseline_metrics = baseline.evaluate(test_set)

    # With LangID
    with_langid = SentimentClassifierWithLangID(lang_id_tagger=lang_id_tagger)
    with_langid.train(train_set, val_set, seed=seed)
    langid_metrics = with_langid.evaluate(test_set)

    improvement = langid_metrics.macro_f1 - baseline_metrics.macro_f1
    meets = improvement >= 0.04

    logger.info(
        "LangID ablation: no_langid_f1=%.4f, with_langid_f1=%.4f, "
        "improvement=%.4f, meets_requirement=%s",
        baseline_metrics.macro_f1, langid_metrics.macro_f1, improvement, meets,
    )

    return AblationResult(
        no_langid_f1=baseline_metrics.macro_f1,
        with_langid_f1=langid_metrics.macro_f1,
        improvement=improvement,
        meets_requirement=meets,
    )


# ---------------------------------------------------------------------------
# Shared metric computation
# ---------------------------------------------------------------------------


def _compute_classification_metrics(
    gold: list[str],
    pred: list[str],
) -> ClassificationMetrics:
    """Compute precision, recall, F1 per class and macro-F1.

    Uses the standard TP/FP/FN formula.  Returns 0.0 for any class with
    no gold or predicted instances.

    Args:
        gold: Gold label list.
        pred: Predicted label list (same length as *gold*).

    Returns:
        A :class:`~models.data_models.ClassificationMetrics` instance.
    """
    from collections import defaultdict

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}

    for cls in SENTIMENT_CLASSES:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_precision[cls] = prec
        per_class_recall[cls] = rec
        per_class_f1[cls] = f1

    macro_f1 = sum(per_class_f1.values()) / len(SENTIMENT_CLASSES)

    return ClassificationMetrics(
        macro_f1=macro_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
    )
