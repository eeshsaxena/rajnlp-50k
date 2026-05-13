"""
Toxicity_Classifier — multi-label toxicity detection model.

Architecture (production): MuRIL + multi-label classification head
  (sigmoid output, 4 binary classifiers: caste_slur, religious, gender, general).
  Binary cross-entropy loss with per-class weights inversely proportional to
  class frequency.

For testability without GPU/model downloads, this module implements a
**keyword-based heuristic classifier** with the same public interface as the
real MuRIL-based classifier.

Heuristic logic
---------------
- ``predict(sentence)`` checks for keyword matches in each of the 4 toxicity
  categories and returns multi-label predictions.
- ``train(train_set, val_set, seed)`` simulates training:
    * calls ``set_all_seeds(seed)``
    * computes per-class weights from training labels
    * runs up to 5 epochs, then checks oversampling fallback condition
    * if any per-category F1 < 0.60 after 5 epochs, activates oversampling
    * continues training up to max_epochs total
    * tracks best macro-F1 and best_epoch
- ``evaluate(test_set)`` computes real metrics against gold labels using
  the heuristic ``predict``.

Requirements: 12.1, 12.2, 12.3, 12.6, 17.1, 17.3
"""

from __future__ import annotations

import logging
from collections import defaultdict

from models.data_models import (
    AnnotatedSentence,
    MultiLabelMetrics,
    ToxicityPrediction,
    TrainingLog,
)
from models.reproducibility import set_all_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Toxicity categories
# ---------------------------------------------------------------------------

TOXICITY_CATEGORIES: tuple[str, ...] = (
    "caste_slur",
    "religious",
    "gender",
    "general",
)

# ---------------------------------------------------------------------------
# Keyword lexicons (placeholder terms for testing purposes only)
# ---------------------------------------------------------------------------

_CASTE_SLUR_KEYWORDS: frozenset[str] = frozenset({
    "chamar",
    "bhangi",
    "neech",
    "jaat",
    "dalit_slur",
    "chamari",
    "bhangin",
    "neechi",
})

_RELIGIOUS_KEYWORDS: frozenset[str] = frozenset({
    "kafir",
    "jihad",
    "mandir_tod",
    "gau_hatya",
    "dharm_virodhi",
    "kafira",
    "jihadis",
    "dharm_nashak",
})

_GENDER_KEYWORDS: frozenset[str] = frozenset({
    "randi",
    "besharmi",
    "aurat_ki_aukat",
    "ladki_bhaag",
    "mahila_virodhi",
    "randwa",
    "besharam",
    "aurat_nahi",
})

_GENERAL_KEYWORDS: frozenset[str] = frozenset({
    "gali",
    "bakwaas",
    "chutiya",
    "harami",
    "saala",
    "gandu",
    "kamina",
    "kutta",
})

_CATEGORY_KEYWORDS: dict[str, frozenset[str]] = {
    "caste_slur": _CASTE_SLUR_KEYWORDS,
    "religious": _RELIGIOUS_KEYWORDS,
    "gender": _GENDER_KEYWORDS,
    "general": _GENERAL_KEYWORDS,
}

# Sigmoid threshold for predicting a positive label
_PREDICT_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Class weight computation for multi-label
# ---------------------------------------------------------------------------


def compute_toxicity_class_weights(
    sentences: list[AnnotatedSentence],
) -> dict[str, float]:
    """Compute per-category class weights inversely proportional to class frequency.

    For each category:
        weight = total / (2 * count_positive)

    If count_positive == 0, weight = 1.0 (neutral default).

    This ensures minority categories receive higher loss penalties during
    training, counteracting class imbalance.

    Args:
        sentences: List of annotated sentences with gold ``toxicity_labels``.

    Returns:
        Dictionary mapping each toxicity category to its weight.

    Requirements: 12.3
    """
    if not sentences:
        return {cat: 1.0 for cat in TOXICITY_CATEGORIES}

    total = len(sentences)
    weights: dict[str, float] = {}

    for cat in TOXICITY_CATEGORIES:
        count_positive = sum(1 for s in sentences if cat in s.toxicity_labels)
        if count_positive == 0:
            weights[cat] = 1.0
        else:
            weights[cat] = total / (2 * count_positive)

    return weights


# ---------------------------------------------------------------------------
# Oversampling fallback logic
# ---------------------------------------------------------------------------


def should_oversample(
    val_metrics: MultiLabelMetrics,
    threshold: float = 0.60,
) -> bool:
    """Return True if any per-category F1 is below *threshold*.

    Used to decide whether to activate random oversampling of minority-class
    examples after 5 training epochs.

    Args:
        val_metrics: Validation metrics from the current training run.
        threshold: Minimum acceptable per-category F1 (default 0.60).

    Returns:
        ``True`` if oversampling should be activated, ``False`` otherwise.

    Requirements: 12.3
    """
    return any(f1 < threshold for f1 in val_metrics.per_category_f1.values())


# ---------------------------------------------------------------------------
# ToxicityClassifier
# ---------------------------------------------------------------------------


class ToxicityClassifier:
    """Heuristic stub for the MuRIL-based multi-label toxicity classifier.

    Public interface mirrors the production MuRIL fine-tuned model so that
    all downstream code and tests work without GPU/model downloads.

    Requirements: 12.1, 12.2, 12.3, 12.6, 17.1, 17.3
    """

    def __init__(
        self,
        category_keywords: dict[str, frozenset[str]] | None = None,
        predict_threshold: float = _PREDICT_THRESHOLD,
    ) -> None:
        """Initialise the classifier.

        Args:
            category_keywords: Custom keyword lexicons per category.
                Defaults to the built-in lexicons.
            predict_threshold: Sigmoid score threshold above which a category
                is predicted as positive (default 0.5).
        """
        self._keywords = (
            category_keywords
            if category_keywords is not None
            else _CATEGORY_KEYWORDS
        )
        self._threshold = predict_threshold

        # State set during training
        self._best_f1: float = 0.0
        self._best_epoch: int = 0
        self._class_weights: dict[str, float] = {cat: 1.0 for cat in TOXICITY_CATEGORIES}
        self._trained: bool = False
        self._oversampling_active: bool = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sentence: str) -> ToxicityPrediction:
        """Predict toxicity labels for *sentence*.

        Uses keyword heuristics: for each category, compute a score based on
        the fraction of category keywords found in the sentence.  If the score
        exceeds the threshold, the category is predicted as positive.

        Args:
            sentence: Input sentence string.

        Returns:
            A :class:`~models.data_models.ToxicityPrediction` with the
            predicted labels and per-category sigmoid scores.

        Requirements: 12.2
        """
        tokens = set(sentence.lower().split())
        # Also check original-case tokens for Devanagari / mixed scripts
        tokens_orig = set(sentence.split())
        all_tokens = tokens | tokens_orig

        per_category_scores: dict[str, float] = {}
        predicted_labels: list[str] = []

        for cat in TOXICITY_CATEGORIES:
            kw_set = self._keywords.get(cat, frozenset())
            if not kw_set:
                score = 0.0
            else:
                hits = sum(1 for kw in kw_set if kw in all_tokens)
                # Sigmoid-like score: scale hits to [0, 1]
                # More hits → higher score, capped at 1.0
                score = min(1.0, hits / max(1, len(kw_set) * 0.2))

            per_category_scores[cat] = score
            if score >= self._threshold:
                predicted_labels.append(cat)

        return ToxicityPrediction(
            labels=predicted_labels,
            per_category_scores=per_category_scores,
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
        oversample_check_epoch: int = 5,
    ) -> TrainingLog:
        """Simulate training the toxicity classifier.

        Steps:
        1. Call ``set_all_seeds(seed)`` to fix all random number generators.
        2. Compute per-class weights from training labels.
        3. Run up to ``oversample_check_epoch`` epochs.
        4. After epoch ``oversample_check_epoch``, check if oversampling is needed.
           If ``should_oversample()`` returns True, log the fallback and set
           ``self._oversampling_active = True``.
        5. Continue training up to ``max_epochs`` total.
        6. Track best macro-F1 and best_epoch.
        7. Return a :class:`~models.data_models.TrainingLog`.

        Args:
            train_set: Training partition of annotated sentences.
            val_set: Validation partition of annotated sentences.
            seed: Random seed for reproducibility.
            max_epochs: Maximum number of training epochs (default 10).
            oversample_check_epoch: Epoch after which to check oversampling
                fallback condition (default 5, per design spec).

        Returns:
            A :class:`~models.data_models.TrainingLog` with training details.

        Requirements: 12.3, 17.1, 17.3
        """
        # Step 1: Fix all random seeds
        set_all_seeds(seed)
        logger.info(
            "ToxicityClassifier.train: seed=%d, max_epochs=%d, "
            "train_size=%d, val_size=%d",
            seed, max_epochs, len(train_set), len(val_set),
        )

        # Step 2: Compute per-class weights from training labels
        self._class_weights = compute_toxicity_class_weights(train_set)
        logger.info("Per-class weights: %s", self._class_weights)

        # Reset oversampling flag
        self._oversampling_active = False

        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(max_epochs):
            # Compute real validation metrics using heuristic predict
            val_metrics = self._compute_val_metrics(val_set)
            val_macro_f1 = val_metrics.macro_f1

            logger.info(
                "Epoch %d/%d — val macro-F1: %.4f, per-category F1: %s",
                epoch + 1, max_epochs, val_macro_f1, val_metrics.per_category_f1,
            )

            # Track best checkpoint
            if val_macro_f1 > best_f1:
                best_f1 = val_macro_f1
                best_epoch = epoch
                logger.info(
                    "New best checkpoint at epoch %d (macro-F1=%.4f)",
                    epoch + 1, val_macro_f1,
                )

            # Step 4: Check oversampling fallback after oversample_check_epoch
            if epoch + 1 == oversample_check_epoch and not self._oversampling_active:
                if should_oversample(val_metrics):
                    logger.warning(
                        "Oversampling fallback triggered at epoch %d: "
                        "per-category F1 below 0.60 threshold. "
                        "Activating random oversampling (minority:majority = 1:3).",
                        epoch + 1,
                    )
                    self._oversampling_active = True
                else:
                    logger.info(
                        "Oversampling not needed at epoch %d: "
                        "all per-category F1 >= 0.60.",
                        epoch + 1,
                    )

        self._best_f1 = best_f1
        self._best_epoch = best_epoch
        self._trained = True

        return TrainingLog(
            best_epoch=best_epoch,
            best_f1=best_f1,
            total_epochs_run=max_epochs,
            seed=seed,
            class_weights=self._class_weights,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_set: list[AnnotatedSentence]) -> MultiLabelMetrics:
        """Evaluate the classifier on *test_set*.

        Computes macro-F1 and per-category precision, recall, and F1 by
        comparing heuristic predictions against gold toxicity labels.

        Args:
            test_set: List of annotated sentences with gold ``toxicity_labels``.

        Returns:
            A :class:`~models.data_models.MultiLabelMetrics` instance.

        Requirements: 12.6
        """
        if not test_set:
            logger.warning("evaluate() called with empty test set; returning zero metrics")
            return MultiLabelMetrics(
                macro_f1=0.0,
                per_category_precision={cat: 0.0 for cat in TOXICITY_CATEGORIES},
                per_category_recall={cat: 0.0 for cat in TOXICITY_CATEGORIES},
                per_category_f1={cat: 0.0 for cat in TOXICITY_CATEGORIES},
            )

        metrics = _compute_multilabel_metrics(test_set, self.predict)
        logger.info(
            "ToxicityClassifier.evaluate: macro_f1=%.4f, per_category_f1=%s",
            metrics.macro_f1, metrics.per_category_f1,
        )
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_val_metrics(self, val_set: list[AnnotatedSentence]) -> MultiLabelMetrics:
        """Compute multi-label metrics on the validation set."""
        if not val_set:
            return MultiLabelMetrics(
                macro_f1=0.0,
                per_category_precision={cat: 0.0 for cat in TOXICITY_CATEGORIES},
                per_category_recall={cat: 0.0 for cat in TOXICITY_CATEGORIES},
                per_category_f1={cat: 0.0 for cat in TOXICITY_CATEGORIES},
            )
        return _compute_multilabel_metrics(val_set, self.predict)


# ---------------------------------------------------------------------------
# Shared metric computation
# ---------------------------------------------------------------------------


def _compute_multilabel_metrics(
    dataset: list[AnnotatedSentence],
    predict_fn,
) -> MultiLabelMetrics:
    """Compute per-category and macro-averaged multi-label metrics.

    For each category, computes binary precision, recall, and F1 by treating
    it as a binary classification problem (positive = category present).

    Args:
        dataset: List of annotated sentences with gold ``toxicity_labels``.
        predict_fn: Callable that takes a sentence string and returns a
            :class:`~models.data_models.ToxicityPrediction`.

    Returns:
        A :class:`~models.data_models.MultiLabelMetrics` instance.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for sentence in dataset:
        gold_labels = set(sentence.toxicity_labels)
        pred = predict_fn(sentence.text)
        pred_labels = set(pred.labels)

        for cat in TOXICITY_CATEGORIES:
            gold_pos = cat in gold_labels
            pred_pos = cat in pred_labels

            if gold_pos and pred_pos:
                tp[cat] += 1
            elif not gold_pos and pred_pos:
                fp[cat] += 1
            elif gold_pos and not pred_pos:
                fn[cat] += 1
            # true negative: neither gold nor predicted → no contribution

    per_category_precision: dict[str, float] = {}
    per_category_recall: dict[str, float] = {}
    per_category_f1: dict[str, float] = {}

    for cat in TOXICITY_CATEGORIES:
        prec = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0.0
        rec = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_category_precision[cat] = prec
        per_category_recall[cat] = rec
        per_category_f1[cat] = f1

    macro_f1 = sum(per_category_f1.values()) / len(TOXICITY_CATEGORIES)

    return MultiLabelMetrics(
        macro_f1=macro_f1,
        per_category_precision=per_category_precision,
        per_category_recall=per_category_recall,
        per_category_f1=per_category_f1,
    )
