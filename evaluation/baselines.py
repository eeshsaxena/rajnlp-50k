"""
Baseline evaluators for the RajNLP-50K evaluation pipeline.

Implements zero-shot mBERT, zero-shot MuRIL, and GPT-4o 5-shot baseline
evaluators as **stub evaluators** with the same interface as real models.
The stubs return fixed plausible F1 scores without requiring GPU/model downloads.

Stub F1 scores:
- mBERT zero-shot:  sentiment=0.45, NER=0.38, toxicity=0.32
- MuRIL zero-shot:  sentiment=0.52, NER=0.44, toxicity=0.38
- GPT-4o 5-shot:    sentiment=0.62, NER=0.58, toxicity=0.51

Requirements: 13.1, 13.2, 13.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

from models.data_models import (
    AnnotatedSentence,
    ClassificationMetrics,
    MultiLabelMetrics,
    NERMetrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENTIMENT_CLASSES = ("positive", "neutral", "negative")
_ENTITY_TYPES = ("PER", "LOC", "ORG")
_TOXICITY_CATEGORIES = ("caste_slur", "religious", "gender", "general")


# ---------------------------------------------------------------------------
# BaselineResult
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """Result from a single baseline evaluator on a single task.

    Attributes:
        model_name: Identifier for the baseline model (e.g., "mBERT-zero-shot").
        task: Task name — one of "sentiment", "ner", "toxicity".
        macro_f1: Macro-averaged F1 score for this model/task combination.
        metrics: Full metrics object (ClassificationMetrics, NERMetrics, or
            MultiLabelMetrics depending on the task).
    """

    model_name: str
    task: str
    macro_f1: float
    metrics: Union[ClassificationMetrics, NERMetrics, MultiLabelMetrics]


# ---------------------------------------------------------------------------
# Helper: build stub metrics objects
# ---------------------------------------------------------------------------


def _make_classification_metrics(macro_f1: float) -> ClassificationMetrics:
    """Build a ClassificationMetrics stub with the given macro_f1.

    Per-class values are set to macro_f1 uniformly so that the macro average
    equals the requested value.
    """
    return ClassificationMetrics(
        macro_f1=macro_f1,
        per_class_precision={cls: macro_f1 for cls in _SENTIMENT_CLASSES},
        per_class_recall={cls: macro_f1 for cls in _SENTIMENT_CLASSES},
        per_class_f1={cls: macro_f1 for cls in _SENTIMENT_CLASSES},
    )


def _make_ner_metrics(macro_f1: float) -> NERMetrics:
    """Build a NERMetrics stub with the given macro_f1."""
    return NERMetrics(
        macro_f1=macro_f1,
        per_type_precision={etype: macro_f1 for etype in _ENTITY_TYPES},
        per_type_recall={etype: macro_f1 for etype in _ENTITY_TYPES},
        per_type_f1={etype: macro_f1 for etype in _ENTITY_TYPES},
    )


def _make_multilabel_metrics(macro_f1: float) -> MultiLabelMetrics:
    """Build a MultiLabelMetrics stub with the given macro_f1."""
    return MultiLabelMetrics(
        macro_f1=macro_f1,
        per_category_precision={cat: macro_f1 for cat in _TOXICITY_CATEGORIES},
        per_category_recall={cat: macro_f1 for cat in _TOXICITY_CATEGORIES},
        per_category_f1={cat: macro_f1 for cat in _TOXICITY_CATEGORIES},
    )


# ---------------------------------------------------------------------------
# ZeroShotMBERTEvaluator
# ---------------------------------------------------------------------------


class ZeroShotMBERTEvaluator:
    """Zero-shot mBERT baseline evaluator (stub implementation).

    Returns fixed plausible F1 scores without loading any model checkpoint:
    - sentiment: 0.45
    - NER:       0.38
    - toxicity:  0.32

    Requirements: 13.1, 13.2, 13.3
    """

    MODEL_NAME: str = "mBERT-zero-shot"

    # Fixed stub F1 scores
    _SENTIMENT_F1: float = 0.45
    _NER_F1: float = 0.38
    _TOXICITY_F1: float = 0.32

    def evaluate_sentiment(
        self, test_set: list[AnnotatedSentence]
    ) -> ClassificationMetrics:
        """Return zero-shot mBERT sentiment metrics (stub).

        Args:
            test_set: Test partition of annotated sentences (used for logging
                only; stub does not run real inference).

        Returns:
            ClassificationMetrics with macro_f1 = 0.45.
        """
        logger.info(
            "%s.evaluate_sentiment: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._SENTIMENT_F1,
        )
        return _make_classification_metrics(self._SENTIMENT_F1)

    def evaluate_ner(
        self, test_set: list[AnnotatedSentence]
    ) -> NERMetrics:
        """Return zero-shot mBERT NER metrics (stub).

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            NERMetrics with macro_f1 = 0.38.
        """
        logger.info(
            "%s.evaluate_ner: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._NER_F1,
        )
        return _make_ner_metrics(self._NER_F1)

    def evaluate_toxicity(
        self, test_set: list[AnnotatedSentence]
    ) -> MultiLabelMetrics:
        """Return zero-shot mBERT toxicity metrics (stub).

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            MultiLabelMetrics with macro_f1 = 0.32.
        """
        logger.info(
            "%s.evaluate_toxicity: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._TOXICITY_F1,
        )
        return _make_multilabel_metrics(self._TOXICITY_F1)

    def run_all(
        self, test_set: list[AnnotatedSentence]
    ) -> list[BaselineResult]:
        """Run all three task evaluations and return a list of BaselineResult.

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            List of three BaselineResult objects (sentiment, ner, toxicity).
        """
        sentiment_metrics = self.evaluate_sentiment(test_set)
        ner_metrics = self.evaluate_ner(test_set)
        toxicity_metrics = self.evaluate_toxicity(test_set)

        return [
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="sentiment",
                macro_f1=sentiment_metrics.macro_f1,
                metrics=sentiment_metrics,
            ),
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="ner",
                macro_f1=ner_metrics.macro_f1,
                metrics=ner_metrics,
            ),
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="toxicity",
                macro_f1=toxicity_metrics.macro_f1,
                metrics=toxicity_metrics,
            ),
        ]


# ---------------------------------------------------------------------------
# ZeroShotMuRILEvaluator
# ---------------------------------------------------------------------------


class ZeroShotMuRILEvaluator:
    """Zero-shot MuRIL baseline evaluator (stub implementation).

    Returns fixed plausible F1 scores without loading any model checkpoint:
    - sentiment: 0.52
    - NER:       0.44
    - toxicity:  0.38

    Requirements: 13.1, 13.2, 13.3
    """

    MODEL_NAME: str = "MuRIL-zero-shot"

    # Fixed stub F1 scores
    _SENTIMENT_F1: float = 0.52
    _NER_F1: float = 0.44
    _TOXICITY_F1: float = 0.38

    def evaluate_sentiment(
        self, test_set: list[AnnotatedSentence]
    ) -> ClassificationMetrics:
        """Return zero-shot MuRIL sentiment metrics (stub).

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            ClassificationMetrics with macro_f1 = 0.52.
        """
        logger.info(
            "%s.evaluate_sentiment: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._SENTIMENT_F1,
        )
        return _make_classification_metrics(self._SENTIMENT_F1)

    def evaluate_ner(
        self, test_set: list[AnnotatedSentence]
    ) -> NERMetrics:
        """Return zero-shot MuRIL NER metrics (stub).

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            NERMetrics with macro_f1 = 0.44.
        """
        logger.info(
            "%s.evaluate_ner: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._NER_F1,
        )
        return _make_ner_metrics(self._NER_F1)

    def evaluate_toxicity(
        self, test_set: list[AnnotatedSentence]
    ) -> MultiLabelMetrics:
        """Return zero-shot MuRIL toxicity metrics (stub).

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            MultiLabelMetrics with macro_f1 = 0.38.
        """
        logger.info(
            "%s.evaluate_toxicity: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._TOXICITY_F1,
        )
        return _make_multilabel_metrics(self._TOXICITY_F1)

    def run_all(
        self, test_set: list[AnnotatedSentence]
    ) -> list[BaselineResult]:
        """Run all three task evaluations and return a list of BaselineResult.

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            List of three BaselineResult objects (sentiment, ner, toxicity).
        """
        sentiment_metrics = self.evaluate_sentiment(test_set)
        ner_metrics = self.evaluate_ner(test_set)
        toxicity_metrics = self.evaluate_toxicity(test_set)

        return [
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="sentiment",
                macro_f1=sentiment_metrics.macro_f1,
                metrics=sentiment_metrics,
            ),
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="ner",
                macro_f1=ner_metrics.macro_f1,
                metrics=ner_metrics,
            ),
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="toxicity",
                macro_f1=toxicity_metrics.macro_f1,
                metrics=toxicity_metrics,
            ),
        ]


# ---------------------------------------------------------------------------
# GPT4o5ShotEvaluator
# ---------------------------------------------------------------------------


class GPT4o5ShotEvaluator:
    """GPT-4o 5-shot baseline evaluator (stub implementation).

    Constructs 5-shot prompts for each task and returns fixed plausible F1
    scores without making real API calls:
    - sentiment: 0.62
    - NER:       0.58
    - toxicity:  0.51

    Requirements: 13.1, 13.2, 13.3
    """

    MODEL_NAME: str = "GPT-4o-5-shot"

    # Fixed stub F1 scores
    _SENTIMENT_F1: float = 0.62
    _NER_F1: float = 0.58
    _TOXICITY_F1: float = 0.51

    # Number of few-shot examples to include in each prompt
    _N_SHOTS: int = 5

    # ---------------------------------------------------------------------------
    # Prompt builders
    # ---------------------------------------------------------------------------

    def build_sentiment_prompt(
        self,
        examples: list[AnnotatedSentence],
        sentence: str,
    ) -> str:
        """Construct a 5-shot sentiment classification prompt.

        Args:
            examples: List of annotated sentences to use as few-shot examples.
                Up to ``_N_SHOTS`` examples are used.
            sentence: The target sentence to classify.

        Returns:
            A formatted prompt string with few-shot examples followed by the
            target sentence.
        """
        shots = examples[: self._N_SHOTS]
        lines = [
            "Classify the sentiment of the following Rajasthani-Hindi sentence.",
            "Labels: positive, neutral, negative.",
            "",
        ]
        for i, ex in enumerate(shots, start=1):
            lines.append(f"Example {i}:")
            lines.append(f"  Sentence: {ex.text}")
            lines.append(f"  Sentiment: {ex.sentiment}")
            lines.append("")
        lines.append(f"Sentence: {sentence}")
        lines.append("Sentiment:")
        return "\n".join(lines)

    def build_ner_prompt(
        self,
        examples: list[AnnotatedSentence],
        sentence: str,
    ) -> str:
        """Construct a 5-shot NER prompt.

        Args:
            examples: List of annotated sentences to use as few-shot examples.
            sentence: The target sentence to tag.

        Returns:
            A formatted prompt string with few-shot NER examples.
        """
        shots = examples[: self._N_SHOTS]
        lines = [
            "Identify named entities (PER, LOC, ORG) in the following "
            "Rajasthani-Hindi sentence.",
            "",
        ]
        for i, ex in enumerate(shots, start=1):
            spans_str = ", ".join(
                f"{s.text} ({s.entity_type})" for s in ex.ner_spans
            ) or "none"
            lines.append(f"Example {i}:")
            lines.append(f"  Sentence: {ex.text}")
            lines.append(f"  Entities: {spans_str}")
            lines.append("")
        lines.append(f"Sentence: {sentence}")
        lines.append("Entities:")
        return "\n".join(lines)

    def build_toxicity_prompt(
        self,
        examples: list[AnnotatedSentence],
        sentence: str,
    ) -> str:
        """Construct a 5-shot toxicity classification prompt.

        Args:
            examples: List of annotated sentences to use as few-shot examples.
            sentence: The target sentence to classify.

        Returns:
            A formatted prompt string with few-shot toxicity examples.
        """
        shots = examples[: self._N_SHOTS]
        lines = [
            "Classify the toxicity of the following Rajasthani-Hindi sentence.",
            "Categories (select all that apply): caste_slur, religious, gender, general.",
            "If none apply, output: none.",
            "",
        ]
        for i, ex in enumerate(shots, start=1):
            labels_str = ", ".join(ex.toxicity_labels) or "none"
            lines.append(f"Example {i}:")
            lines.append(f"  Sentence: {ex.text}")
            lines.append(f"  Toxicity: {labels_str}")
            lines.append("")
        lines.append(f"Sentence: {sentence}")
        lines.append("Toxicity:")
        return "\n".join(lines)

    # ---------------------------------------------------------------------------
    # Evaluation methods (stub — return fixed F1 scores)
    # ---------------------------------------------------------------------------

    def evaluate_sentiment(
        self, test_set: list[AnnotatedSentence]
    ) -> ClassificationMetrics:
        """Return GPT-4o 5-shot sentiment metrics (stub).

        In a real implementation this would:
        1. Select 5 few-shot examples from the training set.
        2. Build a prompt for each test sentence using build_sentiment_prompt().
        3. Call the GPT-4o API and parse the response.
        4. Compute macro-averaged F1 against gold labels.

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            ClassificationMetrics with macro_f1 = 0.62.
        """
        logger.info(
            "%s.evaluate_sentiment: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._SENTIMENT_F1,
        )
        return _make_classification_metrics(self._SENTIMENT_F1)

    def evaluate_ner(
        self, test_set: list[AnnotatedSentence]
    ) -> NERMetrics:
        """Return GPT-4o 5-shot NER metrics (stub).

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            NERMetrics with macro_f1 = 0.58.
        """
        logger.info(
            "%s.evaluate_ner: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._NER_F1,
        )
        return _make_ner_metrics(self._NER_F1)

    def evaluate_toxicity(
        self, test_set: list[AnnotatedSentence]
    ) -> MultiLabelMetrics:
        """Return GPT-4o 5-shot toxicity metrics (stub).

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            MultiLabelMetrics with macro_f1 = 0.51.
        """
        logger.info(
            "%s.evaluate_toxicity: test_size=%d, macro_f1=%.2f (stub)",
            self.MODEL_NAME, len(test_set), self._TOXICITY_F1,
        )
        return _make_multilabel_metrics(self._TOXICITY_F1)

    def run_all(
        self, test_set: list[AnnotatedSentence]
    ) -> list[BaselineResult]:
        """Run all three task evaluations and return a list of BaselineResult.

        Args:
            test_set: Test partition of annotated sentences.

        Returns:
            List of three BaselineResult objects (sentiment, ner, toxicity).
        """
        sentiment_metrics = self.evaluate_sentiment(test_set)
        ner_metrics = self.evaluate_ner(test_set)
        toxicity_metrics = self.evaluate_toxicity(test_set)

        return [
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="sentiment",
                macro_f1=sentiment_metrics.macro_f1,
                metrics=sentiment_metrics,
            ),
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="ner",
                macro_f1=ner_metrics.macro_f1,
                metrics=ner_metrics,
            ),
            BaselineResult(
                model_name=self.MODEL_NAME,
                task="toxicity",
                macro_f1=toxicity_metrics.macro_f1,
                metrics=toxicity_metrics,
            ),
        ]
