"""
Comparison table generator for the RajNLP-50K evaluation pipeline.

Collects results from all baseline and fine-tuned models and produces a
single comparison table with macro-averaged F1 as the primary metric for
all three tasks (sentiment, NER, toxicity).

Required model rows in the table:
- "mBERT-zero-shot"
- "MuRIL-zero-shot"
- "GPT-4o-5-shot"
- "SentimentClassifier-finetuned"
- "NERTagger-finetuned"
- "ToxicityClassifier-finetuned"

Requirements: 13.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from evaluation.baselines import BaselineResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required model names (used for validation)
# ---------------------------------------------------------------------------

REQUIRED_MODEL_NAMES: tuple[str, ...] = (
    "mBERT-zero-shot",
    "MuRIL-zero-shot",
    "GPT-4o-5-shot",
    "SentimentClassifier-finetuned",
    "NERTagger-finetuned",
    "ToxicityClassifier-finetuned",
)

# Sentinel value for missing F1 scores
_MISSING_F1: float = float("nan")


# ---------------------------------------------------------------------------
# ComparisonRow
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRow:
    """A single row in the model comparison table.

    Attributes:
        model_name: Identifier for the model (e.g., "mBERT-zero-shot").
        sentiment_f1: Macro-averaged F1 on the sentiment task.
        ner_f1: Macro-averaged F1 on the NER task.
        toxicity_f1: Macro-averaged F1 on the toxicity task.
    """

    model_name: str
    sentiment_f1: float
    ner_f1: float
    toxicity_f1: float


# ---------------------------------------------------------------------------
# generate_comparison_table
# ---------------------------------------------------------------------------


def generate_comparison_table(results: list[BaselineResult]) -> list[ComparisonRow]:
    """Collect results from all baselines and fine-tuned models and produce a
    comparison table.

    Groups BaselineResult objects by model name and task, then assembles one
    ComparisonRow per model.  Models with missing task results receive
    ``float('nan')`` for those tasks.

    The returned list is ordered according to ``REQUIRED_MODEL_NAMES`` first,
    followed by any additional models found in *results* (in insertion order).

    Args:
        results: List of BaselineResult objects from baseline and fine-tuned
            model evaluations.

    Returns:
        List of ComparisonRow objects, one per unique model name.

    Requirements: 13.4
    """
    # Build a nested dict: model_name → task → macro_f1
    scores: dict[str, dict[str, float]] = {}
    for result in results:
        if result.model_name not in scores:
            scores[result.model_name] = {}
        scores[result.model_name][result.task] = result.macro_f1

    # Determine row order: required models first, then any extras
    ordered_names: list[str] = []
    for name in REQUIRED_MODEL_NAMES:
        ordered_names.append(name)
    for name in scores:
        if name not in ordered_names:
            ordered_names.append(name)

    # Also include required models even if they have no results yet
    for name in REQUIRED_MODEL_NAMES:
        if name not in scores:
            scores[name] = {}

    rows: list[ComparisonRow] = []
    for model_name in ordered_names:
        task_scores = scores.get(model_name, {})
        rows.append(ComparisonRow(
            model_name=model_name,
            sentiment_f1=task_scores.get("sentiment", _MISSING_F1),
            ner_f1=task_scores.get("ner", _MISSING_F1),
            toxicity_f1=task_scores.get("toxicity", _MISSING_F1),
        ))

    logger.info(
        "generate_comparison_table: produced %d rows for models: %s",
        len(rows),
        [r.model_name for r in rows],
    )
    return rows


# ---------------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------------


def format_comparison_table(rows: list[ComparisonRow]) -> str:
    """Format the comparison table as a human-readable string.

    Produces a Markdown-style table with columns:
    Model | Sentiment F1 | NER F1 | Toxicity F1

    Args:
        rows: List of ComparisonRow objects (from generate_comparison_table).

    Returns:
        A formatted string representation of the comparison table.
    """
    if not rows:
        return "(empty comparison table)"

    # Column widths
    col_model = max(len("Model"), max(len(r.model_name) for r in rows))
    col_sentiment = max(len("Sentiment F1"), 12)
    col_ner = max(len("NER F1"), 8)
    col_toxicity = max(len("Toxicity F1"), 11)

    def _fmt_f1(value: float) -> str:
        import math
        if math.isnan(value):
            return "N/A"
        return f"{value:.4f}"

    def _row_str(model: str, sent: str, ner: str, tox: str) -> str:
        return (
            f"| {model:<{col_model}} "
            f"| {sent:>{col_sentiment}} "
            f"| {ner:>{col_ner}} "
            f"| {tox:>{col_toxicity}} |"
        )

    separator = (
        f"|-{'-' * col_model}-"
        f"|-{'-' * col_sentiment}-"
        f"|-{'-' * col_ner}-"
        f"|-{'-' * col_toxicity}-|"
    )

    lines = [
        _row_str("Model", "Sentiment F1", "NER F1", "Toxicity F1"),
        separator,
    ]
    for row in rows:
        lines.append(_row_str(
            row.model_name,
            _fmt_f1(row.sentiment_f1),
            _fmt_f1(row.ner_f1),
            _fmt_f1(row.toxicity_f1),
        ))

    return "\n".join(lines)
