"""
Annotator_Tool — Inter-Annotator Agreement (IAA) computation module.

Computes Cohen's Kappa for sentiment, NER, and toxicity annotation tasks
after Label Studio export.  Batches that fall below task-specific thresholds
are flagged for adjudication review.

Thresholds (from design.md):
  - Sentiment: κ < 0.72 → flag for adjudication  (Requirement 5.4)
  - NER:       κ < 0.78 → flag for adjudication  (Requirement 6.4)
  - Toxicity:  κ < 0.65 → flag for adjudication  (Requirement 7.5)

Requirements: 5.3, 5.4, 6.3, 6.4, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — IAA thresholds
# ---------------------------------------------------------------------------

#: Minimum acceptable Cohen's Kappa for the sentiment task.
SENTIMENT_KAPPA_THRESHOLD: float = 0.72

#: Minimum acceptable Cohen's Kappa for the NER task.
NER_KAPPA_THRESHOLD: float = 0.78

#: Minimum acceptable Cohen's Kappa for the toxicity task.
TOXICITY_KAPPA_THRESHOLD: float = 0.65

#: Mapping from task name to its threshold.
TASK_THRESHOLDS: dict[str, float] = {
    "sentiment": SENTIMENT_KAPPA_THRESHOLD,
    "ner": NER_KAPPA_THRESHOLD,
    "toxicity": TOXICITY_KAPPA_THRESHOLD,
}

TaskName = Literal["sentiment", "ner", "toxicity"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IAAResult:
    """Result of IAA computation for a single batch."""

    batch_id: str
    """Identifier for the annotation batch."""

    task: TaskName
    """Annotation task: 'sentiment', 'ner', or 'toxicity'."""

    kappa: float
    """Computed Cohen's Kappa value."""

    threshold: float
    """Task-specific threshold below which the batch is flagged."""

    flagged: bool
    """True if kappa < threshold (batch requires adjudication)."""


# ---------------------------------------------------------------------------
# Sentiment IAA
# ---------------------------------------------------------------------------


def compute_sentiment_kappa(
    annotator_labels: list[list[str]],
) -> float:
    """Compute Cohen's Kappa for a set of sentiment annotations.

    Computes pairwise Cohen's Kappa across all annotator pairs and returns
    the mean value.

    Args:
        annotator_labels: A list of per-annotator label lists.  Each inner
            list contains one label per sentence.  All inner lists must have
            the same length.  Typically 3 annotators × N sentences.

    Returns:
        Mean pairwise Cohen's Kappa across all annotator pairs.

    Raises:
        ValueError: If fewer than 2 annotators are provided or lists have
            different lengths.
    """
    if len(annotator_labels) < 2:
        raise ValueError(
            f"At least 2 annotators required; got {len(annotator_labels)}"
        )
    lengths = {len(labels) for labels in annotator_labels}
    if len(lengths) > 1:
        raise ValueError(
            f"All annotator label lists must have the same length; got lengths {lengths}"
        )

    n_annotators = len(annotator_labels)
    kappas: list[float] = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            k = cohen_kappa_score(annotator_labels[i], annotator_labels[j])
            kappas.append(k)

    return sum(kappas) / len(kappas)


# ---------------------------------------------------------------------------
# NER span-level IAA
# ---------------------------------------------------------------------------


def _spans_to_binary_vector(
    spans: list[tuple[int, int, str]],
    all_spans: list[tuple[int, int, str]],
) -> list[int]:
    """Convert a list of spans to a binary presence vector over *all_spans*.

    Args:
        spans: The annotator's spans as (start, end, entity_type) tuples.
        all_spans: The universe of all unique spans across all annotators.

    Returns:
        A binary list of length ``len(all_spans)`` where 1 indicates the
        annotator marked that span and 0 indicates they did not.
    """
    span_set = set(spans)
    return [1 if s in span_set else 0 for s in all_spans]


def compute_ner_kappa(
    annotator_spans: list[list[tuple[int, int, str]]],
) -> float:
    """Compute span-level Cohen's Kappa for NER annotations.

    Each annotator's annotation is represented as a list of
    ``(start, end, entity_type)`` tuples.  The universe of all unique spans
    across all annotators is used as the label space.  Each annotator's
    annotation is converted to a binary presence vector over this universe,
    and pairwise Cohen's Kappa is computed over these binary vectors.

    If no spans are present across all annotators (all annotators marked zero
    spans), kappa is defined as 1.0 (perfect agreement on "no entities").

    Args:
        annotator_spans: A list of per-annotator span lists.  Each inner list
            contains ``(start, end, entity_type)`` tuples for one annotator.
            Typically 3 annotators.

    Returns:
        Mean pairwise span-level Cohen's Kappa.

    Raises:
        ValueError: If fewer than 2 annotators are provided.
    """
    if len(annotator_spans) < 2:
        raise ValueError(
            f"At least 2 annotators required; got {len(annotator_spans)}"
        )

    # Collect the universe of all unique spans
    all_spans_set: set[tuple[int, int, str]] = set()
    for spans in annotator_spans:
        all_spans_set.update(spans)

    all_spans = sorted(all_spans_set)

    # If no spans at all, all annotators agree on "no entities" → κ = 1.0
    if not all_spans:
        return 1.0

    # Build binary vectors for each annotator
    vectors = [
        _spans_to_binary_vector(spans, all_spans)
        for spans in annotator_spans
    ]

    n_annotators = len(vectors)
    kappas: list[float] = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            # If both vectors are identical (including all-zero), kappa = 1.0
            if vectors[i] == vectors[j]:
                kappas.append(1.0)
            else:
                k = cohen_kappa_score(vectors[i], vectors[j])
                kappas.append(k)

    return sum(kappas) / len(kappas)


# ---------------------------------------------------------------------------
# Toxicity IAA
# ---------------------------------------------------------------------------


def compute_toxicity_kappa(
    annotator_labels: list[list[list[str]]],
    categories: list[str] | None = None,
) -> float:
    """Compute Cohen's Kappa for multi-label toxicity annotations.

    For each toxicity category, a binary label vector is constructed for each
    annotator (1 if the annotator applied that category, 0 otherwise).
    Pairwise Cohen's Kappa is computed per category, then averaged across
    categories and annotator pairs.

    Args:
        annotator_labels: A list of per-annotator label sets.  Each inner list
            contains the toxicity categories selected by that annotator for
            each sentence.  Shape: [n_annotators][n_sentences] where each
            element is a list of category strings.
        categories: The set of valid toxicity categories.  Defaults to the
            four standard categories.

    Returns:
        Mean pairwise, mean per-category Cohen's Kappa.

    Raises:
        ValueError: If fewer than 2 annotators are provided.
    """
    if categories is None:
        categories = ["caste_slur", "religious", "gender", "general"]

    if len(annotator_labels) < 2:
        raise ValueError(
            f"At least 2 annotators required; got {len(annotator_labels)}"
        )

    n_annotators = len(annotator_labels)
    n_sentences = len(annotator_labels[0])

    # Build per-category binary vectors for each annotator
    # binary_vectors[annotator_idx][category_idx] = list of 0/1 per sentence
    binary_vectors: list[list[list[int]]] = []
    for ann_labels in annotator_labels:
        ann_vectors: list[list[int]] = []
        for cat in categories:
            cat_vector = [1 if cat in sentence_labels else 0 for sentence_labels in ann_labels]
            ann_vectors.append(cat_vector)
        binary_vectors.append(ann_vectors)

    # Compute pairwise kappa per category, then average
    all_kappas: list[float] = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            for cat_idx in range(len(categories)):
                vec_i = binary_vectors[i][cat_idx]
                vec_j = binary_vectors[j][cat_idx]
                if vec_i == vec_j:
                    # Perfect agreement (including all-zero)
                    all_kappas.append(1.0)
                else:
                    k = cohen_kappa_score(vec_i, vec_j)
                    all_kappas.append(k)

    if not all_kappas:
        return 1.0

    return sum(all_kappas) / len(all_kappas)


# ---------------------------------------------------------------------------
# Batch flagging
# ---------------------------------------------------------------------------


def flag_batch(
    batch_id: str,
    task: TaskName,
    kappa: float,
) -> IAAResult:
    """Determine whether a batch should be flagged for adjudication.

    A batch is flagged if its Cohen's Kappa falls below the task-specific
    threshold.  Flagged batches are logged at WARNING level with the batch ID
    and κ value (Requirements 5.4, 6.4, 7.5).

    Args:
        batch_id: Identifier for the annotation batch.
        task: Annotation task name: 'sentiment', 'ner', or 'toxicity'.
        kappa: Computed Cohen's Kappa for the batch.

    Returns:
        An :class:`IAAResult` with the flagging decision.

    Raises:
        ValueError: If *task* is not one of 'sentiment', 'ner', 'toxicity'.
    """
    if task not in TASK_THRESHOLDS:
        raise ValueError(
            f"Unknown task {task!r}; must be one of {list(TASK_THRESHOLDS)}"
        )

    threshold = TASK_THRESHOLDS[task]
    flagged = kappa < threshold

    if flagged:
        logger.warning(
            "Batch %r flagged for adjudication: task=%s, κ=%.4f < threshold=%.2f",
            batch_id,
            task,
            kappa,
            threshold,
        )
    else:
        logger.info(
            "Batch %r passed IAA check: task=%s, κ=%.4f >= threshold=%.2f",
            batch_id,
            task,
            kappa,
            threshold,
        )

    return IAAResult(
        batch_id=batch_id,
        task=task,
        kappa=kappa,
        threshold=threshold,
        flagged=flagged,
    )


# ---------------------------------------------------------------------------
# High-level batch IAA computation
# ---------------------------------------------------------------------------


def compute_batch_iaa(
    batch_id: str,
    task: TaskName,
    annotator_labels: list,
) -> IAAResult:
    """Compute IAA for a batch and flag it if below threshold.

    This is the main entry point for post-export IAA computation.  It
    dispatches to the appropriate kappa computation function based on *task*,
    then calls :func:`flag_batch`.

    Args:
        batch_id: Identifier for the annotation batch.
        task: Annotation task: 'sentiment', 'ner', or 'toxicity'.
        annotator_labels: Per-annotator labels for the batch.

            - For ``'sentiment'``: ``list[list[str]]`` — each inner list is
              one annotator's labels for all sentences in the batch.
            - For ``'ner'``: ``list[list[list[tuple[int, int, str]]]]`` — each
              inner list is one annotator's span lists for all sentences.
            - For ``'toxicity'``: ``list[list[list[str]]]`` — each inner list
              is one annotator's multi-label sets for all sentences.

    Returns:
        An :class:`IAAResult` with the computed kappa and flagging decision.
    """
    if task == "sentiment":
        kappa = compute_sentiment_kappa(annotator_labels)
    elif task == "ner":
        # For NER, flatten per-sentence spans across all sentences in the batch
        # annotator_labels: [n_annotators][n_sentences][n_spans_per_sentence]
        # We need: [n_annotators][all_spans_in_batch]
        flat_spans: list[list[tuple[int, int, str]]] = []
        for ann_sentences in annotator_labels:
            ann_flat: list[tuple[int, int, str]] = []
            for sentence_spans in ann_sentences:
                ann_flat.extend(sentence_spans)
            flat_spans.append(ann_flat)
        kappa = compute_ner_kappa(flat_spans)
    elif task == "toxicity":
        kappa = compute_toxicity_kappa(annotator_labels)
    else:
        raise ValueError(
            f"Unknown task {task!r}; must be one of 'sentiment', 'ner', 'toxicity'"
        )

    return flag_batch(batch_id, task, kappa)
