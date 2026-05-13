"""
NER_Tagger — token-level named entity recognition model.

Architecture (production): MuRIL + token classification head with BIO tags:
  B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, O.

For testability without GPU/model downloads, this module implements a
**rule-based / heuristic tagger** with the same public interface as the real
MuRIL-based tagger.

Heuristic logic
---------------
- Maintains a small built-in entity lexicon:
    * PER: known politician names (e.g., "Gehlot", "Vasundhara", "Modi")
    * LOC: known location names (e.g., "Jaipur", "Jodhpur", "Rajasthan")
    * ORG: known organization names (e.g., "BJP", "Congress", "INC")
- ``tag(sentence)`` scans whitespace-delimited tokens for lexicon matches
  and returns EntitySpan objects with correct character offsets.
- BIO tagging: single-token entities get B- tag; multi-token entities get
  B- for first token, I- for subsequent tokens.
- ``train()`` simulates training: calls ``set_all_seeds(seed)``, iterates
  epochs, tracks best span-F1.
- ``evaluate()`` computes real span-level metrics using seqeval.

Requirements: 11.1, 11.2, 11.5, 17.1, 17.3
"""

from __future__ import annotations

import logging
from collections import defaultdict

from models.data_models import (
    AnnotatedSentence,
    EntitySpan,
    NERMetrics,
    NERPrediction,
    TrainingLog,
)
from models.reproducibility import set_all_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BIO tag constants
# ---------------------------------------------------------------------------

BIO_TAGS: tuple[str, ...] = (
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "O",
)

ENTITY_TYPES: tuple[str, ...] = ("PER", "LOC", "ORG")

# ---------------------------------------------------------------------------
# Built-in entity lexicon
# ---------------------------------------------------------------------------

# Each entry is a tuple of tokens (supports multi-token entities).
# Single-token entries are 1-tuples; multi-token entries are longer tuples.

_PER_LEXICON: list[tuple[str, ...]] = [
    ("Gehlot",),
    ("Ashok", "Gehlot"),
    ("Vasundhara",),
    ("Vasundhara", "Raje"),
    ("Modi",),
    ("Narendra", "Modi"),
    ("Rahul",),
    ("Rahul", "Gandhi"),
    ("Sachin",),
    ("Sachin", "Pilot"),
    ("Ashok",),
    ("Pilot",),
    ("Gandhi",),
    ("Raje",),
]

_LOC_LEXICON: list[tuple[str, ...]] = [
    ("Jaipur",),
    ("Jodhpur",),
    ("Udaipur",),
    ("Rajasthan",),
    ("Delhi",),
    ("New", "Delhi"),
    ("Mumbai",),
    ("India",),
    ("Kota",),
    ("Ajmer",),
    ("Bikaner",),
    ("Alwar",),
]

_ORG_LEXICON: list[tuple[str, ...]] = [
    ("BJP",),
    ("Congress",),
    ("INC",),
    ("AAP",),
    ("RSS",),
    ("AICC",),
    ("Indian", "National", "Congress"),
    ("Bharatiya", "Janata", "Party"),
]


def _build_lexicon_index(
    lexicon: list[tuple[str, ...]],
    entity_type: str,
) -> dict[str, list[tuple[tuple[str, ...], str]]]:
    """Build a lookup index from first token → list of (phrase_tokens, entity_type).

    This allows O(1) lookup by first token during tagging.
    """
    index: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
    for phrase in lexicon:
        index[phrase[0]].append((phrase, entity_type))
    return index


def _build_full_lexicon_index() -> dict[str, list[tuple[tuple[str, ...], str]]]:
    """Combine all entity lexicons into a single first-token index."""
    index: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
    for lexicon, etype in [
        (_PER_LEXICON, "PER"),
        (_LOC_LEXICON, "LOC"),
        (_ORG_LEXICON, "ORG"),
    ]:
        for phrase in lexicon:
            index[phrase[0]].append((phrase, etype))
    return index


# Pre-built at module load time for efficiency
_LEXICON_INDEX: dict[str, list[tuple[tuple[str, ...], str]]] = _build_full_lexicon_index()


# ---------------------------------------------------------------------------
# BIO tag utilities
# ---------------------------------------------------------------------------


def spans_to_bio_tags(tokens: list[str], spans: list[EntitySpan], sentence: str) -> list[str]:
    """Convert a list of EntitySpan objects to a BIO tag sequence.

    Args:
        tokens: Whitespace-delimited tokens of the sentence.
        spans: Named entity spans (character-offset based).
        sentence: The original sentence string.

    Returns:
        A list of BIO tags, one per token.
    """
    # Build token character offsets
    token_offsets: list[tuple[int, int]] = []
    pos = 0
    for token in tokens:
        start = sentence.find(token, pos)
        if start == -1:
            # Fallback: use current position
            start = pos
        end = start + len(token)
        token_offsets.append((start, end))
        pos = end

    # Initialize all tags to O
    bio = ["O"] * len(tokens)

    # For each span, find which tokens it covers and assign BIO tags
    for span in spans:
        covered: list[int] = []
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            # Token overlaps with span if tok_start >= span.start and tok_end <= span.end
            if tok_start >= span.start and tok_end <= span.end:
                covered.append(i)

        if not covered:
            continue

        etype = span.entity_type
        bio[covered[0]] = f"B-{etype}"
        for idx in covered[1:]:
            bio[idx] = f"I-{etype}"

    return bio


def validate_bio_tags(bio_tags: list[str]) -> bool:
    """Validate that no I- tag appears without a preceding B- or I- tag of the same type.

    Args:
        bio_tags: List of BIO tag strings.

    Returns:
        True if the sequence is valid, False otherwise.
    """
    current_entity: str | None = None

    for tag in bio_tags:
        if tag == "O":
            current_entity = None
        elif tag.startswith("B-"):
            current_entity = tag[2:]  # entity type after "B-"
        elif tag.startswith("I-"):
            etype = tag[2:]
            if current_entity != etype:
                return False
            # current_entity stays the same
        else:
            # Unknown tag
            return False

    return True


# ---------------------------------------------------------------------------
# NERTagger
# ---------------------------------------------------------------------------


class NERTagger:
    """Heuristic stub for the MuRIL-based BIO NER tagger.

    Public interface mirrors the production MuRIL fine-tuned model so that
    all downstream code and tests work without GPU/model downloads.

    Requirements: 11.1, 11.2, 11.5, 17.1, 17.3
    """

    def __init__(
        self,
        lexicon_index: dict[str, list[tuple[tuple[str, ...], str]]] | None = None,
    ) -> None:
        """Initialise the NER tagger.

        Args:
            lexicon_index: Custom lexicon index (first-token → phrase list).
                Defaults to the built-in ``_LEXICON_INDEX``.
        """
        self._lexicon_index = lexicon_index if lexicon_index is not None else _LEXICON_INDEX

        # State set during training
        self._best_f1: float = 0.0
        self._best_epoch: int = 0
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def tag(self, sentence: str) -> list[EntitySpan]:
        """Tag named entities in *sentence* using lexicon matching.

        Scans whitespace-delimited tokens for matches in the entity lexicon.
        Longer phrases take priority over shorter ones (greedy left-to-right,
        longest match first).

        Args:
            sentence: Input sentence string.

        Returns:
            A list of :class:`~models.data_models.EntitySpan` objects.

        Requirements: 11.2
        """
        tokens = sentence.split()
        if not tokens:
            return []

        # Build token character offsets
        token_offsets: list[tuple[int, int]] = []
        pos = 0
        for token in tokens:
            start = sentence.find(token, pos)
            if start == -1:
                start = pos
            end = start + len(token)
            token_offsets.append((start, end))
            pos = end

        spans: list[EntitySpan] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            candidates = self._lexicon_index.get(token, [])

            if not candidates:
                i += 1
                continue

            # Sort by phrase length descending (longest match first)
            candidates_sorted = sorted(candidates, key=lambda x: len(x[0]), reverse=True)

            matched = False
            for phrase_tokens, etype in candidates_sorted:
                phrase_len = len(phrase_tokens)
                # Check if the phrase fits within remaining tokens
                if i + phrase_len > len(tokens):
                    continue
                # Check if all tokens match
                if all(tokens[i + j] == phrase_tokens[j] for j in range(phrase_len)):
                    # Found a match
                    span_start = token_offsets[i][0]
                    span_end = token_offsets[i + phrase_len - 1][1]
                    span_text = sentence[span_start:span_end]
                    spans.append(EntitySpan(
                        start=span_start,
                        end=span_end,
                        entity_type=etype,  # type: ignore[arg-type]
                        text=span_text,
                    ))
                    i += phrase_len
                    matched = True
                    break

            if not matched:
                i += 1

        return spans

    def tag_with_bio(self, sentence: str) -> NERPrediction:
        """Tag *sentence* and return both EntitySpan objects and BIO tags.

        Args:
            sentence: Input sentence string.

        Returns:
            A :class:`~models.data_models.NERPrediction` with spans and BIO tags.
        """
        tokens = sentence.split()
        spans = self.tag(sentence)
        bio_tags = spans_to_bio_tags(tokens, spans, sentence)
        return NERPrediction(spans=spans, bio_tags=bio_tags)

    # ------------------------------------------------------------------
    # Training simulation
    # ------------------------------------------------------------------

    def train(
        self,
        train_set: list[AnnotatedSentence],
        val_set: list[AnnotatedSentence],
        seed: int,
        max_epochs: int = 5,
        patience: int = 3,
        lr: float = 3e-5,
        batch_size: int = 16,
    ) -> TrainingLog:
        """Simulate training the NER tagger.

        Steps:
        1. Call ``set_all_seeds(seed)`` to fix all random number generators.
        2. Iterate up to ``max_epochs`` epochs:
           a. Compute validation span-F1 using heuristic tag on val_set.
           b. Track best checkpoint (best_f1 and best_epoch).
           c. Apply early stopping (patience).
        3. Return a :class:`~models.data_models.TrainingLog`.

        Args:
            train_set: Training partition of annotated sentences.
            val_set: Validation partition of annotated sentences.
            seed: Random seed for reproducibility.
            max_epochs: Maximum number of training epochs (default 5).
            patience: Early stopping patience (default 3).
            lr: Learning rate (logged; not used in stub).
            batch_size: Batch size (logged; not used in stub).

        Returns:
            A :class:`~models.data_models.TrainingLog` with training details.

        Requirements: 11.1, 17.1, 17.3
        """
        # Step 1: Fix all random seeds
        set_all_seeds(seed)
        logger.info(
            "NERTagger.train: seed=%d, max_epochs=%d, patience=%d, "
            "lr=%g, batch_size=%d, train_size=%d, val_size=%d",
            seed, max_epochs, patience, lr, batch_size,
            len(train_set), len(val_set),
        )

        best_f1 = 0.0
        best_epoch = 0
        no_improve_count = 0
        total_epochs = 0

        for epoch in range(max_epochs):
            total_epochs = epoch + 1

            # Compute validation span-F1
            if val_set:
                val_metrics = self.evaluate(val_set)
                val_f1 = val_metrics.macro_f1
            else:
                val_f1 = 0.0

            logger.info("Epoch %d/%d — val span-F1: %.4f", epoch + 1, max_epochs, val_f1)

            # Track best checkpoint
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                no_improve_count = 0
                logger.info("New best checkpoint at epoch %d (F1=%.4f)", epoch + 1, val_f1)
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= patience:
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d exhausted)",
                    epoch + 1, patience,
                )
                break

        self._best_f1 = best_f1
        self._best_epoch = best_epoch
        self._trained = True

        return TrainingLog(
            best_epoch=best_epoch,
            best_f1=best_f1,
            total_epochs_run=total_epochs,
            seed=seed,
            class_weights={etype: 1.0 for etype in ENTITY_TYPES},
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_set: list[AnnotatedSentence]) -> NERMetrics:
        """Evaluate the tagger on *test_set* using seqeval span-level metrics.

        Converts gold NER spans and predicted NER spans to BIO tag sequences,
        then uses seqeval to compute span-level precision, recall, and F1 per
        entity type.

        Args:
            test_set: List of annotated sentences with gold ``ner_spans``.

        Returns:
            A :class:`~models.data_models.NERMetrics` instance.

        Requirements: 11.5
        """
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )

        if not test_set:
            logger.warning("evaluate() called with empty test set; returning zero metrics")
            return NERMetrics(
                macro_f1=0.0,
                per_type_precision={etype: 0.0 for etype in ENTITY_TYPES},
                per_type_recall={etype: 0.0 for etype in ENTITY_TYPES},
                per_type_f1={etype: 0.0 for etype in ENTITY_TYPES},
            )

        gold_sequences: list[list[str]] = []
        pred_sequences: list[list[str]] = []

        for sentence in test_set:
            tokens = sentence.text.split()
            if not tokens:
                continue

            # Gold BIO tags from gold ner_spans
            gold_bio = spans_to_bio_tags(tokens, sentence.ner_spans, sentence.text)
            gold_sequences.append(gold_bio)

            # Predicted BIO tags from heuristic tagger
            pred_spans = self.tag(sentence.text)
            pred_bio = spans_to_bio_tags(tokens, pred_spans, sentence.text)
            pred_sequences.append(pred_bio)

        if not gold_sequences:
            return NERMetrics(
                macro_f1=0.0,
                per_type_precision={etype: 0.0 for etype in ENTITY_TYPES},
                per_type_recall={etype: 0.0 for etype in ENTITY_TYPES},
                per_type_f1={etype: 0.0 for etype in ENTITY_TYPES},
            )

        # Compute overall macro-F1 using seqeval
        macro_f1 = f1_score(gold_sequences, pred_sequences, average="macro", zero_division=0)

        # Compute per-type metrics using classification_report
        report = classification_report(
            gold_sequences,
            pred_sequences,
            output_dict=True,
            zero_division=0,
        )

        per_type_precision: dict[str, float] = {}
        per_type_recall: dict[str, float] = {}
        per_type_f1: dict[str, float] = {}

        for etype in ENTITY_TYPES:
            if etype in report:
                per_type_precision[etype] = float(report[etype]["precision"])
                per_type_recall[etype] = float(report[etype]["recall"])
                per_type_f1[etype] = float(report[etype]["f1-score"])
            else:
                per_type_precision[etype] = 0.0
                per_type_recall[etype] = 0.0
                per_type_f1[etype] = 0.0

        logger.info(
            "NERTagger.evaluate: macro_f1=%.4f, per_type_f1=%s",
            macro_f1, per_type_f1,
        )

        return NERMetrics(
            macro_f1=float(macro_f1),
            per_type_precision=per_type_precision,
            per_type_recall=per_type_recall,
            per_type_f1=per_type_f1,
        )
