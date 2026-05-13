"""
Language_ID_Tagger — token-level language boundary classifier.

Classifies each whitespace-delimited token in a sentence as one of:
  - RAJ  : Rajasthani (Devanagari script, Rajasthani-specific lexicon)
  - HIN  : Hindi (Devanagari script, general Hindi vocabulary)
  - ENG  : English (Latin script, known English words)
  - TRL  : Transliterated (Latin script, Rajasthani/Hindi words in Latin)

Architecture note
-----------------
The production design calls for a fine-tuned MuRIL token-classification head.
For testability without GPU/model downloads, this implementation uses a
rule-based heuristic tagger with the same public interface.  The heuristic
approach is:

  1. Tokens in Devanagari script → look up in the Rajasthani lexicon first;
     if found → RAJ, otherwise → HIN.
  2. Tokens in Latin script → look up in the English word list;
     if found → ENG, otherwise → TRL (transliterated Rajasthani/Hindi).
  3. Mixed or other scripts → HIN as default.

Requirements: 9.1, 9.4
"""

from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from models.data_models import AnnotatedSentence, LangIDMetrics, TokenLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in lexicons
# ---------------------------------------------------------------------------

#: Rajasthani-specific Devanagari tokens (at least 10 entries).
#: These are words that are characteristic of Rajasthani dialect and would
#: NOT typically appear in standard Hindi.
_RAJASTHANI_LEXICON: frozenset[str] = frozenset({
    # Pronouns / possessives
    "म्हारो",   # mharo  — my / our (Rajasthani)
    "म्हारी",   # mhari  — my / our (feminine)
    "म्हाने",   # mhane  — to me / us
    "थारो",     # tharo  — your (Rajasthani)
    "थारी",     # thari  — your (feminine)
    "थाने",     # thane  — to you
    "केई",      # kei    — some / any (Rajasthani)
    "कोनी",     # koni   — is not / are not (Rajasthani negation)
    "बावड़ी",   # bawdi  — step-well (Rajasthani cultural term)
    "पाणी",     # pani   — water (Rajasthani variant)
    "घणो",      # ghano  — a lot / very (Rajasthani)
    "घणी",      # ghani  — a lot (feminine)
    "आवे",      # aave   — comes (Rajasthani verb form)
    "जावे",     # jaave  — goes (Rajasthani verb form)
    "बोलो",     # bolo   — speak (Rajasthani imperative)
    "राजस्थानी", # Rajasthani (the language/people)
    "मारवाड़ी",  # Marwadi (dialect/community name)
    "ढोलो",     # dholo  — drum (Rajasthani folk term)
    "बाईसा",    # baisa  — respectful address for women (Rajasthani)
    "सा",       # sa     — honorific suffix (Rajasthani)
})

#: Common Hindi Devanagari tokens that are NOT Rajasthani-specific.
#: Used to confirm HIN classification (anything Devanagari not in RAJ lexicon).
_HINDI_COMMON: frozenset[str] = frozenset({
    "है", "और", "का", "में", "को", "की", "के", "से", "पर", "यह",
    "वह", "हम", "तुम", "आप", "मैं", "नहीं", "हो", "था", "थी", "थे",
    "होगा", "होगी", "करना", "करते", "करती", "जाना", "आना", "देना",
    "लेना", "बोलना", "कहना", "सुनना", "देखना", "जाता", "आता",
})

#: Known English words (Latin script).  At least 10 entries.
_ENGLISH_LEXICON: frozenset[str] = frozenset({
    "the", "is", "are", "was", "were", "and", "or", "but", "not",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "government", "election", "minister", "party", "state", "india",
    "news", "people", "time", "year", "day", "new", "old", "good",
    "bad", "big", "small", "first", "last", "also", "very", "just",
    "that", "this", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "must", "can", "do", "does", "did",
    "said", "says", "like", "know", "think", "come", "go", "get",
    "make", "take", "see", "look", "want", "need", "use", "work",
})

# ---------------------------------------------------------------------------
# Script detection helpers
# ---------------------------------------------------------------------------

_DEVANAGARI_RANGE = range(0x0900, 0x097F + 1)


def _is_devanagari_char(ch: str) -> bool:
    """Return True if *ch* is a Devanagari Unicode character."""
    return unicodedata.category(ch) != "Zs" and 0x0900 <= ord(ch) <= 0x097F


def _is_latin_char(ch: str) -> bool:
    """Return True if *ch* is a basic Latin letter."""
    return ch.isascii() and ch.isalpha()


def _dominant_script(token: str) -> Literal["devanagari", "latin", "other"]:
    """Determine the dominant script of *token*.

    Returns:
        ``'devanagari'`` if the majority of alphabetic characters are
        Devanagari, ``'latin'`` if the majority are ASCII Latin, or
        ``'other'`` for mixed / unknown scripts.
    """
    devanagari_count = sum(1 for ch in token if _is_devanagari_char(ch))
    latin_count = sum(1 for ch in token if _is_latin_char(ch))
    total = devanagari_count + latin_count

    if total == 0:
        return "other"
    if devanagari_count >= latin_count:
        return "devanagari"
    return "latin"


# ---------------------------------------------------------------------------
# LanguageIDTagger
# ---------------------------------------------------------------------------


class LanguageIDTagger:
    """Token-level language boundary classifier for Rajasthani-Hindi text.

    Assigns each whitespace-delimited token in a sentence exactly one label
    from ``{"RAJ", "HIN", "ENG", "TRL"}``.

    The heuristic classification logic:
    - Devanagari tokens in the Rajasthani lexicon → RAJ (confidence 0.90)
    - Devanagari tokens not in the Rajasthani lexicon → HIN (confidence 0.80)
    - Latin tokens in the English lexicon → ENG (confidence 0.95)
    - Latin tokens not in the English lexicon → TRL (confidence 0.75)
    - Other / mixed script tokens → HIN (confidence 0.60)

    Requirements: 9.1, 9.4
    """

    # Confidence scores per classification path
    _CONF_RAJ: float = 0.90
    _CONF_HIN: float = 0.80
    _CONF_ENG: float = 0.95
    _CONF_TRL: float = 0.75
    _CONF_DEFAULT: float = 0.60

    def __init__(
        self,
        rajasthani_lexicon: frozenset[str] | None = None,
        english_lexicon: frozenset[str] | None = None,
    ) -> None:
        """Initialise the tagger with optional custom lexicons.

        Args:
            rajasthani_lexicon: Custom set of Rajasthani Devanagari tokens.
                Defaults to the built-in ``_RAJASTHANI_LEXICON``.
            english_lexicon: Custom set of English Latin tokens (lowercase).
                Defaults to the built-in ``_ENGLISH_LEXICON``.
        """
        self._raj_lexicon: frozenset[str] = (
            rajasthani_lexicon if rajasthani_lexicon is not None
            else _RAJASTHANI_LEXICON
        )
        self._eng_lexicon: frozenset[str] = (
            english_lexicon if english_lexicon is not None
            else _ENGLISH_LEXICON
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tag(self, sentence: str) -> list[TokenLabel]:
        """Assign a language label to every whitespace-delimited token.

        Args:
            sentence: The input sentence string.  Tokens are split on
                whitespace (``str.split()``).

        Returns:
            A list of :class:`~models.data_models.TokenLabel` objects, one
            per token, in the same order as the tokens in *sentence*.
            The list length equals ``len(sentence.split())``.

        Note:
            An empty string (or a string containing only whitespace) returns
            an empty list.
        """
        tokens = sentence.split()
        return [self._classify_token(tok) for tok in tokens]

    def evaluate(self, test_set: list[AnnotatedSentence]) -> LangIDMetrics:
        """Evaluate the tagger on a labelled test set.

        For each sentence in *test_set*, the tagger's predictions are compared
        against the gold ``token_language_labels``.  Token-level accuracy and
        per-class F1 are computed and returned.

        Args:
            test_set: A list of :class:`~models.data_models.AnnotatedSentence`
                objects with gold ``token_language_labels``.

        Returns:
            A :class:`~models.data_models.LangIDMetrics` instance with
            ``token_accuracy`` and ``per_class_f1``.

        Requirements: 9.4
        """
        all_gold: list[str] = []
        all_pred: list[str] = []

        for sentence in test_set:
            gold_labels = [tl.label for tl in sentence.token_language_labels]
            pred_labels = [tl.label for tl in self.tag(sentence.text)]

            # Align lengths: use the shorter of the two (handles edge cases)
            min_len = min(len(gold_labels), len(pred_labels))
            all_gold.extend(gold_labels[:min_len])
            all_pred.extend(pred_labels[:min_len])

        if not all_gold:
            logger.warning("evaluate() called with empty test set; returning zero metrics")
            return LangIDMetrics(
                token_accuracy=0.0,
                per_class_f1={label: 0.0 for label in ("RAJ", "HIN", "ENG", "TRL")},
            )

        # Token accuracy
        correct = sum(g == p for g, p in zip(all_gold, all_pred))
        token_accuracy = correct / len(all_gold)

        # Per-class F1
        per_class_f1 = self._compute_per_class_f1(all_gold, all_pred)

        logger.info(
            "Language_ID_Tagger evaluation: token_accuracy=%.4f, per_class_f1=%s",
            token_accuracy,
            per_class_f1,
        )

        return LangIDMetrics(
            token_accuracy=token_accuracy,
            per_class_f1=per_class_f1,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_token(self, token: str) -> TokenLabel:
        """Classify a single token and return a :class:`TokenLabel`."""
        script = _dominant_script(token)

        if script == "devanagari":
            if token in self._raj_lexicon:
                return TokenLabel(token=token, label="RAJ", confidence=self._CONF_RAJ)
            return TokenLabel(token=token, label="HIN", confidence=self._CONF_HIN)

        if script == "latin":
            if token.lower() in self._eng_lexicon:
                return TokenLabel(token=token, label="ENG", confidence=self._CONF_ENG)
            return TokenLabel(token=token, label="TRL", confidence=self._CONF_TRL)

        # Mixed / other script → default to HIN
        return TokenLabel(token=token, label="HIN", confidence=self._CONF_DEFAULT)

    @staticmethod
    def _compute_per_class_f1(
        gold: list[str],
        pred: list[str],
    ) -> dict[str, float]:
        """Compute per-class F1 scores from gold and predicted label lists.

        Uses the standard precision/recall/F1 formula:
            precision = TP / (TP + FP)
            recall    = TP / (TP + FN)
            F1        = 2 * precision * recall / (precision + recall)

        Returns 0.0 for any class with no gold or predicted instances.
        """
        classes = ("RAJ", "HIN", "ENG", "TRL")
        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)

        for g, p in zip(gold, pred):
            if g == p:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1

        per_class_f1: dict[str, float] = {}
        for cls in classes:
            precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
            recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            per_class_f1[cls] = f1

        return per_class_f1
