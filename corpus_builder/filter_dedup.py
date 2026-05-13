"""
Corpus_Builder — filtering and deduplication for RajNLP-50K.

Provides two public functions:

- ``filter_rajasthani``: Retains only sentences that contain at least
  ``min_tokens`` (default 2) tokens matching the bundled Rajasthani lexicon.

- ``deduplicate``: Removes duplicate sentences from a combined pool using a
  two-pass strategy:
    1. Exact string match after Unicode NFC normalization.
    2. Near-duplicate detection via MinHash LSH (Jaccard threshold 0.85).
"""

from __future__ import annotations

import logging
import unicodedata
from importlib import resources
from pathlib import Path
from typing import Iterable

from datasketch import MinHash, MinHashLSH

from models.data_models import RawSentence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default minimum number of Rajasthani lexicon tokens required to keep a sentence.
DEFAULT_MIN_TOKENS: int = 2

#: Jaccard similarity threshold for near-duplicate detection.
_JACCARD_THRESHOLD: float = 0.85

#: Number of permutations for MinHash (controls accuracy vs. speed trade-off).
_MINHASH_NUM_PERM: int = 128

#: Path to the bundled Rajasthani lexicon file, relative to this module's package.
_LEXICON_FILENAME: str = "rajasthani_lexicon.txt"


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------


def _load_lexicon() -> frozenset[str]:
    """Load the bundled Rajasthani lexicon and return it as a frozenset of tokens.

    Lines starting with ``#`` and blank lines are ignored.  All tokens are
    lowercased and Unicode-NFC-normalised for consistent matching.

    Returns:
        A :class:`frozenset` of lowercase NFC-normalised Rajasthani tokens.
    """
    lexicon_path = Path(__file__).parent / _LEXICON_FILENAME
    tokens: set[str] = set()
    with lexicon_path.open(encoding="utf-8") as fh:
        for line in fh:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            tokens.add(unicodedata.normalize("NFC", token.lower()))
    logger.debug("Loaded Rajasthani lexicon with %d tokens from %s", len(tokens), lexicon_path)
    return frozenset(tokens)


# Module-level singleton — loaded once on first import.
_RAJASTHANI_LEXICON: frozenset[str] | None = None


def _get_lexicon() -> frozenset[str]:
    """Return the cached lexicon, loading it on first call."""
    global _RAJASTHANI_LEXICON
    if _RAJASTHANI_LEXICON is None:
        _RAJASTHANI_LEXICON = _load_lexicon()
    return _RAJASTHANI_LEXICON


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split *text* into whitespace-delimited tokens, NFC-normalised and lowercased.

    Args:
        text: Raw sentence text.

    Returns:
        List of normalised token strings.
    """
    return [
        unicodedata.normalize("NFC", tok.lower())
        for tok in text.split()
        if tok.strip()
    ]


def filter_rajasthani(
    sentences: list[RawSentence],
    min_tokens: int = DEFAULT_MIN_TOKENS,
    lexicon: frozenset[str] | None = None,
) -> list[RawSentence]:
    """Retain only sentences containing at least *min_tokens* Rajasthani lexicon tokens.

    The filter is applied identically to sentences from both Twitter/X and
    ShareChat (Requirements 1.2, 2.2).

    Args:
        sentences: Input list of :class:`~models.data_models.RawSentence` objects.
        min_tokens: Minimum number of Rajasthani-specific tokens required to
            keep a sentence.  Defaults to 2.
        lexicon: Optional pre-loaded lexicon frozenset (used in tests to inject
            a custom lexicon without touching the file system).  If ``None``,
            the bundled lexicon is loaded automatically.

    Returns:
        A new list containing only the sentences that pass the filter.
    """
    if lexicon is None:
        lexicon = _get_lexicon()

    kept: list[RawSentence] = []
    for sentence in sentences:
        tokens = _tokenize(sentence.text)
        raj_count = sum(1 for tok in tokens if tok in lexicon)
        if raj_count >= min_tokens:
            kept.append(sentence)

    logger.info(
        "filter_rajasthani: kept %d / %d sentences (min_tokens=%d)",
        len(kept),
        len(sentences),
        min_tokens,
    )
    return kept


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


def _nfc_text(text: str) -> str:
    """Return the Unicode NFC normalisation of *text*."""
    return unicodedata.normalize("NFC", text)


def _make_minhash(text: str, num_perm: int = _MINHASH_NUM_PERM) -> MinHash:
    """Create a MinHash signature for *text* using character 3-grams.

    Character n-grams are more robust than word tokens for near-duplicate
    detection in multilingual / code-switched text.

    Args:
        text: The sentence text to hash.
        num_perm: Number of permutations for the MinHash.

    Returns:
        A :class:`datasketch.MinHash` object.
    """
    mh = MinHash(num_perm=num_perm)
    # Use character 3-grams of the NFC-normalised text
    normalised = _nfc_text(text)
    for i in range(len(normalised) - 2):
        mh.update(normalised[i : i + 3].encode("utf-8"))
    return mh


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate(
    sentences: list[RawSentence],
    jaccard_threshold: float = _JACCARD_THRESHOLD,
    num_perm: int = _MINHASH_NUM_PERM,
) -> list[RawSentence]:
    """Remove duplicate and near-duplicate sentences from *sentences*.

    Two-pass strategy (Requirements 1.3, 2.3):

    **Pass 1 — Exact deduplication**
        Normalise each sentence text to Unicode NFC and keep only the first
        occurrence of each unique normalised string.

    **Pass 2 — Near-duplicate deduplication (MinHash LSH)**
        Build a MinHash LSH index with the given Jaccard threshold.  For each
        remaining sentence, query the index; if a near-duplicate is found,
        skip the sentence.  Otherwise, insert it into the index and keep it.

    Deduplication is applied across the combined pool regardless of platform,
    so a sentence appearing on both Twitter/X and ShareChat is deduplicated.

    Args:
        sentences: Input list of :class:`~models.data_models.RawSentence` objects.
        jaccard_threshold: Jaccard similarity threshold above which two
            sentences are considered near-duplicates.  Defaults to 0.85.
        num_perm: Number of MinHash permutations.  Defaults to 128.

    Returns:
        A new list with duplicates and near-duplicates removed.  The first
        occurrence of each unique sentence is retained.
    """
    if not sentences:
        return []

    # ------------------------------------------------------------------
    # Pass 1: Exact deduplication via NFC-normalised string
    # ------------------------------------------------------------------
    seen_exact: set[str] = set()
    after_exact: list[RawSentence] = []
    for sentence in sentences:
        key = _nfc_text(sentence.text)
        if key not in seen_exact:
            seen_exact.add(key)
            after_exact.append(sentence)

    exact_removed = len(sentences) - len(after_exact)
    logger.info(
        "deduplicate pass 1 (exact): removed %d duplicates, %d remaining",
        exact_removed,
        len(after_exact),
    )

    # ------------------------------------------------------------------
    # Pass 2: Near-duplicate deduplication via MinHash LSH
    # ------------------------------------------------------------------
    # MinHashLSH requires at least one item; guard against empty input.
    if not after_exact:
        return []

    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    kept: list[RawSentence] = []

    for sentence in after_exact:
        mh = _make_minhash(sentence.text, num_perm=num_perm)
        # Query for near-duplicates already in the index
        candidates = lsh.query(mh)
        if candidates:
            # A near-duplicate already exists — skip this sentence
            logger.debug(
                "Near-duplicate detected for sentence_id=%s (similar to %s) — skipping",
                sentence.sentence_id,
                candidates[0],
            )
            continue
        # No near-duplicate found — insert and keep
        lsh.insert(sentence.sentence_id, mh)
        kept.append(sentence)

    near_dup_removed = len(after_exact) - len(kept)
    logger.info(
        "deduplicate pass 2 (MinHash LSH, threshold=%.2f): removed %d near-duplicates, %d remaining",
        jaccard_threshold,
        near_dup_removed,
        len(kept),
    )

    return kept
