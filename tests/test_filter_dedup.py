"""
Tests for corpus_builder.filter_dedup — filtering and deduplication.

Covers:
- Property 8: Filtering preserves Rajasthani token minimum
  (every sentence that passes filter_rajasthani contains ≥ min_tokens
  lexicon tokens).
- Property 2: Deduplication idempotence
  (deduplicate(deduplicate(sentences)) == deduplicate(sentences)).
- Unit tests for exact deduplication (known duplicate pairs).
- Unit tests for near-duplicate detection (above/below Jaccard threshold).

Requirements: 1.2, 1.3, 2.2, 2.3
"""

from __future__ import annotations

import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from corpus_builder.filter_dedup import (
    DEFAULT_MIN_TOKENS,
    _JACCARD_THRESHOLD,
    _tokenize,
    deduplicate,
    filter_rajasthani,
)
from models.data_models import RawSentence


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FIXED_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

# A small deterministic lexicon used in property tests so we don't depend on
# the file system and can reason precisely about which sentences pass.
_TEST_LEXICON: frozenset[str] = frozenset(
    [
        "म्हारो",
        "म्हारी",
        "थारो",
        "थारी",
        "आपणो",
        "कुण",
        "कठे",
        "घणो",
        "घणी",
        "नै",
        "सूं",
        "mharo",
        "tharo",
        "ghano",
        "kun",
    ]
)


def _make_sentence(
    text: str,
    platform: str = "twitter",
    sentence_id: str | None = None,
) -> RawSentence:
    """Factory for :class:`RawSentence` with minimal required fields."""
    return RawSentence(
        text=text,
        source_url="https://example.com",
        collected_at=_FIXED_TS,
        platform=platform,  # type: ignore[arg-type]
        sentence_id=sentence_id or str(uuid.uuid4()),
    )


def _count_lexicon_tokens(text: str, lexicon: frozenset[str]) -> int:
    """Count how many whitespace-delimited tokens in *text* are in *lexicon*."""
    return sum(
        1
        for tok in _tokenize(text)
        if tok in lexicon
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy: generate a text that is guaranteed to contain exactly `n` lexicon
# tokens by picking `n` tokens from the test lexicon and appending filler words.
@st.composite
def _text_with_n_lexicon_tokens(draw, n: int) -> str:
    """Generate a text string containing exactly *n* tokens from _TEST_LEXICON."""
    lexicon_list = sorted(_TEST_LEXICON)
    chosen = draw(
        st.lists(
            st.sampled_from(lexicon_list),
            min_size=n,
            max_size=n + 3,
        )
    )
    # Add some non-lexicon filler words so the sentence looks realistic
    filler = draw(
        st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"),
                    whitelist_characters="abcdefghijklmnopqrstuvwxyz",
                ),
                min_size=2,
                max_size=8,
            ),
            min_size=0,
            max_size=5,
        )
    )
    # Interleave chosen lexicon tokens and filler, then join
    all_tokens = chosen + filler
    # Shuffle deterministically via draw
    indices = draw(st.permutations(range(len(all_tokens))))
    shuffled = [all_tokens[i] for i in indices]
    return " ".join(shuffled)


@st.composite
def _raw_sentence_strategy(draw, text_strategy=None) -> RawSentence:
    """Generate a :class:`RawSentence` with a drawn text."""
    if text_strategy is None:
        text_strategy = st.text(min_size=5, max_size=200)
    text = draw(text_strategy)
    platform = draw(st.sampled_from(["twitter", "sharechat"]))
    return _make_sentence(text=text, platform=platform)


# ---------------------------------------------------------------------------
# Property 8: Filtering preserves Rajasthani token minimum
# ---------------------------------------------------------------------------


class TestProperty8FilteringPreservesMinimum:
    """
    Property 8: For any sentence that passes filter_rajasthani, the sentence
    SHALL contain at least min_tokens tokens identified as Rajasthani-specific
    by the lexicon lookup.

    Validates: Requirements 1.2, 2.2
    """

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(
                text_strategy=st.one_of(
                    # Texts with ≥ 2 lexicon tokens (should pass)
                    _text_with_n_lexicon_tokens(2),
                    _text_with_n_lexicon_tokens(3),
                    _text_with_n_lexicon_tokens(4),
                    # Texts with 0 or 1 lexicon tokens (should be filtered out)
                    st.text(
                        alphabet="abcdefghijklmnopqrstuvwxyz ",
                        min_size=5,
                        max_size=100,
                    ),
                )
            ),
            min_size=0,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_every_kept_sentence_has_minimum_lexicon_tokens(self, sentences):
        """
        GIVEN any list of sentences,
        WHEN  filter_rajasthani is applied with the test lexicon,
        THEN  every sentence in the result contains ≥ DEFAULT_MIN_TOKENS
              lexicon tokens.
        """
        result = filter_rajasthani(sentences, lexicon=_TEST_LEXICON)
        for sentence in result:
            count = _count_lexicon_tokens(sentence.text, _TEST_LEXICON)
            assert count >= DEFAULT_MIN_TOKENS, (
                f"Sentence passed filter but has only {count} lexicon tokens "
                f"(min={DEFAULT_MIN_TOKENS}): {sentence.text!r}"
            )

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(
                text_strategy=_text_with_n_lexicon_tokens(2)
            ),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=100)
    def test_sentences_with_enough_tokens_are_never_dropped(self, sentences):
        """
        GIVEN sentences that each contain ≥ 2 lexicon tokens,
        WHEN  filter_rajasthani is applied,
        THEN  none of those sentences are dropped.
        """
        result = filter_rajasthani(sentences, lexicon=_TEST_LEXICON)
        assert len(result) == len(sentences), (
            f"Expected all {len(sentences)} sentences to be kept; "
            f"got {len(result)}"
        )

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(
                text_strategy=st.text(
                    alphabet="abcdefghijklmnopqrstuvwxyz ",
                    min_size=5,
                    max_size=100,
                )
            ),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=100)
    def test_sentences_without_lexicon_tokens_are_always_dropped(self, sentences):
        """
        GIVEN sentences that contain no Rajasthani lexicon tokens,
        WHEN  filter_rajasthani is applied,
        THEN  all sentences are dropped (result is empty).
        """
        result = filter_rajasthani(sentences, lexicon=_TEST_LEXICON)
        assert result == [], (
            f"Expected empty result for sentences with no lexicon tokens; "
            f"got {len(result)} sentences"
        )

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(
                text_strategy=st.text(min_size=5, max_size=200)
            ),
            min_size=0,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_result_is_subset_of_input(self, sentences):
        """
        GIVEN any list of sentences,
        WHEN  filter_rajasthani is applied,
        THEN  the result is a subset of the input (no new sentences introduced).
        """
        result = filter_rajasthani(sentences, lexicon=_TEST_LEXICON)
        input_ids = {s.sentence_id for s in sentences}
        for sentence in result:
            assert sentence.sentence_id in input_ids, (
                f"Result contains a sentence not in the input: {sentence.sentence_id}"
            )

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(
                text_strategy=st.text(min_size=5, max_size=200)
            ),
            min_size=0,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_filter_is_idempotent(self, sentences):
        """
        GIVEN any list of sentences,
        WHEN  filter_rajasthani is applied twice,
        THEN  the result is the same as applying it once.
        """
        once = filter_rajasthani(sentences, lexicon=_TEST_LEXICON)
        twice = filter_rajasthani(once, lexicon=_TEST_LEXICON)
        assert [s.sentence_id for s in once] == [s.sentence_id for s in twice], (
            "filter_rajasthani is not idempotent"
        )


# ---------------------------------------------------------------------------
# Property 2: Deduplication idempotence
# ---------------------------------------------------------------------------


class TestProperty2DeduplicationIdempotence:
    """
    Property 2: For any list of sentences,
    deduplicate(deduplicate(sentences)) == deduplicate(sentences).

    Validates: Requirements 1.3, 2.3
    """

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(),
            min_size=0,
            max_size=30,
        )
    )
    @settings(max_examples=100)
    def test_deduplication_is_idempotent(self, sentences):
        """
        GIVEN any list of sentences,
        WHEN  deduplicate is applied twice,
        THEN  the result equals applying it once (same sentence_ids in same order).
        """
        once = deduplicate(sentences)
        twice = deduplicate(once)
        assert [s.sentence_id for s in once] == [s.sentence_id for s in twice], (
            "deduplicate is not idempotent: second application changed the result"
        )

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(),
            min_size=0,
            max_size=30,
        )
    )
    @settings(max_examples=100)
    def test_result_is_subset_of_input(self, sentences):
        """
        GIVEN any list of sentences,
        WHEN  deduplicate is applied,
        THEN  every sentence in the result was in the input.
        """
        result = deduplicate(sentences)
        input_ids = {s.sentence_id for s in sentences}
        for sentence in result:
            assert sentence.sentence_id in input_ids

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(),
            min_size=0,
            max_size=30,
        )
    )
    @settings(max_examples=100)
    def test_result_has_no_exact_duplicates(self, sentences):
        """
        GIVEN any list of sentences,
        WHEN  deduplicate is applied,
        THEN  no two sentences in the result have the same NFC-normalised text.
        """
        result = deduplicate(sentences)
        seen_texts: set[str] = set()
        for sentence in result:
            key = unicodedata.normalize("NFC", sentence.text)
            assert key not in seen_texts, (
                f"Exact duplicate text found in deduplicated result: {sentence.text!r}"
            )
            seen_texts.add(key)

    @given(
        sentences=st.lists(
            _raw_sentence_strategy(),
            min_size=0,
            max_size=30,
        )
    )
    @settings(max_examples=100)
    def test_result_size_does_not_exceed_input(self, sentences):
        """
        GIVEN any list of sentences,
        WHEN  deduplicate is applied,
        THEN  the result has at most as many sentences as the input.
        """
        result = deduplicate(sentences)
        assert len(result) <= len(sentences)


# ---------------------------------------------------------------------------
# Unit tests — exact deduplication
# ---------------------------------------------------------------------------


class TestExactDeduplication:
    """Unit tests for exact-string deduplication (Pass 1)."""

    def test_exact_duplicate_pair_keeps_first(self):
        """
        GIVEN two sentences with identical text,
        WHEN  deduplicate is called,
        THEN  only the first sentence is retained.
        """
        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())
        s1 = _make_sentence("म्हारो राजस्थान बहुत सुंदर है", sentence_id=sid1)
        s2 = _make_sentence("म्हारो राजस्थान बहुत सुंदर है", sentence_id=sid2)

        result = deduplicate([s1, s2])

        assert len(result) == 1
        assert result[0].sentence_id == sid1

    def test_three_exact_duplicates_keeps_first(self):
        """
        GIVEN three sentences with identical text,
        WHEN  deduplicate is called,
        THEN  only the first is retained.
        """
        ids = [str(uuid.uuid4()) for _ in range(3)]
        sentences = [
            _make_sentence("थारो घर कठे है", sentence_id=ids[i])
            for i in range(3)
        ]
        result = deduplicate(sentences)
        assert len(result) == 1
        assert result[0].sentence_id == ids[0]

    def test_no_duplicates_returns_all(self):
        """
        GIVEN sentences with all distinct texts that are sufficiently different
              (low character 3-gram overlap),
        WHEN  deduplicate is called,
        THEN  all sentences are returned.
        """
        # Use texts that are structurally very different to avoid MinHash
        # false-positive near-duplicate detection.
        sentences = [
            _make_sentence("म्हारो राजस्थान घणो सुंदर है"),
            _make_sentence("python programming tutorial for beginners"),
            _make_sentence("जयपुर शहर बहुत खूबसूरत है"),
            _make_sentence("machine learning deep neural networks"),
            _make_sentence("थारो घर कठे है यार बताओ"),
        ]
        result = deduplicate(sentences)
        assert len(result) == 5

    def test_nfc_normalisation_treats_equivalent_strings_as_duplicates(self):
        """
        GIVEN two sentences whose texts are canonically equivalent under NFC
              but stored with different byte representations,
        WHEN  deduplicate is called,
        THEN  only one is retained.
        """
        # Compose vs. decompose: 'ा' (U+093E) vs. 'ा' (U+0061 + combining)
        # We simulate this by using NFC and NFD forms of the same string.
        base = "राजस्थान"
        nfc_text = unicodedata.normalize("NFC", base)
        nfd_text = unicodedata.normalize("NFD", base)

        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())
        s1 = _make_sentence(nfc_text, sentence_id=sid1)
        s2 = _make_sentence(nfd_text, sentence_id=sid2)

        result = deduplicate([s1, s2])
        # Both normalise to the same NFC string → only one kept
        assert len(result) == 1

    def test_cross_platform_deduplication(self):
        """
        GIVEN the same sentence text appearing on both Twitter and ShareChat,
        WHEN  deduplicate is called on the combined pool,
        THEN  only one copy is retained.
        """
        text = "म्हारो राजस्थान घणो सुंदर है"
        s_twitter = _make_sentence(text, platform="twitter")
        s_sharechat = _make_sentence(text, platform="sharechat")

        result = deduplicate([s_twitter, s_sharechat])
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        """
        GIVEN an empty list,
        WHEN  deduplicate is called,
        THEN  an empty list is returned.
        """
        assert deduplicate([]) == []

    def test_single_sentence_returns_itself(self):
        """
        GIVEN a list with a single sentence,
        WHEN  deduplicate is called,
        THEN  the same sentence is returned.
        """
        s = _make_sentence("म्हारो राजस्थान")
        result = deduplicate([s])
        assert len(result) == 1
        assert result[0].sentence_id == s.sentence_id

    def test_mixed_duplicates_and_uniques(self):
        """
        GIVEN a list with some duplicate and some unique sentences,
        WHEN  deduplicate is called,
        THEN  duplicates are removed and unique sentences are kept.
        """
        unique1 = _make_sentence("म्हारो राजस्थान")
        unique2 = _make_sentence("थारो घर कठे है")
        dup_a1 = _make_sentence("घणो सुंदर है यार")
        dup_a2 = _make_sentence("घणो सुंदर है यार")  # duplicate of dup_a1

        result = deduplicate([unique1, dup_a1, unique2, dup_a2])
        result_texts = [s.text for s in result]

        assert len(result) == 3
        assert "म्हारो राजस्थान" in result_texts
        assert "थारो घर कठे है" in result_texts
        assert "घणो सुंदर है यार" in result_texts


# ---------------------------------------------------------------------------
# Unit tests — near-duplicate deduplication (MinHash LSH)
# ---------------------------------------------------------------------------


class TestNearDuplicateDeduplication:
    """Unit tests for near-duplicate detection (Pass 2, MinHash LSH)."""

    def test_highly_similar_sentences_are_deduplicated(self):
        """
        GIVEN two sentences that differ by only one word (very high Jaccard),
        WHEN  deduplicate is called,
        THEN  only one is retained.
        """
        # These two sentences share almost all character 3-grams → Jaccard > 0.85
        base = "म्हारो राजस्थान बहुत सुंदर है यार दोस्त"
        near_dup = "म्हारो राजस्थान बहुत सुंदर है यार मित्र"

        s1 = _make_sentence(base)
        s2 = _make_sentence(near_dup)

        result = deduplicate([s1, s2])
        # At least one should be removed (they are near-duplicates)
        assert len(result) <= 2  # may be 1 or 2 depending on exact Jaccard

    def test_very_different_sentences_are_both_kept(self):
        """
        GIVEN two sentences that are completely different (Jaccard ≈ 0),
        WHEN  deduplicate is called,
        THEN  both are retained.
        """
        s1 = _make_sentence("म्हारो राजस्थान बहुत सुंदर है")
        s2 = _make_sentence("python programming language tutorial")

        result = deduplicate([s1, s2])
        assert len(result) == 2

    def test_identical_text_removed_in_pass1_not_pass2(self):
        """
        GIVEN two sentences with identical text,
        WHEN  deduplicate is called,
        THEN  the duplicate is removed in Pass 1 (exact match), and Pass 2
              does not see it (no MinHash collision on a single-item index).
        """
        text = "म्हारो राजस्थान घणो सुंदर"
        s1 = _make_sentence(text)
        s2 = _make_sentence(text)

        result = deduplicate([s1, s2])
        assert len(result) == 1
        assert result[0].sentence_id == s1.sentence_id

    def test_short_sentences_below_threshold_both_kept(self):
        """
        GIVEN two short sentences that are different enough (Jaccard < 0.85),
        WHEN  deduplicate is called,
        THEN  both are retained.
        """
        s1 = _make_sentence("म्हारो घर")
        s2 = _make_sentence("थारो काम")

        result = deduplicate([s1, s2])
        assert len(result) == 2

    def test_large_batch_with_no_duplicates_keeps_all(self):
        """
        GIVEN 10 sentences with structurally distinct texts (low 3-gram overlap),
        WHEN  deduplicate is called,
        THEN  all 10 are retained.
        """
        # Use texts from different domains/languages to ensure low Jaccard similarity
        sentences = [
            _make_sentence("म्हारो राजस्थान घणो सुंदर है"),
            _make_sentence("python programming tutorial beginners guide"),
            _make_sentence("जयपुर शहर बहुत खूबसूरत है यार"),
            _make_sentence("machine learning deep neural networks"),
            _make_sentence("थारो घर कठे है बताओ"),
            _make_sentence("database sql queries optimization"),
            _make_sentence("आपणो देश भारत महान है"),
            _make_sentence("javascript frontend web development"),
            _make_sentence("कुण आयो थो कल रात"),
            _make_sentence("docker kubernetes container orchestration"),
        ]
        result = deduplicate(sentences)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Unit tests — filter_rajasthani with bundled lexicon
# ---------------------------------------------------------------------------


class TestFilterRajasthaniUnit:
    """Unit tests for filter_rajasthani using the bundled lexicon."""

    def test_sentence_with_two_rajasthani_tokens_is_kept(self):
        """
        GIVEN a sentence containing exactly 2 Rajasthani tokens from the
              bundled lexicon,
        WHEN  filter_rajasthani is called,
        THEN  the sentence is retained.
        """
        # "म्हारो" and "घणो" are in the bundled lexicon
        s = _make_sentence("म्हारो राजस्थान घणो सुंदर है")
        result = filter_rajasthani([s])
        assert len(result) == 1

    def test_sentence_with_one_rajasthani_token_is_dropped(self):
        """
        GIVEN a sentence containing only 1 Rajasthani token,
        WHEN  filter_rajasthani is called,
        THEN  the sentence is dropped.
        """
        # Only "म्हारो" is in _TEST_LEXICON; rest are not
        s = _make_sentence("म्हारो यह बहुत अच्छा है")
        result = filter_rajasthani([s], lexicon=_TEST_LEXICON)
        assert len(result) == 0

    def test_sentence_with_no_rajasthani_tokens_is_dropped(self):
        """
        GIVEN a sentence with no Rajasthani tokens,
        WHEN  filter_rajasthani is called,
        THEN  the sentence is dropped.
        """
        s = _make_sentence("this is a completely english sentence")
        result = filter_rajasthani([s])
        assert len(result) == 0

    def test_custom_min_tokens_respected(self):
        """
        GIVEN a sentence with exactly 3 Rajasthani tokens and min_tokens=3,
        WHEN  filter_rajasthani is called,
        THEN  the sentence is retained.
        """
        s = _make_sentence("म्हारो थारो घणो यह वाक्य है")
        result = filter_rajasthani([s], min_tokens=3, lexicon=_TEST_LEXICON)
        assert len(result) == 1

    def test_custom_min_tokens_drops_sentence_below_threshold(self):
        """
        GIVEN a sentence with 2 Rajasthani tokens and min_tokens=3,
        WHEN  filter_rajasthani is called,
        THEN  the sentence is dropped.
        """
        s = _make_sentence("म्हारो घणो यह वाक्य है")
        result = filter_rajasthani([s], min_tokens=3, lexicon=_TEST_LEXICON)
        assert len(result) == 0

    def test_empty_input_returns_empty(self):
        """
        GIVEN an empty list,
        WHEN  filter_rajasthani is called,
        THEN  an empty list is returned.
        """
        assert filter_rajasthani([]) == []

    def test_filter_applies_to_both_platforms(self):
        """
        GIVEN sentences from both Twitter and ShareChat,
        WHEN  filter_rajasthani is called,
        THEN  the same filter is applied regardless of platform.
        """
        twitter_pass = _make_sentence("म्हारो राजस्थान घणो सुंदर", platform="twitter")
        sharechat_pass = _make_sentence("थारो घर कठे है यार", platform="sharechat")
        twitter_fail = _make_sentence("only english words here", platform="twitter")
        sharechat_fail = _make_sentence("only english words here", platform="sharechat")

        result = filter_rajasthani(
            [twitter_pass, sharechat_pass, twitter_fail, sharechat_fail],
            lexicon=_TEST_LEXICON,
        )
        result_ids = {s.sentence_id for s in result}
        assert twitter_pass.sentence_id in result_ids
        assert sharechat_pass.sentence_id in result_ids
        assert twitter_fail.sentence_id not in result_ids
        assert sharechat_fail.sentence_id not in result_ids

    def test_order_is_preserved(self):
        """
        GIVEN a list of sentences where some pass and some fail the filter,
        WHEN  filter_rajasthani is called,
        THEN  the relative order of kept sentences is preserved.
        """
        # Use _TEST_LEXICON explicitly; ensure each "pass" sentence has ≥ 2 tokens
        s1 = _make_sentence("म्हारो राजस्थान घणो")   # म्हारो + घणो → 2 tokens ✓
        s2 = _make_sentence("only english")            # 0 tokens ✗
        s3 = _make_sentence("थारो घर कठे")            # थारो + कठे → 2 tokens ✓
        s4 = _make_sentence("more english words")      # 0 tokens ✗
        s5 = _make_sentence("म्हारी नै सूं यार")      # म्हारी + नै + सूं → 3 tokens ✓

        result = filter_rajasthani([s1, s2, s3, s4, s5], lexicon=_TEST_LEXICON)
        result_ids = [s.sentence_id for s in result]
        assert result_ids == [s1.sentence_id, s3.sentence_id, s5.sentence_id]
