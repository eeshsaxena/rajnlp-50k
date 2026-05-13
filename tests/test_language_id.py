"""
Tests for language_id.tagger — Language_ID_Tagger.

Covers:
- Property 7: Language ID labels cover all tokens
  For any sentence tagged by the Language_ID_Tagger, the number of
  ``token_language_labels`` SHALL equal the number of whitespace-delimited
  tokens in the sentence, and every label SHALL be one of
  {"RAJ", "HIN", "ENG", "TRL"}.
  (Validates: Requirements 9.1)

- Unit tests for Language_ID_Tagger
  - Known Rajasthani tokens are labeled RAJ
  - Known English tokens are labeled ENG
  - Latin-script Rajasthani tokens are labeled TRL
  (Requirements: 9.1)

Requirements: 9.1, 9.4
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from language_id.tagger import LanguageIDTagger
from models.data_models import LangIDMetrics, TokenLabel

# ---------------------------------------------------------------------------
# Shared tagger instance
# ---------------------------------------------------------------------------

_TAGGER = LanguageIDTagger()

# ---------------------------------------------------------------------------
# Valid label set
# ---------------------------------------------------------------------------

_VALID_LABELS: frozenset[str] = frozenset({"RAJ", "HIN", "ENG", "TRL"})

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy: generate a single non-empty token (no whitespace)
_token_strategy = st.text(
    alphabet=st.characters(
        blacklist_categories=("Zs", "Cc", "Cf"),  # no whitespace or control chars
        blacklist_characters="\t\n\r\x0b\x0c",
    ),
    min_size=1,
    max_size=20,
).filter(lambda t: t.strip() != "")


# Strategy: generate a non-empty list of tokens, then join with spaces
@st.composite
def _sentence_strategy(draw) -> str:
    """Generate a non-empty sentence as a space-joined list of tokens."""
    tokens = draw(st.lists(_token_strategy, min_size=1, max_size=15))
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Property 7: Language ID labels cover all tokens
# ---------------------------------------------------------------------------


class TestProperty7LanguageIDCoverage:
    """
    Property 7: Language ID labels cover all tokens.

    For any sentence tagged by the Language_ID_Tagger:
    1. ``len(tagger.tag(sentence)) == len(sentence.split())``
    2. Every label in the result is one of {"RAJ", "HIN", "ENG", "TRL"}
    3. Every confidence value is in [0.0, 1.0]

    Validates: Requirements 9.1
    """

    @given(sentence=_sentence_strategy())
    @settings(max_examples=100)
    def test_label_count_equals_token_count(self, sentence: str) -> None:
        """
        GIVEN any non-empty sentence,
        WHEN  tagger.tag(sentence) is called,
        THEN  the number of returned TokenLabels equals len(sentence.split()).
        """
        result = _TAGGER.tag(sentence)
        expected_count = len(sentence.split())

        assert len(result) == expected_count, (
            f"Expected {expected_count} labels for sentence {sentence!r}, "
            f"got {len(result)}"
        )

    @given(sentence=_sentence_strategy())
    @settings(max_examples=100)
    def test_every_label_is_valid(self, sentence: str) -> None:
        """
        GIVEN any non-empty sentence,
        WHEN  tagger.tag(sentence) is called,
        THEN  every label in the result is one of {"RAJ", "HIN", "ENG", "TRL"}.
        """
        result = _TAGGER.tag(sentence)

        for token_label in result:
            assert token_label.label in _VALID_LABELS, (
                f"Invalid label {token_label.label!r} for token {token_label.token!r}; "
                f"must be one of {_VALID_LABELS}"
            )

    @given(sentence=_sentence_strategy())
    @settings(max_examples=100)
    def test_every_confidence_is_in_unit_interval(self, sentence: str) -> None:
        """
        GIVEN any non-empty sentence,
        WHEN  tagger.tag(sentence) is called,
        THEN  every confidence value is in [0.0, 1.0].
        """
        result = _TAGGER.tag(sentence)

        for token_label in result:
            assert 0.0 <= token_label.confidence <= 1.0, (
                f"Confidence {token_label.confidence} for token {token_label.token!r} "
                f"is outside [0.0, 1.0]"
            )

    @given(sentence=_sentence_strategy())
    @settings(max_examples=100)
    def test_token_field_matches_original_token(self, sentence: str) -> None:
        """
        GIVEN any non-empty sentence,
        WHEN  tagger.tag(sentence) is called,
        THEN  the ``token`` field of each TokenLabel matches the original token.
        """
        tokens = sentence.split()
        result = _TAGGER.tag(sentence)

        for original_token, token_label in zip(tokens, result):
            assert token_label.token == original_token, (
                f"TokenLabel.token {token_label.token!r} does not match "
                f"original token {original_token!r}"
            )

    @given(sentence=_sentence_strategy())
    @settings(max_examples=100)
    def test_result_is_list_of_token_labels(self, sentence: str) -> None:
        """
        GIVEN any non-empty sentence,
        WHEN  tagger.tag(sentence) is called,
        THEN  the result is a list of TokenLabel instances.
        """
        result = _TAGGER.tag(sentence)

        assert isinstance(result, list), (
            f"Expected list, got {type(result).__name__}"
        )
        for item in result:
            assert isinstance(item, TokenLabel), (
                f"Expected TokenLabel, got {type(item).__name__}"
            )


# ---------------------------------------------------------------------------
# Unit tests — Rajasthani tokens labeled RAJ
# ---------------------------------------------------------------------------


class TestRajasthaniTokensLabeledRAJ:
    """Verify known Rajasthani tokens are labeled RAJ (Requirement 9.1)."""

    _RAJASTHANI_TOKENS = [
        "म्हारो",   # mharo  — my / our
        "थारो",     # tharo  — your
        "केई",      # kei    — some / any
        "बावड़ी",   # bawdi  — step-well
        "म्हाने",   # mhane  — to me / us
        "कोनी",     # koni   — is not (Rajasthani negation)
        "घणो",      # ghano  — a lot / very
        "पाणी",     # pani   — water (Rajasthani variant)
        "आवे",      # aave   — comes
        "जावे",     # jaave  — goes
    ]

    @pytest.mark.parametrize("token", _RAJASTHANI_TOKENS)
    def test_rajasthani_token_is_labeled_raj(self, token: str) -> None:
        """
        GIVEN a known Rajasthani Devanagari token,
        WHEN  tagger.tag is called with that token as the sentence,
        THEN  the label is RAJ.
        """
        result = _TAGGER.tag(token)
        assert len(result) == 1
        assert result[0].label == "RAJ", (
            f"Expected RAJ for Rajasthani token {token!r}, got {result[0].label!r}"
        )

    def test_rajasthani_sentence_contains_raj_labels(self) -> None:
        """
        GIVEN a sentence with multiple known Rajasthani tokens,
        WHEN  tagger.tag is called,
        THEN  those tokens are labeled RAJ.
        """
        sentence = "म्हारो थारो घणो"
        result = _TAGGER.tag(sentence)
        assert len(result) == 3
        for token_label in result:
            assert token_label.label == "RAJ", (
                f"Expected RAJ for {token_label.token!r}, got {token_label.label!r}"
            )

    def test_rajasthani_token_confidence_is_high(self) -> None:
        """
        GIVEN a known Rajasthani token,
        WHEN  tagger.tag is called,
        THEN  the confidence is >= 0.85 (high confidence for lexicon match).
        """
        result = _TAGGER.tag("म्हारो")
        assert result[0].confidence >= 0.85, (
            f"Expected high confidence for RAJ token, got {result[0].confidence}"
        )


# ---------------------------------------------------------------------------
# Unit tests — Hindi tokens labeled HIN
# ---------------------------------------------------------------------------


class TestHindiTokensLabeledHIN:
    """Verify known Hindi tokens are labeled HIN (Requirement 9.1)."""

    _HINDI_TOKENS = [
        "है",   # is
        "और",   # and
        "का",   # of
        "में",  # in
        "को",   # to
    ]

    @pytest.mark.parametrize("token", _HINDI_TOKENS)
    def test_hindi_token_is_labeled_hin(self, token: str) -> None:
        """
        GIVEN a known Hindi Devanagari token (not in Rajasthani lexicon),
        WHEN  tagger.tag is called,
        THEN  the label is HIN.
        """
        result = _TAGGER.tag(token)
        assert len(result) == 1
        assert result[0].label == "HIN", (
            f"Expected HIN for Hindi token {token!r}, got {result[0].label!r}"
        )

    def test_devanagari_non_rajasthani_defaults_to_hin(self) -> None:
        """
        GIVEN a Devanagari token not in the Rajasthani lexicon,
        WHEN  tagger.tag is called,
        THEN  the label is HIN (default for unknown Devanagari).
        """
        # "राजनीति" (politics) is a common Hindi word, not Rajasthani-specific
        result = _TAGGER.tag("राजनीति")
        assert result[0].label == "HIN"


# ---------------------------------------------------------------------------
# Unit tests — English tokens labeled ENG
# ---------------------------------------------------------------------------


class TestEnglishTokensLabeledENG:
    """Verify known English tokens are labeled ENG (Requirement 9.1)."""

    _ENGLISH_TOKENS = [
        "the",
        "is",
        "and",
        "government",
        "election",
        "minister",
        "party",
        "state",
        "india",
        "news",
    ]

    @pytest.mark.parametrize("token", _ENGLISH_TOKENS)
    def test_english_token_is_labeled_eng(self, token: str) -> None:
        """
        GIVEN a known English Latin-script token,
        WHEN  tagger.tag is called,
        THEN  the label is ENG.
        """
        result = _TAGGER.tag(token)
        assert len(result) == 1
        assert result[0].label == "ENG", (
            f"Expected ENG for English token {token!r}, got {result[0].label!r}"
        )

    def test_english_token_case_insensitive(self) -> None:
        """
        GIVEN an English token in uppercase,
        WHEN  tagger.tag is called,
        THEN  the label is ENG (case-insensitive lookup).
        """
        result = _TAGGER.tag("The")
        assert result[0].label == "ENG", (
            f"Expected ENG for 'The' (case-insensitive), got {result[0].label!r}"
        )

    def test_english_sentence_all_eng(self) -> None:
        """
        GIVEN a sentence of known English words,
        WHEN  tagger.tag is called,
        THEN  all tokens are labeled ENG.
        """
        sentence = "the government and election"
        result = _TAGGER.tag(sentence)
        assert len(result) == 4
        for token_label in result:
            assert token_label.label == "ENG", (
                f"Expected ENG for {token_label.token!r}, got {token_label.label!r}"
            )

    def test_english_token_confidence_is_high(self) -> None:
        """
        GIVEN a known English token,
        WHEN  tagger.tag is called,
        THEN  the confidence is >= 0.90 (high confidence for lexicon match).
        """
        result = _TAGGER.tag("the")
        assert result[0].confidence >= 0.90, (
            f"Expected high confidence for ENG token, got {result[0].confidence}"
        )


# ---------------------------------------------------------------------------
# Unit tests — Transliterated tokens labeled TRL
# ---------------------------------------------------------------------------


class TestTransliteratedTokensLabeledTRL:
    """Verify Latin-script Rajasthani tokens are labeled TRL (Requirement 9.1)."""

    _TRL_TOKENS = [
        "mharo",      # म्हारो in Latin script
        "tharo",      # थारो in Latin script
        "rajasthan",  # Rajasthan (not in English lexicon)
        "jaipur",     # Jaipur (not in English lexicon)
        "ghano",      # घणो in Latin script
        "koni",       # कोनी in Latin script
        "paani",      # पाणी in Latin script
        "aave",       # आवे in Latin script
        "bawdi",      # बावड़ी in Latin script
        "marwadi",    # मारवाड़ी in Latin script
    ]

    @pytest.mark.parametrize("token", _TRL_TOKENS)
    def test_transliterated_token_is_labeled_trl(self, token: str) -> None:
        """
        GIVEN a Latin-script Rajasthani/Hindi token not in the English lexicon,
        WHEN  tagger.tag is called,
        THEN  the label is TRL (transliterated).
        """
        result = _TAGGER.tag(token)
        assert len(result) == 1
        assert result[0].label == "TRL", (
            f"Expected TRL for transliterated token {token!r}, got {result[0].label!r}"
        )

    def test_unknown_latin_token_defaults_to_trl(self) -> None:
        """
        GIVEN a Latin-script token not in the English lexicon,
        WHEN  tagger.tag is called,
        THEN  the label is TRL (default for unknown Latin tokens).
        """
        # "xyzabc" is not a real word in any lexicon
        result = _TAGGER.tag("xyzabc")
        assert result[0].label == "TRL"

    def test_mixed_sentence_with_trl_tokens(self) -> None:
        """
        GIVEN a sentence mixing transliterated and English tokens,
        WHEN  tagger.tag is called,
        THEN  transliterated tokens are TRL and English tokens are ENG.
        """
        # "mharo" is TRL, "the" is ENG, "rajasthan" is TRL
        sentence = "mharo the rajasthan"
        result = _TAGGER.tag(sentence)
        assert len(result) == 3
        assert result[0].label == "TRL", f"Expected TRL for 'mharo', got {result[0].label!r}"
        assert result[1].label == "ENG", f"Expected ENG for 'the', got {result[1].label!r}"
        assert result[2].label == "TRL", f"Expected TRL for 'rajasthan', got {result[2].label!r}"


# ---------------------------------------------------------------------------
# Unit tests — edge cases and general behaviour
# ---------------------------------------------------------------------------


class TestLanguageIDTaggerEdgeCases:
    """Edge case and general behaviour tests for LanguageIDTagger."""

    def test_empty_string_returns_empty_list(self) -> None:
        """
        GIVEN an empty string,
        WHEN  tagger.tag is called,
        THEN  an empty list is returned.
        """
        result = _TAGGER.tag("")
        assert result == []

    def test_whitespace_only_string_returns_empty_list(self) -> None:
        """
        GIVEN a whitespace-only string,
        WHEN  tagger.tag is called,
        THEN  an empty list is returned.
        """
        result = _TAGGER.tag("   \t  ")
        assert result == []

    def test_single_token_returns_one_label(self) -> None:
        """
        GIVEN a single-token sentence,
        WHEN  tagger.tag is called,
        THEN  exactly one TokenLabel is returned.
        """
        result = _TAGGER.tag("म्हारो")
        assert len(result) == 1

    def test_mixed_script_sentence_returns_correct_count(self) -> None:
        """
        GIVEN a code-switched sentence with Devanagari and Latin tokens,
        WHEN  tagger.tag is called,
        THEN  the number of labels equals the number of whitespace tokens.
        """
        sentence = "म्हारो government में है"
        tokens = sentence.split()
        result = _TAGGER.tag(sentence)
        assert len(result) == len(tokens)

    def test_result_token_fields_match_input_tokens(self) -> None:
        """
        GIVEN a sentence,
        WHEN  tagger.tag is called,
        THEN  the ``token`` field of each result matches the corresponding
              whitespace-split token from the input.
        """
        sentence = "म्हारो rajasthan the है"
        tokens = sentence.split()
        result = _TAGGER.tag(sentence)
        for original, labeled in zip(tokens, result):
            assert labeled.token == original

    def test_all_labels_are_valid_strings(self) -> None:
        """
        GIVEN a mixed-script sentence,
        WHEN  tagger.tag is called,
        THEN  all labels are valid string values from the allowed set.
        """
        sentence = "म्हारो mharo the है election"
        result = _TAGGER.tag(sentence)
        for token_label in result:
            assert token_label.label in _VALID_LABELS

    def test_evaluate_returns_lang_id_metrics(self) -> None:
        """
        GIVEN a non-empty test set,
        WHEN  tagger.evaluate is called,
        THEN  a LangIDMetrics instance is returned with valid fields.
        """
        from datetime import datetime, timezone

        from models.data_models import AnnotatedSentence, EntitySpan

        text = "म्हारो the है"
        sentence = AnnotatedSentence(
            sentence_id="test-001",
            text=text,
            platform="twitter",
            split="test",
            sentiment="neutral",
            sentiment_annotator_labels=["neutral", "neutral", "neutral"],
            ner_spans=[],
            ner_annotator_spans=[[], [], []],
            toxicity_labels=[],
            toxicity_annotator_labels=[[], [], []],
            token_language_labels=[
                TokenLabel(token="म्हारो", label="RAJ", confidence=0.90),
                TokenLabel(token="the", label="ENG", confidence=0.95),
                TokenLabel(token="है", label="HIN", confidence=0.80),
            ],
            source_url="https://example.com",
            collected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            annotated_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )

        metrics = _TAGGER.evaluate([sentence])

        assert isinstance(metrics, LangIDMetrics)
        assert 0.0 <= metrics.token_accuracy <= 1.0
        assert isinstance(metrics.per_class_f1, dict)
        for label in ("RAJ", "HIN", "ENG", "TRL"):
            assert label in metrics.per_class_f1
            assert 0.0 <= metrics.per_class_f1[label] <= 1.0

    def test_evaluate_empty_test_set_returns_zero_metrics(self) -> None:
        """
        GIVEN an empty test set,
        WHEN  tagger.evaluate is called,
        THEN  a LangIDMetrics with token_accuracy=0.0 is returned.
        """
        metrics = _TAGGER.evaluate([])
        assert metrics.token_accuracy == 0.0
        assert all(v == 0.0 for v in metrics.per_class_f1.values())

    def test_custom_lexicons_are_respected(self) -> None:
        """
        GIVEN a tagger initialised with a custom Rajasthani lexicon,
        WHEN  tagger.tag is called with a token in the custom lexicon,
        THEN  the token is labeled RAJ.
        """
        custom_raj = frozenset({"customword"})
        # "customword" is Latin, but we need a Devanagari token for RAJ
        # Use a custom Devanagari token not in the default lexicon
        custom_raj_dev = frozenset({"परीक्षण"})  # "parikshan" = test/examination
        tagger = LanguageIDTagger(rajasthani_lexicon=custom_raj_dev)
        result = tagger.tag("परीक्षण")
        assert result[0].label == "RAJ"

    def test_default_tagger_labels_unknown_devanagari_as_hin(self) -> None:
        """
        GIVEN a Devanagari token not in the Rajasthani lexicon,
        WHEN  tagger.tag is called with the default tagger,
        THEN  the token is labeled HIN.
        """
        # "परीक्षण" is not in the default Rajasthani lexicon
        result = _TAGGER.tag("परीक्षण")
        assert result[0].label == "HIN"
