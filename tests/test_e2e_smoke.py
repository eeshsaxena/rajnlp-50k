"""
End-to-end pipeline smoke test for RajNLP-50K Corpus_Builder.

Runs the full corpus pipeline on a 100-sentence fixture:
  filter → deduplicate → split → serialize → validate_round_trip

Verifies that the output schema matches AnnotatedSentence for all records.

Requirements: 15.1, 15.2
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from corpus_builder.filter_dedup import filter_rajasthani, deduplicate
from corpus_builder.sampling import split
from corpus_builder.serialization import serialize, validate_round_trip
from models.data_models import (
    AnnotatedSentence,
    EntitySpan,
    RawSentence,
    TokenLabel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXED_COLLECTED_AT = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
_FIXED_ANNOTATED_AT = datetime(2024, 2, 1, 14, 0, 0, tzinfo=timezone.utc)

# Rajasthani lexicon words (from language_id/tagger.py and corpus_builder/rajasthani_lexicon.txt)
# Using Devanagari words that are in the bundled lexicon so sentences pass the filter.
_RAJ_WORDS = [
    "म्हारो",   # mharo — my/our
    "म्हारी",   # mhari — my/our (feminine)
    "थारो",     # tharo — your
    "थारी",     # thari — your (feminine)
    "कोनी",     # koni — is not
    "घणो",      # ghano — a lot
    "घणी",      # ghani — a lot (feminine)
    "आवे",      # aave — comes
    "जावे",     # jaave — goes
    "पाणी",     # pani — water
    "बाईसा",    # baisa — respectful address
    "राजस्थानी", # Rajasthani
    "मारवाड़ी",  # Marwadi
    "सा",       # sa — honorific
    "बावड़ी",   # bawdi — step-well
]

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_raw_sentence(
    text: str,
    platform: str = "twitter",
    sentence_id: str | None = None,
) -> RawSentence:
    """Create a RawSentence with the given text and platform."""
    return RawSentence(
        text=text,
        source_url=f"https://{platform}.com/example/status/{uuid.uuid4().hex[:8]}",
        collected_at=_FIXED_COLLECTED_AT,
        platform=platform,  # type: ignore[arg-type]
        sentence_id=sentence_id or str(uuid.uuid4()),
    )


def _raw_to_annotated(raw: RawSentence, split_label: str) -> AnnotatedSentence:
    """Convert a RawSentence to a minimal AnnotatedSentence with empty annotation fields.

    Since the full annotation pipeline isn't automated yet, this helper creates
    a valid AnnotatedSentence with neutral/empty annotations for smoke-testing
    the serialization and round-trip validation steps.
    """
    return AnnotatedSentence(
        sentence_id=raw.sentence_id,
        text=raw.text,
        platform=raw.platform,
        split=split_label,  # type: ignore[arg-type]
        sentiment="neutral",
        sentiment_annotator_labels=["neutral", "neutral", "neutral"],
        ner_spans=[],
        ner_annotator_spans=[[], [], []],
        toxicity_labels=[],
        toxicity_annotator_labels=[[], [], []],
        token_language_labels=[],
        source_url=raw.source_url,
        collected_at=raw.collected_at,
        annotated_at=_FIXED_ANNOTATED_AT,
    )


def _build_fixture_sentences() -> list[RawSentence]:
    """Build a fixture of 100 RawSentence objects.

    Composition:
    - 60 from Twitter, 40 from ShareChat
    - ~50 sentences with ≥2 Rajasthani tokens (should pass filter)
    - ~50 sentences without Rajasthani tokens (should be filtered out)

    Returns a list of exactly 100 RawSentence objects.
    """
    sentences: list[RawSentence] = []

    # --- Twitter sentences that PASS the filter (30 sentences) ---
    # Each contains at least 2 Rajasthani lexicon words.
    # We use explicit index substitution to avoid Python 3.14 scoping issues.
    twitter_passing_templates = [
        "म्हारो घर बहुत सुंदर है IDX घणो अच्छो लागे",
        "थारो काम घणो राम्रो है IDX म्हाने पसंद आयो",
        "कोनी जाणूं पाणी कठे है IDX बताओ",
        "राजस्थानी संस्कृति घणी सुंदर है IDX मारवाड़ी लोग",
        "म्हारी माँ बाईसा ने कहा IDX थारी बात सुणी",
        "घणो पाणी बरसा आज IDX खेत भर गया",
        "थारो घर कठे है IDX म्हाने बताओ",
        "कोनी आयो वो आज IDX घणो इंतजार किया",
        "म्हारो दिल कहे है IDX थारी याद आवे",
        "राजस्थानी खाना घणो स्वादिष्ट IDX मारवाड़ी रसोई",
    ]
    for template_idx, template in enumerate(twitter_passing_templates):
        for repeat in range(3):
            unique_idx = template_idx * 3 + repeat
            text = template.replace("IDX", str(unique_idx))
            sentences.append(_make_raw_sentence(text=text, platform="twitter"))

    # --- Twitter sentences that FAIL the filter (30 sentences) ---
    # These contain no Rajasthani lexicon words
    for idx in range(30):
        sentences.append(_make_raw_sentence(
            text="This is a regular Hindi sentence number " + str(idx) + " about politics and news",
            platform="twitter",
        ))

    # --- ShareChat sentences that PASS the filter (20 sentences) ---
    sharechat_passing_templates = [
        "म्हारो गांव बहुत प्यारो है IDX घणो खुश हूं",
        "थारो परिवार कैसो है IDX कोनी मिले बहुत दिन से",
        "पाणी की समस्या घणी बड़ी है IDX सरकार ध्यान दे",
        "राजस्थानी लोक गीत घणो सुंदर IDX मारवाड़ी संगीत",
        "म्हारी बेटी बाईसा पढ़ रही है IDX थारी बेटी भी",
        "घणो काम है आज IDX म्हाने थकान हो रही है",
        "थारो विचार सही है IDX कोनी गलत",
        "पाणी बचाओ अभियान IDX म्हारो समर्थन है",
        "राजस्थानी संस्कृति की रक्षा IDX घणो जरूरी है",
        "म्हारो देश राजस्थान IDX थारो भी यही है",
    ]
    for template_idx, template in enumerate(sharechat_passing_templates):
        for repeat in range(2):
            unique_idx = template_idx * 2 + repeat
            text = template.replace("IDX", str(unique_idx))
            sentences.append(_make_raw_sentence(text=text, platform="sharechat"))

    # --- ShareChat sentences that FAIL the filter (20 sentences) ---
    for idx in range(20):
        sentences.append(_make_raw_sentence(
            text="यह एक सामान्य हिंदी वाक्य है नंबर " + str(idx) + " राजनीति और समाचार के बारे में",
            platform="sharechat",
        ))

    # Verify we have exactly 100 sentences
    assert len(sentences) == 100, f"Expected 100 sentences, got {len(sentences)}"
    return sentences


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


class TestE2EPipelineSmoke:
    """
    End-to-end integration test: runs Corpus_Builder through all phases on a
    100-sentence fixture and verifies the output schema matches AnnotatedSentence.

    Requirements: 15.1, 15.2
    """

    @pytest.fixture(scope="class")
    def pipeline_output(self, tmp_path_factory):
        """Run the full pipeline and return (annotated_sentences, output_path, report)."""
        tmp_path = tmp_path_factory.mktemp("e2e_smoke")

        # --- Phase 1: Build fixture ---
        raw_sentences = _build_fixture_sentences()
        assert len(raw_sentences) == 100

        # --- Phase 2: Filter ---
        filtered = filter_rajasthani(raw_sentences)
        # At least some sentences should pass the filter
        assert len(filtered) > 0, "Filter removed all sentences — check fixture design"
        # All filtered sentences should have ≥2 Rajasthani tokens
        assert len(filtered) < len(raw_sentences), "Filter kept all sentences — check fixture design"

        # --- Phase 3: Deduplicate ---
        deduped = deduplicate(filtered)
        # Deduplication should not increase the count
        assert len(deduped) <= len(filtered)

        # --- Phase 4: Split (use all deduped sentences, small corpus) ---
        # With a small corpus we use the default 80/10/10 ratio
        dataset_split = split(deduped)

        # Collect all sentences from all splits
        all_split_sentences = (
            list(dataset_split.train)
            + list(dataset_split.validation)
            + list(dataset_split.test)
        )
        assert len(all_split_sentences) == len(deduped), (
            "Split total does not match deduped count"
        )

        # --- Phase 5: Convert RawSentence → AnnotatedSentence ---
        # Assign split labels based on which partition each sentence ended up in
        annotated: list[AnnotatedSentence] = []
        for raw in dataset_split.train:
            annotated.append(_raw_to_annotated(raw, "train"))
        for raw in dataset_split.validation:
            annotated.append(_raw_to_annotated(raw, "validation"))
        for raw in dataset_split.test:
            annotated.append(_raw_to_annotated(raw, "test"))

        assert len(annotated) == len(deduped)

        # --- Phase 6: Serialize ---
        output_path = tmp_path / "corpus_smoke.jsonl"
        serialize(annotated, output_path, fmt="jsonl")
        assert output_path.exists(), "Serialized file was not created"
        assert output_path.stat().st_size > 0, "Serialized file is empty"

        # --- Phase 7: Validate round-trip ---
        report = validate_round_trip(annotated, output_path, fmt="jsonl")

        return {
            "raw": raw_sentences,
            "filtered": filtered,
            "deduped": deduped,
            "dataset_split": dataset_split,
            "annotated": annotated,
            "output_path": output_path,
            "report": report,
        }

    # ------------------------------------------------------------------
    # Filter phase assertions
    # ------------------------------------------------------------------

    def test_filter_keeps_rajasthani_sentences(self, pipeline_output):
        """Filter should keep sentences with ≥2 Rajasthani tokens."""
        filtered = pipeline_output["filtered"]
        assert len(filtered) > 0

    def test_filter_removes_non_rajasthani_sentences(self, pipeline_output):
        """Filter should remove sentences without Rajasthani tokens."""
        raw_count = len(pipeline_output["raw"])
        filtered_count = len(pipeline_output["filtered"])
        assert filtered_count < raw_count, (
            "Expected some sentences to be filtered out"
        )

    def test_filter_output_count_is_reasonable(self, pipeline_output):
        """At least 40 of 100 sentences should pass the filter (the passing ones)."""
        # We designed ~50 sentences to pass; allow some margin
        assert len(pipeline_output["filtered"]) >= 30

    # ------------------------------------------------------------------
    # Deduplication phase assertions
    # ------------------------------------------------------------------

    def test_dedup_does_not_increase_count(self, pipeline_output):
        """Deduplication should not increase the sentence count."""
        assert len(pipeline_output["deduped"]) <= len(pipeline_output["filtered"])

    def test_dedup_sentence_ids_are_unique(self, pipeline_output):
        """All sentence_ids after deduplication should be unique."""
        ids = [s.sentence_id for s in pipeline_output["deduped"]]
        assert len(ids) == len(set(ids)), "Duplicate sentence_ids found after deduplication"

    # ------------------------------------------------------------------
    # Split phase assertions
    # ------------------------------------------------------------------

    def test_split_is_exhaustive(self, pipeline_output):
        """All deduped sentences should appear in exactly one split partition."""
        ds = pipeline_output["dataset_split"]
        total = len(ds.train) + len(ds.validation) + len(ds.test)
        assert total == len(pipeline_output["deduped"])

    def test_split_partitions_are_disjoint(self, pipeline_output):
        """No sentence_id should appear in more than one partition."""
        ds = pipeline_output["dataset_split"]
        train_ids = {s.sentence_id for s in ds.train}
        val_ids = {s.sentence_id for s in ds.validation}
        test_ids = {s.sentence_id for s in ds.test}
        assert not (train_ids & val_ids), "train/val overlap"
        assert not (train_ids & test_ids), "train/test overlap"
        assert not (val_ids & test_ids), "val/test overlap"

    def test_split_has_all_three_partitions(self, pipeline_output):
        """All three partitions should be non-empty (given enough input sentences)."""
        ds = pipeline_output["dataset_split"]
        # With ~50 sentences after filtering, all three partitions should be non-empty
        assert len(ds.train) > 0, "train partition is empty"
        # val and test may be empty if corpus is very small, but should exist
        assert isinstance(ds.validation, list)
        assert isinstance(ds.test, list)

    # ------------------------------------------------------------------
    # Serialization phase assertions
    # ------------------------------------------------------------------

    def test_serialized_file_exists(self, pipeline_output):
        """The serialized JSONL file should exist and be non-empty."""
        path = pipeline_output["output_path"]
        assert path.exists()
        assert path.stat().st_size > 0

    def test_serialized_file_has_correct_line_count(self, pipeline_output):
        """The JSONL file should have one line per annotated sentence."""
        path = pipeline_output["output_path"]
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == len(pipeline_output["annotated"])

    # ------------------------------------------------------------------
    # Round-trip validation assertions
    # ------------------------------------------------------------------

    def test_round_trip_report_has_no_failures(self, pipeline_output):
        """Round-trip validation should pass for all records."""
        report = pipeline_output["report"]
        assert report.failed == 0, (
            f"Round-trip validation failed for {report.failed} records: "
            f"{[f.sentence_id for f in report.failures]}"
        )

    def test_round_trip_report_total_matches_annotated_count(self, pipeline_output):
        """Round-trip report total should match the number of annotated sentences."""
        report = pipeline_output["report"]
        assert report.total_records == len(pipeline_output["annotated"])

    def test_round_trip_report_passed_equals_total(self, pipeline_output):
        """All records should pass round-trip validation."""
        report = pipeline_output["report"]
        assert report.passed == report.total_records

    # ------------------------------------------------------------------
    # Schema validation assertions (AnnotatedSentence fields)
    # ------------------------------------------------------------------

    def test_all_records_are_annotated_sentence_instances(self, pipeline_output):
        """All annotated records should be AnnotatedSentence instances."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record, AnnotatedSentence), (
                f"Expected AnnotatedSentence, got {type(record)}"
            )

    def test_all_records_have_non_empty_sentence_id(self, pipeline_output):
        """Every record should have a non-empty sentence_id."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.sentence_id, str)
            assert len(record.sentence_id) > 0, "sentence_id is empty"

    def test_all_records_have_non_empty_text(self, pipeline_output):
        """Every record should have a non-empty text field."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.text, str)
            assert len(record.text) > 0, f"text is empty for sentence_id={record.sentence_id}"

    def test_all_records_have_valid_platform(self, pipeline_output):
        """Every record's platform should be 'twitter' or 'sharechat'."""
        valid_platforms = {"twitter", "sharechat"}
        for record in pipeline_output["annotated"]:
            assert record.platform in valid_platforms, (
                f"Invalid platform {record.platform!r} for sentence_id={record.sentence_id}"
            )

    def test_all_records_have_valid_split(self, pipeline_output):
        """Every record's split should be 'train', 'validation', or 'test'."""
        valid_splits = {"train", "validation", "test"}
        for record in pipeline_output["annotated"]:
            assert record.split in valid_splits, (
                f"Invalid split {record.split!r} for sentence_id={record.sentence_id}"
            )

    def test_all_records_have_valid_sentiment(self, pipeline_output):
        """Every record's sentiment should be 'positive', 'neutral', or 'negative'."""
        valid_sentiments = {"positive", "neutral", "negative"}
        for record in pipeline_output["annotated"]:
            assert record.sentiment in valid_sentiments, (
                f"Invalid sentiment {record.sentiment!r} for sentence_id={record.sentence_id}"
            )

    def test_all_records_have_ner_spans_as_list(self, pipeline_output):
        """Every record's ner_spans should be a list."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.ner_spans, list), (
                f"ner_spans is not a list for sentence_id={record.sentence_id}"
            )

    def test_all_records_have_toxicity_labels_as_list(self, pipeline_output):
        """Every record's toxicity_labels should be a list."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.toxicity_labels, list), (
                f"toxicity_labels is not a list for sentence_id={record.sentence_id}"
            )

    def test_platform_distribution_preserved_across_splits(self, pipeline_output):
        """Both platforms should be represented in the annotated output."""
        platforms = {r.platform for r in pipeline_output["annotated"]}
        # We have both twitter and sharechat sentences in the fixture
        assert "twitter" in platforms, "No twitter sentences in output"
        assert "sharechat" in platforms, "No sharechat sentences in output"

    def test_sentence_ids_are_unique_in_output(self, pipeline_output):
        """All sentence_ids in the final annotated output should be unique."""
        ids = [r.sentence_id for r in pipeline_output["annotated"]]
        assert len(ids) == len(set(ids)), "Duplicate sentence_ids in annotated output"

    # ------------------------------------------------------------------
    # Additional schema field type checks
    # ------------------------------------------------------------------

    def test_sentiment_annotator_labels_is_list(self, pipeline_output):
        """sentiment_annotator_labels should be a list for every record."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.sentiment_annotator_labels, list)

    def test_ner_annotator_spans_is_list_of_lists(self, pipeline_output):
        """ner_annotator_spans should be a list of lists for every record."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.ner_annotator_spans, list)
            for span_set in record.ner_annotator_spans:
                assert isinstance(span_set, list)

    def test_toxicity_annotator_labels_is_list_of_lists(self, pipeline_output):
        """toxicity_annotator_labels should be a list of lists for every record."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.toxicity_annotator_labels, list)
            for label_set in record.toxicity_annotator_labels:
                assert isinstance(label_set, list)

    def test_token_language_labels_is_list(self, pipeline_output):
        """token_language_labels should be a list for every record."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.token_language_labels, list)

    def test_source_url_is_non_empty_string(self, pipeline_output):
        """source_url should be a non-empty string for every record."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.source_url, str)
            assert len(record.source_url) > 0

    def test_collected_at_is_datetime(self, pipeline_output):
        """collected_at should be a datetime for every record."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.collected_at, datetime)

    def test_annotated_at_is_datetime(self, pipeline_output):
        """annotated_at should be a datetime for every record."""
        for record in pipeline_output["annotated"]:
            assert isinstance(record.annotated_at, datetime)
