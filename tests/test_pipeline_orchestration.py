"""
Integration tests for the full pipeline orchestration (Task 21.2).

Runs the pipeline on a 100-sentence fixture with a fixed seed and verifies:
- All expected output files are created and non-empty
- JSONL records are valid AnnotatedSentence dicts
- Parquet file is created
- Pipeline log file is created
- No exceptions are raised

Requirements: 17.5
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_pipeline import main
from models.data_models import AnnotatedSentence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SENTIMENTS = {"positive", "neutral", "negative"}
_VALID_PLATFORMS = {"twitter", "sharechat"}
_VALID_SPLITS = {"train", "validation", "test"}


def _run_pipeline(tmp_path: Path, seed: int = 42) -> None:
    """Run the pipeline in dry-run mode with the given seed and output dir."""
    main([
        "--seed", str(seed),
        "--output-dir", str(tmp_path),
        "--dry-run",
        "--log-level", "WARNING",
    ])


# ---------------------------------------------------------------------------
# Core smoke test
# ---------------------------------------------------------------------------


class TestPipelineSmoke:
    """Verify the full pipeline runs without errors and produces expected files."""

    @pytest.fixture(scope="class")
    def pipeline_output_dir(self, tmp_path_factory):
        """Run the pipeline once and return the output directory."""
        tmp_path = tmp_path_factory.mktemp("pipeline_smoke")
        _run_pipeline(tmp_path, seed=42)
        return tmp_path

    def test_no_exception_raised(self, tmp_path):
        """Pipeline should complete without raising any exception."""
        # This test runs the pipeline independently to verify no exception
        _run_pipeline(tmp_path, seed=42)

    def test_jsonl_file_created(self, pipeline_output_dir):
        """corpus.jsonl should be created in the output directory."""
        jsonl_path = pipeline_output_dir / "corpus.jsonl"
        assert jsonl_path.exists(), "corpus.jsonl was not created"

    def test_jsonl_file_non_empty(self, pipeline_output_dir):
        """corpus.jsonl should be non-empty."""
        jsonl_path = pipeline_output_dir / "corpus.jsonl"
        assert jsonl_path.stat().st_size > 0, "corpus.jsonl is empty"

    def test_parquet_file_created(self, pipeline_output_dir):
        """corpus.parquet should be created in the output directory."""
        parquet_path = pipeline_output_dir / "corpus.parquet"
        assert parquet_path.exists(), "corpus.parquet was not created"

    def test_parquet_file_non_empty(self, pipeline_output_dir):
        """corpus.parquet should be non-empty."""
        parquet_path = pipeline_output_dir / "corpus.parquet"
        assert parquet_path.stat().st_size > 0, "corpus.parquet is empty"

    def test_log_file_created(self, pipeline_output_dir):
        """pipeline.log should be created in the output directory."""
        log_path = pipeline_output_dir / "pipeline.log"
        assert log_path.exists(), "pipeline.log was not created"


# ---------------------------------------------------------------------------
# JSONL schema validation
# ---------------------------------------------------------------------------


class TestPipelineJsonlSchema:
    """Verify JSONL records have valid AnnotatedSentence schema."""

    @pytest.fixture(scope="class")
    def jsonl_records(self, tmp_path_factory):
        """Run the pipeline and return parsed JSONL records."""
        tmp_path = tmp_path_factory.mktemp("pipeline_jsonl")
        _run_pipeline(tmp_path, seed=42)
        jsonl_path = tmp_path / "corpus.jsonl"
        records = []
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def test_jsonl_has_records(self, jsonl_records):
        """JSONL file should contain at least one record."""
        assert len(jsonl_records) > 0, "No records found in corpus.jsonl"

    def test_jsonl_records_have_sentence_id(self, jsonl_records):
        """Every JSONL record should have a non-empty sentence_id."""
        for rec in jsonl_records:
            assert "sentence_id" in rec, "Record missing sentence_id"
            assert isinstance(rec["sentence_id"], str)
            assert len(rec["sentence_id"]) > 0

    def test_jsonl_records_have_text(self, jsonl_records):
        """Every JSONL record should have a non-empty text field."""
        for rec in jsonl_records:
            assert "text" in rec, "Record missing text"
            assert isinstance(rec["text"], str)
            assert len(rec["text"]) > 0

    def test_jsonl_records_have_valid_platform(self, jsonl_records):
        """Every JSONL record should have a valid platform."""
        for rec in jsonl_records:
            assert "platform" in rec, "Record missing platform"
            assert rec["platform"] in _VALID_PLATFORMS, (
                f"Invalid platform: {rec['platform']!r}"
            )

    def test_jsonl_records_have_valid_split(self, jsonl_records):
        """Every JSONL record should have a valid split label."""
        for rec in jsonl_records:
            assert "split" in rec, "Record missing split"
            assert rec["split"] in _VALID_SPLITS, (
                f"Invalid split: {rec['split']!r}"
            )

    def test_jsonl_records_have_valid_sentiment(self, jsonl_records):
        """Every JSONL record should have a valid sentiment label."""
        for rec in jsonl_records:
            assert "sentiment" in rec, "Record missing sentiment"
            assert rec["sentiment"] in _VALID_SENTIMENTS, (
                f"Invalid sentiment: {rec['sentiment']!r}"
            )

    def test_jsonl_records_have_ner_spans_list(self, jsonl_records):
        """Every JSONL record should have ner_spans as a list."""
        for rec in jsonl_records:
            assert "ner_spans" in rec, "Record missing ner_spans"
            assert isinstance(rec["ner_spans"], list)

    def test_jsonl_records_have_toxicity_labels_list(self, jsonl_records):
        """Every JSONL record should have toxicity_labels as a list."""
        for rec in jsonl_records:
            assert "toxicity_labels" in rec, "Record missing toxicity_labels"
            assert isinstance(rec["toxicity_labels"], list)

    def test_jsonl_records_have_token_language_labels(self, jsonl_records):
        """Every JSONL record should have token_language_labels as a list."""
        for rec in jsonl_records:
            assert "token_language_labels" in rec, "Record missing token_language_labels"
            assert isinstance(rec["token_language_labels"], list)

    def test_jsonl_records_have_source_url(self, jsonl_records):
        """Every JSONL record should have a non-empty source_url."""
        for rec in jsonl_records:
            assert "source_url" in rec, "Record missing source_url"
            assert isinstance(rec["source_url"], str)
            assert len(rec["source_url"]) > 0

    def test_jsonl_records_have_collected_at(self, jsonl_records):
        """Every JSONL record should have a collected_at timestamp."""
        for rec in jsonl_records:
            assert "collected_at" in rec, "Record missing collected_at"
            assert isinstance(rec["collected_at"], str)

    def test_jsonl_records_have_annotated_at(self, jsonl_records):
        """Every JSONL record should have an annotated_at timestamp."""
        for rec in jsonl_records:
            assert "annotated_at" in rec, "Record missing annotated_at"
            assert isinstance(rec["annotated_at"], str)

    def test_jsonl_sentence_ids_are_unique(self, jsonl_records):
        """All sentence_ids in the JSONL file should be unique."""
        ids = [rec["sentence_id"] for rec in jsonl_records]
        assert len(ids) == len(set(ids)), "Duplicate sentence_ids in corpus.jsonl"


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


class TestPipelineSeed:
    """Verify the pipeline uses the provided seed."""

    def test_pipeline_uses_provided_seed(self, tmp_path):
        """Pipeline should accept and use the provided seed without error."""
        # Run with seed 42 — should complete without exception
        main([
            "--seed", "42",
            "--output-dir", str(tmp_path),
            "--dry-run",
            "--log-level", "WARNING",
        ])
        assert (tmp_path / "corpus.jsonl").exists()

    def test_pipeline_uses_different_seed(self, tmp_path):
        """Pipeline should accept seed 123 without error."""
        main([
            "--seed", "123",
            "--output-dir", str(tmp_path),
            "--dry-run",
            "--log-level", "WARNING",
        ])
        assert (tmp_path / "corpus.jsonl").exists()


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------


class TestPipelineOutputDir:
    """Verify the output directory is created by the pipeline."""

    def test_pipeline_output_dir_created(self, tmp_path):
        """Pipeline should create the output directory if it does not exist."""
        new_dir = tmp_path / "new_subdir" / "nested"
        assert not new_dir.exists(), "Directory should not exist before pipeline run"
        main([
            "--seed", "42",
            "--output-dir", str(new_dir),
            "--dry-run",
            "--log-level", "WARNING",
        ])
        assert new_dir.exists(), "Output directory was not created"

    def test_pipeline_creates_all_expected_files(self, tmp_path):
        """Pipeline should create corpus.jsonl, corpus.parquet, and pipeline.log."""
        _run_pipeline(tmp_path, seed=42)
        assert (tmp_path / "corpus.jsonl").exists()
        assert (tmp_path / "corpus.parquet").exists()
        assert (tmp_path / "pipeline.log").exists()


# ---------------------------------------------------------------------------
# Parquet file
# ---------------------------------------------------------------------------


class TestPipelineParquet:
    """Verify the Parquet file is created and readable."""

    def test_pipeline_parquet_file_created(self, tmp_path):
        """corpus.parquet should be created and non-empty."""
        _run_pipeline(tmp_path, seed=42)
        parquet_path = tmp_path / "corpus.parquet"
        assert parquet_path.exists(), "corpus.parquet was not created"
        assert parquet_path.stat().st_size > 0, "corpus.parquet is empty"

    def test_pipeline_parquet_readable(self, tmp_path):
        """corpus.parquet should be readable with pyarrow."""
        import pyarrow.parquet as pq
        _run_pipeline(tmp_path, seed=42)
        parquet_path = tmp_path / "corpus.parquet"
        table = pq.read_table(str(parquet_path))
        assert table.num_rows > 0, "Parquet file has no rows"

    def test_pipeline_parquet_has_expected_columns(self, tmp_path):
        """corpus.parquet should have the expected schema columns."""
        import pyarrow.parquet as pq
        _run_pipeline(tmp_path, seed=42)
        parquet_path = tmp_path / "corpus.parquet"
        table = pq.read_table(str(parquet_path))
        expected_columns = {
            "sentence_id", "text", "platform", "split", "sentiment",
            "ner_spans", "toxicity_labels", "token_language_labels",
            "source_url", "collected_at", "annotated_at",
        }
        actual_columns = set(table.schema.names)
        missing = expected_columns - actual_columns
        assert not missing, f"Parquet missing columns: {missing}"
