"""
Unit tests for ShareChatCollector (corpus_builder.sharechat_collector).

Covers:
- HTTP 404 / error-page detection: verify URL is logged and scraper continues
  to the next page (Requirements 2.4, 2.5).
- Timeout handling: verify a single retry is attempted with 2× timeout; if
  the retry also times out the page is skipped and remaining pages are
  processed (Requirement 2.5).
- Successful collection: verify RawSentence fields are populated correctly
  (Requirements 2.1, 2.4).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest
from selenium.common.exceptions import TimeoutException, WebDriverException

from corpus_builder.sharechat_collector import (
    ShareChatCollector,
    _build_driver,
    _extract_texts,
    _is_error_page,
)
from models.data_models import RawSentence


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_collector(timeout: int = 30, headless: bool = True) -> ShareChatCollector:
    """Return a ShareChatCollector with default settings."""
    return ShareChatCollector(timeout=timeout, headless=headless)


def _make_driver_mock(
    title: str = "ShareChat - Rajasthan News",
    body_text: str = "म्हारो राजस्थान बहुत सुंदर है यार",
    post_texts: list[str] | None = None,
) -> MagicMock:
    """Return a mock WebDriver that simulates a successful page load.

    Args:
        title: Simulated page title.
        body_text: Simulated body text (used for error-page detection).
        post_texts: List of text strings returned by ``find_elements``.
    """
    if post_texts is None:
        post_texts = [body_text]

    driver = MagicMock()
    driver.title = title

    # Simulate body element
    body_el = MagicMock()
    body_el.text = body_text

    # Simulate post elements (CSS selector results)
    post_elements = []
    for text in post_texts:
        el = MagicMock()
        el.text = text
        post_elements.append(el)

    # find_element returns body; find_elements returns post elements
    driver.find_element.return_value = body_el
    driver.find_elements.return_value = post_elements

    return driver


def _make_error_driver_mock(status_code: str = "404") -> MagicMock:
    """Return a mock WebDriver that simulates an HTTP error page."""
    driver = MagicMock()
    driver.title = f"{status_code} Not Found"

    body_el = MagicMock()
    body_el.text = f"{status_code} Page Not Found - The page you requested does not exist."

    driver.find_element.return_value = body_el
    driver.find_elements.return_value = []

    return driver


# ---------------------------------------------------------------------------
# HTTP error page tests
# ---------------------------------------------------------------------------


class TestHttpErrorHandling:
    """Verify that HTTP error pages are detected, logged, and skipped."""

    def test_404_page_is_skipped_and_logged(self, caplog):
        """
        GIVEN a page whose title and body contain '404 Not Found',
        WHEN  collect_sharechat is called with that URL,
        THEN  the URL is logged as a warning and no sentences are returned
              for that page.
        """
        error_url = "https://sharechat.com/tag/nonexistent"
        error_driver = _make_error_driver_mock("404")

        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=error_driver,
        ):
            with caplog.at_level(logging.WARNING, logger="corpus_builder.sharechat_collector"):
                results = collector.collect_sharechat([error_url])

        assert results == [], "Expected no sentences from a 404 page"
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(error_url in msg for msg in warning_messages), (
            f"Expected the URL to appear in a warning log; got: {warning_messages}"
        )

    def test_404_page_scraper_continues_to_next_page(self, caplog):
        """
        GIVEN a first URL that returns a 404 error page and a second URL
              that returns valid content,
        WHEN  collect_sharechat is called with both URLs,
        THEN  the scraper skips the 404 page and returns sentences from the
              second page.
        """
        error_url = "https://sharechat.com/tag/nonexistent"
        good_url = "https://sharechat.com/tag/rajasthan"
        good_text = "राजस्थान की ताज़ा खबरें यहाँ पढ़ें"

        error_driver = _make_error_driver_mock("404")
        good_driver = _make_driver_mock(
            title="ShareChat - Rajasthan",
            body_text=good_text,
            post_texts=[good_text],
        )

        # The same driver instance is reused across URLs; we control its
        # behaviour by making get() switch the driver's state.
        combined_driver = MagicMock()
        combined_driver.title = "404 Not Found"

        body_404 = MagicMock()
        body_404.text = "404 Page Not Found"

        body_good = MagicMock()
        body_good.text = good_text

        good_el = MagicMock()
        good_el.text = good_text

        call_count = {"n": 0}

        def fake_get(url):
            call_count["n"] += 1
            if call_count["n"] == 1:
                combined_driver.title = "404 Not Found"
                combined_driver.find_element.return_value = body_404
                combined_driver.find_elements.return_value = []
            else:
                combined_driver.title = "ShareChat - Rajasthan"
                combined_driver.find_element.return_value = body_good
                combined_driver.find_elements.return_value = [good_el]

        combined_driver.get.side_effect = fake_get

        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=combined_driver,
        ):
            with caplog.at_level(logging.WARNING, logger="corpus_builder.sharechat_collector"):
                results = collector.collect_sharechat([error_url, good_url])

        assert len(results) >= 1, "Expected at least one sentence from the good page"
        assert all(r.source_url == good_url for r in results), (
            "All returned sentences should come from the good URL"
        )

    def test_404_does_not_raise_exception(self):
        """
        GIVEN a page that returns a 404 error,
        WHEN  collect_sharechat is called,
        THEN  no exception is raised.
        """
        error_driver = _make_error_driver_mock("404")
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=error_driver,
        ):
            # Should not raise
            results = collector.collect_sharechat(["https://sharechat.com/tag/nonexistent"])

        assert isinstance(results, list)

    def test_500_error_page_is_skipped_and_logged(self, caplog):
        """
        GIVEN a page whose title contains '500',
        WHEN  collect_sharechat is called,
        THEN  the URL is logged and the page is skipped.
        """
        error_url = "https://sharechat.com/tag/broken"
        error_driver = _make_error_driver_mock("500")
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=error_driver,
        ):
            with caplog.at_level(logging.WARNING, logger="corpus_builder.sharechat_collector"):
                results = collector.collect_sharechat([error_url])

        assert results == []
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(error_url in msg for msg in warning_messages)


# ---------------------------------------------------------------------------
# Timeout handling tests
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Verify that timeouts trigger a single retry with 2× timeout, then skip."""

    def test_timeout_triggers_single_retry(self):
        """
        GIVEN a page that raises TimeoutException on the first load attempt,
        WHEN  collect_sharechat is called,
        THEN  the driver.get method is called exactly twice (original + retry).
        """
        timeout_url = "https://sharechat.com/tag/slow"
        good_text = "राजस्थान समाचार"

        driver = MagicMock()
        driver.title = "ShareChat"

        body_el = MagicMock()
        body_el.text = good_text
        driver.find_element.return_value = body_el

        good_el = MagicMock()
        good_el.text = good_text
        driver.find_elements.return_value = [good_el]

        # First get() raises TimeoutException; second succeeds
        driver.get.side_effect = [TimeoutException("Timed out"), None]

        collector = _make_collector(timeout=30)

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat([timeout_url])

        assert driver.get.call_count == 2, (
            f"Expected exactly 2 get() calls (original + retry); got {driver.get.call_count}"
        )

    def test_timeout_retry_uses_double_timeout(self):
        """
        GIVEN a page that raises TimeoutException on the first load,
        WHEN  the retry is attempted,
        THEN  set_page_load_timeout is called with 2× the original timeout.
        """
        timeout_url = "https://sharechat.com/tag/slow"
        original_timeout = 30

        driver = MagicMock()
        driver.title = "ShareChat"

        body_el = MagicMock()
        body_el.text = "राजस्थान समाचार"
        driver.find_element.return_value = body_el

        good_el = MagicMock()
        good_el.text = "राजस्थान समाचार"
        driver.find_elements.return_value = [good_el]

        driver.get.side_effect = [TimeoutException("Timed out"), None]

        collector = _make_collector(timeout=original_timeout)

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            collector.collect_sharechat([timeout_url])

        timeout_calls = driver.set_page_load_timeout.call_args_list
        assert len(timeout_calls) == 2, (
            f"Expected 2 set_page_load_timeout calls; got {timeout_calls}"
        )
        first_timeout = timeout_calls[0][0][0]
        second_timeout = timeout_calls[1][0][0]
        assert first_timeout == original_timeout
        assert second_timeout == original_timeout * 2

    def test_double_timeout_skips_page(self, caplog):
        """
        GIVEN a page that raises TimeoutException on both the original and
              retry attempts,
        WHEN  collect_sharechat is called,
        THEN  the page is skipped (no sentences returned for it) and a
              warning is logged.
        """
        timeout_url = "https://sharechat.com/tag/very-slow"

        driver = MagicMock()
        driver.get.side_effect = TimeoutException("Always times out")

        collector = _make_collector(timeout=10)

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            with caplog.at_level(logging.WARNING, logger="corpus_builder.sharechat_collector"):
                results = collector.collect_sharechat([timeout_url])

        assert results == [], "Expected no sentences when both attempts time out"
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(timeout_url in msg for msg in warning_messages), (
            f"Expected the URL to appear in a warning log; got: {warning_messages}"
        )

    def test_timeout_on_first_url_continues_to_second(self):
        """
        GIVEN a first URL that always times out and a second URL that loads
              successfully,
        WHEN  collect_sharechat is called with both URLs,
        THEN  sentences from the second URL are returned.
        """
        timeout_url = "https://sharechat.com/tag/slow"
        good_url = "https://sharechat.com/tag/rajasthan"
        good_text = "राजस्थान की ताज़ा खबरें"

        driver = MagicMock()

        body_el = MagicMock()
        body_el.text = good_text
        driver.find_element.return_value = body_el

        good_el = MagicMock()
        good_el.text = good_text
        driver.find_elements.return_value = [good_el]

        call_count = {"n": 0}

        def fake_get(url):
            call_count["n"] += 1
            if url == timeout_url:
                raise TimeoutException("Timed out")
            # Good URL: set title to non-error
            driver.title = "ShareChat - Rajasthan"

        driver.get.side_effect = fake_get
        driver.title = "ShareChat"

        collector = _make_collector(timeout=5)

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat([timeout_url, good_url])

        assert len(results) >= 1
        assert all(r.source_url == good_url for r in results)

    def test_timeout_does_not_raise_exception(self):
        """
        GIVEN a page that always times out,
        WHEN  collect_sharechat is called,
        THEN  no exception is raised.
        """
        driver = MagicMock()
        driver.get.side_effect = TimeoutException("Always times out")

        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat(["https://sharechat.com/tag/slow"])

        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Successful collection tests
# ---------------------------------------------------------------------------


class TestSuccessfulCollection:
    """Verify that RawSentence fields are populated correctly on a successful scrape."""

    def test_returns_raw_sentence_objects(self):
        """
        GIVEN a page that loads successfully with post text,
        WHEN  collect_sharechat is called,
        THEN  the result contains RawSentence instances.
        """
        text = "राजस्थान की ताज़ा खबरें यहाँ पढ़ें"
        driver = _make_driver_mock(post_texts=[text])
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat(["https://sharechat.com/tag/rajasthan"])

        assert len(results) >= 1
        assert all(isinstance(r, RawSentence) for r in results)

    def test_platform_is_sharechat(self):
        """
        GIVEN a successful page load,
        WHEN  collect_sharechat returns results,
        THEN  every RawSentence.platform is "sharechat".
        """
        texts = ["राजस्थान समाचार", "जयपुर की खबरें", "राजनीति की ताज़ा जानकारी"]
        driver = _make_driver_mock(post_texts=texts)
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat(["https://sharechat.com/tag/rajasthan"])

        assert all(r.platform == "sharechat" for r in results)

    def test_source_url_matches_page_url(self):
        """
        GIVEN a page at a specific URL,
        WHEN  collect_sharechat returns results,
        THEN  every RawSentence.source_url equals the page URL.
        """
        page_url = "https://sharechat.com/tag/rajasthan"
        text = "राजस्थान की ताज़ा खबरें"
        driver = _make_driver_mock(post_texts=[text])
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat([page_url])

        assert all(r.source_url == page_url for r in results)

    def test_collected_at_is_utc_datetime(self):
        """
        GIVEN a successful page load,
        WHEN  collect_sharechat returns results,
        THEN  every RawSentence.collected_at is a timezone-aware UTC datetime.
        """
        text = "राजस्थान की ताज़ा खबरें"
        driver = _make_driver_mock(post_texts=[text])
        collector = _make_collector()

        before = datetime.now(tz=timezone.utc)

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat(["https://sharechat.com/tag/rajasthan"])

        after = datetime.now(tz=timezone.utc)

        for r in results:
            assert r.collected_at.tzinfo is not None, "collected_at must be timezone-aware"
            assert r.collected_at.utcoffset().total_seconds() == 0, (
                "collected_at must be UTC"
            )
            assert before <= r.collected_at <= after

    def test_sentence_id_is_valid_uuid(self):
        """
        GIVEN a successful page load,
        WHEN  collect_sharechat returns results,
        THEN  every RawSentence.sentence_id is a valid UUID string.
        """
        text = "राजस्थान की ताज़ा खबरें"
        driver = _make_driver_mock(post_texts=[text])
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat(["https://sharechat.com/tag/rajasthan"])

        for r in results:
            parsed = uuid.UUID(r.sentence_id)  # raises ValueError if invalid
            assert str(parsed) == r.sentence_id

    def test_each_sentence_has_unique_id(self):
        """
        GIVEN a page with multiple text snippets,
        WHEN  collect_sharechat returns results,
        THEN  every RawSentence has a distinct sentence_id.
        """
        texts = [
            "राजस्थान की ताज़ा खबरें",
            "जयपुर में बड़ा ऐलान",
            "अशोक गहलोत ने कहा",
        ]
        driver = _make_driver_mock(post_texts=texts)
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat(["https://sharechat.com/tag/rajasthan"])

        ids = [r.sentence_id for r in results]
        assert len(ids) == len(set(ids)), "sentence_ids must be unique"

    def test_text_matches_page_content(self):
        """
        GIVEN a page with a specific text snippet,
        WHEN  collect_sharechat returns results,
        THEN  the RawSentence.text matches the extracted text.
        """
        expected_text = "राजस्थान की ताज़ा खबरें यहाँ पढ़ें"
        driver = _make_driver_mock(post_texts=[expected_text])
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat(["https://sharechat.com/tag/rajasthan"])

        assert len(results) >= 1
        assert results[0].text == expected_text

    def test_empty_page_urls_returns_empty_list(self):
        """
        GIVEN an empty page_urls list,
        WHEN  collect_sharechat is called,
        THEN  an empty list is returned without creating a WebDriver.
        """
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver"
        ) as mock_build:
            results = collector.collect_sharechat([])

        assert results == []
        mock_build.assert_not_called()

    def test_multiple_urls_aggregates_results(self):
        """
        GIVEN two pages each with valid text content,
        WHEN  collect_sharechat is called with both URLs,
        THEN  sentences from both pages are returned.
        """
        url1 = "https://sharechat.com/tag/rajasthan"
        url2 = "https://sharechat.com/tag/jaipur"
        text1 = "राजस्थान की ताज़ा खबरें"
        text2 = "जयपुर में बड़ा ऐलान हुआ"

        driver = MagicMock()
        driver.title = "ShareChat"

        body_el = MagicMock()
        driver.find_element.return_value = body_el

        call_count = {"n": 0}

        def fake_get(url):
            call_count["n"] += 1
            if url == url1:
                body_el.text = text1
                el = MagicMock()
                el.text = text1
                driver.find_elements.return_value = [el]
            else:
                body_el.text = text2
                el = MagicMock()
                el.text = text2
                driver.find_elements.return_value = [el]

        driver.get.side_effect = fake_get

        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat([url1, url2])

        assert len(results) >= 2
        source_urls = {r.source_url for r in results}
        assert url1 in source_urls
        assert url2 in source_urls

    def test_driver_is_closed_after_collection(self):
        """
        GIVEN a successful collection run,
        WHEN  collect_sharechat completes,
        THEN  driver.quit() is called exactly once.
        """
        text = "राजस्थान की ताज़ा खबरें"
        driver = _make_driver_mock(post_texts=[text])
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            collector.collect_sharechat(["https://sharechat.com/tag/rajasthan"])

        driver.quit.assert_called_once()

    def test_driver_is_closed_even_on_error(self):
        """
        GIVEN a page that raises an unexpected WebDriverException,
        WHEN  collect_sharechat is called,
        THEN  driver.quit() is still called (cleanup via try/finally).
        """
        driver = MagicMock()
        driver.get.side_effect = WebDriverException("Unexpected error")
        collector = _make_collector()

        with patch(
            "corpus_builder.sharechat_collector._build_driver",
            return_value=driver,
        ):
            results = collector.collect_sharechat(["https://sharechat.com/tag/rajasthan"])

        driver.quit.assert_called_once()
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# _is_error_page unit tests
# ---------------------------------------------------------------------------


class TestIsErrorPage:
    """Unit tests for the _is_error_page helper function."""

    def test_404_in_title_returns_true(self):
        driver = MagicMock()
        driver.title = "404 Not Found"
        body_el = MagicMock()
        body_el.text = "The page you requested does not exist."
        driver.find_element.return_value = body_el
        assert _is_error_page(driver) is True

    def test_normal_title_returns_false(self):
        driver = MagicMock()
        driver.title = "ShareChat - Rajasthan News"
        body_el = MagicMock()
        body_el.text = "राजस्थान की ताज़ा खबरें"
        driver.find_element.return_value = body_el
        assert _is_error_page(driver) is False

    def test_error_in_body_returns_true(self):
        driver = MagicMock()
        driver.title = "ShareChat"
        body_el = MagicMock()
        body_el.text = "404 page not found - this content is unavailable"
        driver.find_element.return_value = body_el
        assert _is_error_page(driver) is True

    def test_webdriver_exception_returns_false(self):
        """If the driver raises an exception, _is_error_page should return False."""
        driver = MagicMock()
        driver.title = "ShareChat"
        driver.find_element.side_effect = WebDriverException("stale element")
        # Should not raise; returns False
        assert _is_error_page(driver) is False
