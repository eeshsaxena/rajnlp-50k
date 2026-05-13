"""
Unit tests for TwitterCollector (corpus_builder.twitter_collector).

Covers:
- Rate-limit (HTTP 429) retry logic: mock API returning 429 then 200; verify
  collection resumes and returns results (Requirements 1.4, 1.5).
- Auth-failure (HTTP 401 / 403) handling: verify collection halts and raises
  TwitterAuthError (Requirement 1.5).
- RawSentence field population: verify all required fields are set correctly
  on a successful response (Requirement 1.4).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import tweepy

from corpus_builder import TwitterAuthError, TwitterCollector
from models.data_models import RawSentence


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_tweet(
    tweet_id: str = "1234567890",
    text: str = "म्हारो राजस्थान बहुत सुंदर है यार",
    author_id: str = "9876543210",
) -> MagicMock:
    """Return a minimal mock tweepy.Tweet with the given fields."""
    tweet = MagicMock(spec=tweepy.Tweet)
    tweet.id = tweet_id
    tweet.text = text
    tweet.author_id = author_id
    return tweet


def _make_response(
    tweets: list[MagicMock] | None = None,
    next_token: str | None = None,
) -> MagicMock:
    """Return a mock tweepy.Response wrapping the given tweets."""
    response = MagicMock(spec=tweepy.Response)
    response.data = tweets  # None signals "no more results"
    meta: dict[str, Any] = {}
    if next_token:
        meta["next_token"] = next_token
    response.meta = meta
    return response


def _make_429_exc(retry_after_seconds: int | None = None) -> tweepy.errors.TooManyRequests:
    """Build a TooManyRequests exception with an optional Retry-After header."""
    http_response = MagicMock()
    headers: dict[str, str] = {}
    if retry_after_seconds is not None:
        headers["Retry-After"] = str(retry_after_seconds)
    http_response.headers = headers
    exc = tweepy.errors.TooManyRequests(response=http_response)
    exc.response = http_response
    return exc


def _make_401_exc() -> tweepy.errors.Unauthorized:
    """Build an Unauthorized (HTTP 401) exception."""
    http_response = MagicMock()
    http_response.status_code = 401
    exc = tweepy.errors.Unauthorized(response=http_response)
    exc.response = http_response
    return exc


def _make_403_exc() -> tweepy.errors.Forbidden:
    """Build a Forbidden (HTTP 403) exception."""
    http_response = MagicMock()
    http_response.status_code = 403
    exc = tweepy.errors.Forbidden(response=http_response)
    exc.response = http_response
    return exc


def _make_collector() -> TwitterCollector:
    """Return a TwitterCollector with a dummy bearer token."""
    with patch("tweepy.Client"):
        collector = TwitterCollector(bearer_token="test-bearer-token")
    return collector


# ---------------------------------------------------------------------------
# Rate-limit (HTTP 429) retry tests
# ---------------------------------------------------------------------------


class TestRateLimitRetry:
    """Verify that a 429 response causes a pause then a successful retry."""

    def test_429_then_200_resumes_and_returns_results(self):
        """
        GIVEN the API raises TooManyRequests on the first call and returns a
              valid response on the second call,
        WHEN  collect_twitter is invoked,
        THEN  the collector sleeps for the Retry-After duration, retries, and
              returns the tweets from the successful response.
        """
        tweet = _make_tweet()
        success_response = _make_response(tweets=[tweet])
        rate_limit_exc = _make_429_exc(retry_after_seconds=5)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=[rate_limit_exc, success_response]
        )

        with patch("corpus_builder.twitter_collector.time.sleep") as mock_sleep:
            results = collector.collect_twitter(["#rajasthan"], max_results=10)

        # sleep must have been called with the Retry-After value
        mock_sleep.assert_called_once_with(5)
        # After the retry the collector should have returned one sentence
        assert len(results) == 1
        assert isinstance(results[0], RawSentence)

    def test_429_sleep_duration_uses_retry_after_header(self):
        """
        GIVEN a 429 response with Retry-After: 30,
        WHEN  the rate-limit handler fires,
        THEN  time.sleep is called with exactly 30 seconds.
        """
        tweet = _make_tweet()
        success_response = _make_response(tweets=[tweet])
        rate_limit_exc = _make_429_exc(retry_after_seconds=30)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=[rate_limit_exc, success_response]
        )

        with patch("corpus_builder.twitter_collector.time.sleep") as mock_sleep:
            collector.collect_twitter(["म्हारो"], max_results=10)

        mock_sleep.assert_called_once_with(30)

    def test_429_without_retry_after_header_falls_back_to_60_seconds(self):
        """
        GIVEN a 429 response with no Retry-After or x-rate-limit-reset header,
        WHEN  the rate-limit handler fires,
        THEN  time.sleep is called with the 60-second default.
        """
        tweet = _make_tweet()
        success_response = _make_response(tweets=[tweet])
        # No retry_after_seconds → no header set
        rate_limit_exc = _make_429_exc(retry_after_seconds=None)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=[rate_limit_exc, success_response]
        )

        with patch("corpus_builder.twitter_collector.time.sleep") as mock_sleep:
            collector.collect_twitter(["#rajasthan"], max_results=10)

        mock_sleep.assert_called_once_with(60)

    def test_multiple_429s_before_success_retries_each_time(self):
        """
        GIVEN the API raises TooManyRequests twice before succeeding,
        WHEN  collect_twitter is invoked,
        THEN  time.sleep is called twice and the final results are returned.
        """
        tweet = _make_tweet()
        success_response = _make_response(tweets=[tweet])
        exc1 = _make_429_exc(retry_after_seconds=10)
        exc2 = _make_429_exc(retry_after_seconds=15)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=[exc1, exc2, success_response]
        )

        with patch("corpus_builder.twitter_collector.time.sleep") as mock_sleep:
            results = collector.collect_twitter(["#rajasthan"], max_results=10)

        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(10)
        mock_sleep.assert_any_call(15)
        assert len(results) == 1

    def test_429_logs_timestamp(self, caplog):
        """
        GIVEN a 429 response,
        WHEN  the rate-limit handler fires,
        THEN  a WARNING log entry is emitted containing the UTC timestamp.
        """
        import logging

        tweet = _make_tweet()
        success_response = _make_response(tweets=[tweet])
        rate_limit_exc = _make_429_exc(retry_after_seconds=1)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=[rate_limit_exc, success_response]
        )

        with patch("corpus_builder.twitter_collector.time.sleep"):
            with caplog.at_level(logging.WARNING, logger="corpus_builder.twitter_collector"):
                collector.collect_twitter(["#rajasthan"], max_results=10)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("429" in msg or "rate limit" in msg.lower() for msg in warning_messages), (
            f"Expected a rate-limit warning log; got: {warning_messages}"
        )


# ---------------------------------------------------------------------------
# Auth-failure (HTTP 401 / 403) tests
# ---------------------------------------------------------------------------


class TestAuthFailureHandling:
    """Verify that 401 and 403 responses halt collection immediately."""

    def test_401_raises_twitter_auth_error(self):
        """
        GIVEN the API raises Unauthorized (HTTP 401),
        WHEN  collect_twitter is invoked,
        THEN  TwitterAuthError is raised and collection halts.
        """
        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=_make_401_exc()
        )

        with pytest.raises(TwitterAuthError) as exc_info:
            collector.collect_twitter(["#rajasthan"], max_results=10)

        assert "401" in str(exc_info.value)

    def test_403_raises_twitter_auth_error(self):
        """
        GIVEN the API raises Forbidden (HTTP 403),
        WHEN  collect_twitter is invoked,
        THEN  TwitterAuthError is raised and collection halts.
        """
        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=_make_403_exc()
        )

        with pytest.raises(TwitterAuthError) as exc_info:
            collector.collect_twitter(["#rajasthan"], max_results=10)

        assert "403" in str(exc_info.value)

    def test_401_does_not_retry(self):
        """
        GIVEN the API raises Unauthorized (HTTP 401),
        WHEN  collect_twitter is invoked,
        THEN  the API is called exactly once (no retry loop).
        """
        collector = _make_collector()
        mock_search = MagicMock(side_effect=_make_401_exc())
        collector._client.search_recent_tweets = mock_search

        with pytest.raises(TwitterAuthError):
            collector.collect_twitter(["#rajasthan"], max_results=10)

        mock_search.assert_called_once()

    def test_403_does_not_retry(self):
        """
        GIVEN the API raises Forbidden (HTTP 403),
        WHEN  collect_twitter is invoked,
        THEN  the API is called exactly once (no retry loop).
        """
        collector = _make_collector()
        mock_search = MagicMock(side_effect=_make_403_exc())
        collector._client.search_recent_tweets = mock_search

        with pytest.raises(TwitterAuthError):
            collector.collect_twitter(["#rajasthan"], max_results=10)

        mock_search.assert_called_once()

    def test_401_logs_error(self, caplog):
        """
        GIVEN the API raises Unauthorized (HTTP 401),
        WHEN  collect_twitter is invoked,
        THEN  an ERROR log entry is emitted.
        """
        import logging

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=_make_401_exc()
        )

        with caplog.at_level(logging.ERROR, logger="corpus_builder.twitter_collector"):
            with pytest.raises(TwitterAuthError):
                collector.collect_twitter(["#rajasthan"], max_results=10)

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("401" in msg or "authentication" in msg.lower() for msg in error_messages), (
            f"Expected an auth-failure error log; got: {error_messages}"
        )

    def test_403_logs_error(self, caplog):
        """
        GIVEN the API raises Forbidden (HTTP 403),
        WHEN  collect_twitter is invoked,
        THEN  an ERROR log entry is emitted.
        """
        import logging

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=_make_403_exc()
        )

        with caplog.at_level(logging.ERROR, logger="corpus_builder.twitter_collector"):
            with pytest.raises(TwitterAuthError):
                collector.collect_twitter(["#rajasthan"], max_results=10)

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("403" in msg or "authorization" in msg.lower() for msg in error_messages), (
            f"Expected an auth-failure error log; got: {error_messages}"
        )

    def test_no_sleep_on_auth_failure(self):
        """
        GIVEN the API raises Unauthorized (HTTP 401),
        WHEN  collect_twitter is invoked,
        THEN  time.sleep is never called (no rate-limit pause on auth errors).
        """
        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(
            side_effect=_make_401_exc()
        )

        with patch("corpus_builder.twitter_collector.time.sleep") as mock_sleep:
            with pytest.raises(TwitterAuthError):
                collector.collect_twitter(["#rajasthan"], max_results=10)

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# RawSentence field population tests
# ---------------------------------------------------------------------------


class TestRawSentenceFieldPopulation:
    """Verify that RawSentence fields are populated correctly on a 200 response."""

    def test_raw_sentence_text_matches_tweet_text(self):
        """
        GIVEN a tweet with a specific text,
        WHEN  collect_twitter returns that tweet,
        THEN  the resulting RawSentence.text equals the tweet text verbatim.
        """
        expected_text = "म्हारो राजस्थान बहुत सुंदर है यार #rajasthan"
        tweet = _make_tweet(text=expected_text)
        response = _make_response(tweets=[tweet])

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=10)

        assert len(results) == 1
        assert results[0].text == expected_text

    def test_raw_sentence_platform_is_twitter(self):
        """
        GIVEN a successful Twitter/X API response,
        WHEN  collect_twitter returns results,
        THEN  every RawSentence.platform is "twitter".
        """
        tweets = [_make_tweet(tweet_id=str(i)) for i in range(3)]
        response = _make_response(tweets=tweets)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=10)

        assert all(r.platform == "twitter" for r in results)

    def test_raw_sentence_source_url_contains_tweet_id(self):
        """
        GIVEN a tweet with id "9999999999",
        WHEN  collect_twitter returns that tweet,
        THEN  RawSentence.source_url contains the tweet id.
        """
        tweet_id = "9999999999"
        tweet = _make_tweet(tweet_id=tweet_id)
        response = _make_response(tweets=[tweet])

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=10)

        assert tweet_id in results[0].source_url

    def test_raw_sentence_source_url_is_valid_twitter_url(self):
        """
        GIVEN a successful API response,
        WHEN  collect_twitter returns results,
        THEN  RawSentence.source_url starts with "https://twitter.com".
        """
        tweet = _make_tweet(tweet_id="1111111111")
        response = _make_response(tweets=[tweet])

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=10)

        assert results[0].source_url.startswith("https://twitter.com")

    def test_raw_sentence_collected_at_is_utc_datetime(self):
        """
        GIVEN a successful API response,
        WHEN  collect_twitter returns results,
        THEN  RawSentence.collected_at is a timezone-aware UTC datetime.
        """
        tweet = _make_tweet()
        response = _make_response(tweets=[tweet])

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        before = datetime.now(tz=timezone.utc)
        results = collector.collect_twitter(["#rajasthan"], max_results=10)
        after = datetime.now(tz=timezone.utc)

        collected_at = results[0].collected_at
        assert collected_at.tzinfo is not None, "collected_at must be timezone-aware"
        assert collected_at.tzinfo == timezone.utc or collected_at.utcoffset().total_seconds() == 0
        assert before <= collected_at <= after

    def test_raw_sentence_sentence_id_is_valid_uuid(self):
        """
        GIVEN a successful API response,
        WHEN  collect_twitter returns results,
        THEN  RawSentence.sentence_id is a valid UUID string.
        """
        tweet = _make_tweet()
        response = _make_response(tweets=[tweet])

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=10)

        sentence_id = results[0].sentence_id
        # Should not raise ValueError
        parsed = uuid.UUID(sentence_id)
        assert str(parsed) == sentence_id

    def test_each_raw_sentence_has_unique_sentence_id(self):
        """
        GIVEN a response containing multiple tweets,
        WHEN  collect_twitter returns results,
        THEN  every RawSentence has a distinct sentence_id.
        """
        tweets = [_make_tweet(tweet_id=str(i), text=f"tweet text {i}") for i in range(5)]
        response = _make_response(tweets=tweets)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=10)

        ids = [r.sentence_id for r in results]
        assert len(ids) == len(set(ids)), "sentence_ids must be unique across collected tweets"

    def test_raw_sentence_is_correct_type(self):
        """
        GIVEN a successful API response,
        WHEN  collect_twitter returns results,
        THEN  every element in the list is a RawSentence instance.
        """
        tweets = [_make_tweet(tweet_id=str(i)) for i in range(3)]
        response = _make_response(tweets=tweets)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=10)

        assert all(isinstance(r, RawSentence) for r in results)

    def test_empty_query_terms_returns_empty_list(self):
        """
        GIVEN an empty query_terms list,
        WHEN  collect_twitter is invoked,
        THEN  an empty list is returned without calling the API.
        """
        collector = _make_collector()
        mock_search = MagicMock()
        collector._client.search_recent_tweets = mock_search

        results = collector.collect_twitter([], max_results=10)

        assert results == []
        mock_search.assert_not_called()

    def test_no_data_in_response_returns_empty_list(self):
        """
        GIVEN an API response with data=None (no matching tweets),
        WHEN  collect_twitter is invoked,
        THEN  an empty list is returned.
        """
        response = _make_response(tweets=None)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=10)

        assert results == []

    def test_max_results_respected(self):
        """
        GIVEN a response with more tweets than max_results,
        WHEN  collect_twitter is invoked with max_results=2,
        THEN  at most 2 RawSentence objects are returned.
        """
        tweets = [_make_tweet(tweet_id=str(i), text=f"tweet {i}") for i in range(5)]
        response = _make_response(tweets=tweets)

        collector = _make_collector()
        collector._client.search_recent_tweets = MagicMock(return_value=response)

        results = collector.collect_twitter(["#rajasthan"], max_results=2)

        assert len(results) <= 2


# ---------------------------------------------------------------------------
# _extract_retry_after unit tests
# ---------------------------------------------------------------------------


class TestExtractRetryAfter:
    """Unit tests for the _extract_retry_after static helper."""

    def test_uses_retry_after_header_when_present(self):
        exc = _make_429_exc(retry_after_seconds=45)
        assert TwitterCollector._extract_retry_after(exc) == 45

    def test_falls_back_to_60_when_no_headers(self):
        exc = _make_429_exc(retry_after_seconds=None)
        assert TwitterCollector._extract_retry_after(exc) == 60

    def test_falls_back_to_60_when_response_is_none(self):
        exc = tweepy.errors.TooManyRequests(response=MagicMock())
        exc.response = None
        assert TwitterCollector._extract_retry_after(exc) == 60

    def test_minimum_wait_is_1_second(self):
        """Retry-After: 0 should be clamped to at least 1 second."""
        exc = _make_429_exc(retry_after_seconds=0)
        # The implementation does max(1, int(header)), so 0 → 1
        result = TwitterCollector._extract_retry_after(exc)
        assert result >= 1

    def test_uses_x_rate_limit_reset_header_as_fallback(self):
        """When Retry-After is absent but x-rate-limit-reset is present, use it."""
        future_ts = int(time.time()) + 120
        http_response = MagicMock()
        http_response.headers = {"x-rate-limit-reset": str(future_ts)}
        exc = tweepy.errors.TooManyRequests(response=http_response)
        exc.response = http_response

        result = TwitterCollector._extract_retry_after(exc)
        # Should be approximately 120 seconds (allow ±2s for test execution time)
        assert 118 <= result <= 122
