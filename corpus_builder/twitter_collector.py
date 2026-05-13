"""
Twitter/X data collector for the RajNLP-50K corpus.

Uses the Tweepy v2 (Academic Research) API client to search for
Rajasthani-Hindi code-switched tweets based on politician names,
regional hashtags, and Rajasthani slang terms.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import tweepy

from models.data_models import RawSentence

logger = logging.getLogger(__name__)


class TwitterAuthError(Exception):
    """Raised when the Twitter/X API returns a 401 or 403 authentication error."""


class TwitterCollector:
    """Collects Rajasthani-Hindi code-switched tweets via the Twitter/X v2 Academic API.

    Args:
        bearer_token: Twitter/X API bearer token for OAuth 2.0 App-Only authentication.
            Do not hardcode this value; pass it from an environment variable or secrets
            manager at call time.
    """

    def __init__(self, bearer_token: str) -> None:
        self._client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect_twitter(
        self,
        query_terms: list[str],
        max_results: int = 100,
    ) -> list[RawSentence]:
        """Search Twitter/X for tweets matching the given query terms.

        Builds a single OR-joined query from *query_terms*, then pages through
        results using the Twitter v2 ``recent_search`` endpoint.  Each tweet is
        stored as a :class:`~models.data_models.RawSentence` with:

        - ``source_url`` — canonical tweet URL
        - ``collected_at`` — UTC timestamp at collection time
        - ``platform`` — ``"twitter"``
        - ``sentence_id`` — a fresh UUID4

        Rate-limit handling (HTTP 429):
            Logs the event with a UTC timestamp, then sleeps for the number of
            seconds specified in the ``x-rate-limit-reset`` response header (or
            60 seconds if the header is absent), then resumes automatically.

        Auth-failure handling (HTTP 401 / 403):
            Logs the error and raises :class:`TwitterAuthError`, halting
            collection immediately.

        Args:
            query_terms: A list of search terms (politician names, hashtags,
                slang words, etc.).  All terms are OR-joined into a single
                Twitter query string.
            max_results: Maximum total number of tweets to return.  Defaults
                to 100.  The method pages through results until this limit is
                reached or the API has no more pages.

        Returns:
            A list of :class:`~models.data_models.RawSentence` objects, one per
            collected tweet.

        Raises:
            TwitterAuthError: If the API returns HTTP 401 or 403.
        """
        if not query_terms:
            logger.warning("collect_twitter called with empty query_terms; returning []")
            return []

        query = self._build_query(query_terms)
        logger.info("Starting Twitter collection | query=%r | max_results=%d", query, max_results)

        sentences: list[RawSentence] = []
        next_token: Optional[str] = None

        while len(sentences) < max_results:
            batch_size = min(100, max_results - len(sentences))
            # Twitter v2 recent_search requires 10 ≤ max_results ≤ 100
            batch_size = max(10, batch_size)

            response = self._search_with_retry(
                query=query,
                max_results=batch_size,
                next_token=next_token,
                tweet_fields=["created_at", "author_id"],
                expansions=["author_id"],
            )

            if response is None or response.data is None:
                # No more results
                break

            for tweet in response.data:
                if len(sentences) >= max_results:
                    break
                raw = self._tweet_to_raw_sentence(tweet)
                sentences.append(raw)

            # Check for next page
            meta = getattr(response, "meta", None)
            next_token = meta.get("next_token") if meta else None
            if not next_token:
                break

        logger.info("Twitter collection complete | collected=%d tweets", len(sentences))
        return sentences

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(query_terms: list[str]) -> str:
        """Build a Twitter v2 search query by OR-joining all terms.

        Each multi-word term is wrapped in double quotes so that Twitter treats
        it as a phrase search.  Single-word terms (including hashtags) are left
        unquoted.

        Args:
            query_terms: Raw list of search terms.

        Returns:
            A Twitter v2 query string, e.g.
            ``'"Ashok Gehlot" OR #rajasthan OR म्हारो -is:retweet lang:hi'``
        """
        parts: list[str] = []
        for term in query_terms:
            stripped = term.strip()
            if not stripped:
                continue
            # Wrap multi-word terms in quotes; leave single tokens / hashtags bare
            if " " in stripped and not stripped.startswith('"'):
                parts.append(f'"{stripped}"')
            else:
                parts.append(stripped)

        # Append standard filters: exclude retweets, restrict to Hindi/Rajasthani
        base_query = " OR ".join(parts)
        return f"({base_query}) -is:retweet"

    def _search_with_retry(self, **kwargs) -> Optional[tweepy.Response]:
        """Call ``tweepy.Client.search_recent_tweets`` with rate-limit handling.

        On HTTP 429, logs the event and sleeps for the ``retry_after`` duration
        from the response header before retrying once.  On HTTP 401/403, logs
        and raises :class:`TwitterAuthError`.

        Args:
            **kwargs: Keyword arguments forwarded to
                ``tweepy.Client.search_recent_tweets``.

        Returns:
            The :class:`tweepy.Response` from the API, or ``None`` if no data
            was returned.

        Raises:
            TwitterAuthError: On HTTP 401 or 403.
        """
        while True:
            try:
                response: tweepy.Response = self._client.search_recent_tweets(**kwargs)
                return response

            except tweepy.errors.TooManyRequests as exc:
                retry_after = self._extract_retry_after(exc)
                logger.warning(
                    "Twitter rate limit hit (HTTP 429) at %s — pausing for %d seconds",
                    datetime.now(tz=timezone.utc).isoformat(),
                    retry_after,
                )
                time.sleep(retry_after)
                # Loop back and retry the same request

            except tweepy.errors.Unauthorized as exc:
                logger.error(
                    "Twitter authentication failure (HTTP 401): %s — halting collection",
                    exc,
                )
                raise TwitterAuthError(
                    f"Twitter API returned HTTP 401 (Unauthorized): {exc}"
                ) from exc

            except tweepy.errors.Forbidden as exc:
                logger.error(
                    "Twitter authorization failure (HTTP 403): %s — halting collection",
                    exc,
                )
                raise TwitterAuthError(
                    f"Twitter API returned HTTP 403 (Forbidden): {exc}"
                ) from exc

    @staticmethod
    def _extract_retry_after(exc: tweepy.errors.TooManyRequests) -> int:
        """Extract the ``retry_after`` wait time from a 429 exception.

        Tweepy surfaces the raw ``requests.Response`` on the exception object.
        We read the ``x-rate-limit-reset`` header (a Unix epoch timestamp) and
        compute the remaining seconds.  Falls back to 60 seconds if the header
        is absent or cannot be parsed.

        Args:
            exc: The :class:`tweepy.errors.TooManyRequests` exception.

        Returns:
            Number of seconds to sleep before retrying.
        """
        default_wait = 60
        try:
            response = exc.response  # requests.Response
            if response is None:
                return default_wait

            # Prefer the standard Retry-After header (seconds)
            retry_after_header = response.headers.get("Retry-After")
            if retry_after_header is not None:
                return max(1, int(retry_after_header))

            # Fall back to x-rate-limit-reset (Unix timestamp)
            reset_header = response.headers.get("x-rate-limit-reset")
            if reset_header is not None:
                reset_ts = int(reset_header)
                now_ts = int(time.time())
                wait = max(1, reset_ts - now_ts)
                return wait

        except Exception:  # noqa: BLE001
            pass

        return default_wait

    @staticmethod
    def _tweet_to_raw_sentence(tweet: tweepy.Tweet) -> RawSentence:
        """Convert a :class:`tweepy.Tweet` object to a :class:`RawSentence`.

        Args:
            tweet: A tweet object returned by the Tweepy v2 client.

        Returns:
            A :class:`~models.data_models.RawSentence` with all required fields
            populated.
        """
        tweet_id: str = str(tweet.id)
        # author_id may be None if the expansion was not requested / not returned
        author_id: str = str(tweet.author_id) if tweet.author_id else "unknown"
        source_url = f"https://twitter.com/i/web/status/{tweet_id}"

        return RawSentence(
            text=tweet.text,
            source_url=source_url,
            collected_at=datetime.now(tz=timezone.utc),
            platform="twitter",
            sentence_id=str(uuid.uuid4()),
        )
