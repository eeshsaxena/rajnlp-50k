"""
ShareChat data collector for the RajNLP-50K corpus.

Uses Selenium WebDriver (headless Chrome) to scrape ShareChat pages targeting
Rajasthani politics and news content.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from models.data_models import RawSentence

logger = logging.getLogger(__name__)

# CSS selectors to try when extracting post text from ShareChat pages.
# Listed in order of specificity — fall back to broader selectors if needed.
_POST_TEXT_SELECTORS: list[str] = [
    ".share-chat-post",
    "[data-testid='post-text']",
    "p",
]

# Minimum character length for a text snippet to be considered a valid sentence.
_MIN_TEXT_LENGTH: int = 10

# Indicators in page title / body that signal an HTTP error page.
_ERROR_INDICATORS: tuple[str, ...] = (
    "404",
    "not found",
    "page not found",
    "error",
    "403",
    "forbidden",
    "500",
    "502",
    "503",
    "service unavailable",
)


def _build_driver(headless: bool = True) -> webdriver.Chrome:
    """Create and return a configured Chrome WebDriver instance.

    Args:
        headless: If ``True`` (default), run Chrome in headless mode.

    Returns:
        A :class:`selenium.webdriver.Chrome` instance.
    """
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=chrome_options)


def _is_error_page(driver: webdriver.Chrome) -> bool:
    """Return ``True`` if the current page appears to be an HTTP error page.

    Selenium does not expose HTTP status codes directly, so we inspect the
    page title and a small portion of the body text for well-known error
    indicators.

    Args:
        driver: An active WebDriver instance positioned on the page to check.

    Returns:
        ``True`` if the page looks like an error page, ``False`` otherwise.
    """
    try:
        title = driver.title.lower()
        if any(indicator in title for indicator in _ERROR_INDICATORS):
            return True
        # Also check the first 500 characters of body text
        body_text = driver.find_element(By.TAG_NAME, "body").text[:500].lower()
        if any(indicator in body_text for indicator in _ERROR_INDICATORS):
            return True
    except WebDriverException:
        pass
    return False


def _extract_texts(driver: webdriver.Chrome) -> list[str]:
    """Extract non-empty text snippets from the current page.

    Tries each CSS selector in :data:`_POST_TEXT_SELECTORS` in order.  Falls
    back to extracting text from all ``<div>`` elements with substantial
    content if no selector matches.

    Args:
        driver: An active WebDriver instance positioned on the target page.

    Returns:
        A list of non-empty text strings found on the page.
    """
    texts: list[str] = []

    for selector in _POST_TEXT_SELECTORS:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for el in elements:
                text = el.text.strip()
                if len(text) >= _MIN_TEXT_LENGTH:
                    texts.append(text)
            if texts:
                return texts
        except WebDriverException:
            continue

    # Fallback: collect text from <div> elements with substantial content
    if not texts:
        try:
            divs = driver.find_elements(By.TAG_NAME, "div")
            for div in divs:
                text = div.text.strip()
                if len(text) >= _MIN_TEXT_LENGTH:
                    texts.append(text)
        except WebDriverException:
            pass

    return texts


class ShareChatCollector:
    """Collects Rajasthani-Hindi code-switched text from ShareChat via Selenium.

    Args:
        timeout: Page-load timeout in seconds (default 30).
        headless: If ``True`` (default), run Chrome in headless mode.
    """

    def __init__(self, timeout: int = 30, headless: bool = True) -> None:
        self._timeout = timeout
        self._headless = headless

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect_sharechat(self, page_urls: list[str]) -> list[RawSentence]:
        """Scrape ShareChat pages and return collected sentences.

        For each URL in *page_urls*:

        1. Navigate to the URL with the configured timeout.
        2. Detect HTTP error pages (4xx/5xx) by inspecting page content.
           If an error page is detected, log the URL and skip to the next.
        3. Extract text snippets from post elements.
        4. Create a :class:`~models.data_models.RawSentence` for each snippet.

        Timeout handling:
            On :class:`~selenium.common.exceptions.TimeoutException`, retry
            once with 2× the configured timeout.  If the retry also times out,
            log the URL and skip the page.

        Args:
            page_urls: List of ShareChat page URLs to scrape.

        Returns:
            A list of :class:`~models.data_models.RawSentence` objects, one
            per extracted text snippet.
        """
        if not page_urls:
            logger.warning("collect_sharechat called with empty page_urls; returning []")
            return []

        sentences: list[RawSentence] = []
        driver = _build_driver(headless=self._headless)

        try:
            for url in page_urls:
                page_sentences = self._scrape_url(driver, url, self._timeout)
                sentences.extend(page_sentences)
        finally:
            driver.quit()

        logger.info(
            "ShareChat collection complete | collected=%d sentences from %d URLs",
            len(sentences),
            len(page_urls),
        )
        return sentences

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scrape_url(
        self,
        driver: webdriver.Chrome,
        url: str,
        timeout: int,
    ) -> list[RawSentence]:
        """Attempt to scrape a single URL, with one retry on timeout.

        Args:
            driver: Active WebDriver instance.
            url: The ShareChat page URL to scrape.
            timeout: Page-load timeout in seconds for this attempt.

        Returns:
            A (possibly empty) list of :class:`~models.data_models.RawSentence`
            objects extracted from the page.
        """
        try:
            return self._load_and_extract(driver, url, timeout)
        except TimeoutException:
            logger.warning(
                "ShareChat page load timed out (timeout=%ds) for URL: %s — retrying with %ds",
                timeout,
                url,
                timeout * 2,
            )
            # Retry once with 2× timeout
            try:
                return self._load_and_extract(driver, url, timeout * 2)
            except TimeoutException:
                logger.warning(
                    "ShareChat page load timed out again (timeout=%ds) for URL: %s — skipping",
                    timeout * 2,
                    url,
                )
                return []
        except WebDriverException as exc:
            logger.error(
                "WebDriverException while loading URL %s: %s — skipping",
                url,
                exc,
            )
            return []

    def _load_and_extract(
        self,
        driver: webdriver.Chrome,
        url: str,
        timeout: int,
    ) -> list[RawSentence]:
        """Navigate to *url*, check for errors, and extract text.

        Args:
            driver: Active WebDriver instance.
            url: The ShareChat page URL to load.
            timeout: Page-load timeout in seconds.

        Returns:
            A list of :class:`~models.data_models.RawSentence` objects.

        Raises:
            TimeoutException: If the page does not load within *timeout* seconds.
            WebDriverException: On any other Selenium navigation error.
        """
        driver.set_page_load_timeout(timeout)
        driver.get(url)

        # Wait for the body element to be present (basic page-load check)
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except TimeoutException:
            raise

        # Detect HTTP error pages (Selenium doesn't expose status codes)
        if _is_error_page(driver):
            # Try to extract a status code hint from the page title for logging
            title = driver.title
            status_hint = "unknown"
            for code in ("404", "403", "500", "502", "503"):
                if code in title or code in driver.find_element(By.TAG_NAME, "body").text[:200]:
                    status_hint = code
                    break
            logger.warning(
                "ShareChat page returned HTTP error (status=%s) for URL: %s — skipping",
                status_hint,
                url,
            )
            return []

        # Extract text snippets from the page
        texts = _extract_texts(driver)
        collected_at = datetime.now(tz=timezone.utc)

        sentences: list[RawSentence] = []
        for text in texts:
            sentences.append(
                RawSentence(
                    text=text,
                    source_url=url,
                    collected_at=collected_at,
                    platform="sharechat",
                    sentence_id=str(uuid.uuid4()),
                )
            )

        logger.debug(
            "Extracted %d sentences from URL: %s",
            len(sentences),
            url,
        )
        return sentences
