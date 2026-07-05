"""Throttled, retrying HTTP layer.

pbpstats' stats.nba.com loaders all fetch through
``pbpstats.data_loader.stats_nba.web_loader``'s module-level ``requests``
reference. ``install_throttle()`` swaps that reference for a shim that rate
limits, retries transient failures, and sends modern browser headers, so
every web fetch made by pbpstats is polite and robust without forking it.
"""

import logging
import random
import time

import requests

logger = logging.getLogger(__name__)

# Headers that keep stats.nba.com happy.
STATS_NBA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ThrottledRequests:
    """Drop-in replacement for the ``requests`` module exposing ``get``."""

    def __init__(self, min_interval: float = 0.6, max_retries: int = 5):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_request_at = 0.0
        self._session = requests.Session()

    def _wait(self):
        elapsed = time.monotonic() - self._last_request_at
        delay = self.min_interval + random.uniform(0, 0.25) - elapsed
        if delay > 0:
            time.sleep(delay)

    def get(self, url, params=None, headers=None, timeout=30, **kwargs):
        merged_headers = dict(headers or {})
        merged_headers.update(STATS_NBA_HEADERS)
        last_exc = None
        for attempt in range(self.max_retries):
            self._wait()
            self._last_request_at = time.monotonic()
            try:
                response = self._session.get(
                    url, params=params, headers=merged_headers,
                    timeout=timeout, **kwargs,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                backoff = 2 ** attempt + random.uniform(0, 1)
                logger.warning(
                    "Request error for %s (%s); retrying in %.1fs", url, exc, backoff
                )
                time.sleep(backoff)
                continue
            if response.status_code in RETRYABLE_STATUS:
                backoff = 3 * 2 ** attempt + random.uniform(0, 2)
                logger.warning(
                    "HTTP %s for %s; retrying in %.1fs",
                    response.status_code, url, backoff,
                )
                time.sleep(backoff)
                last_exc = None
                continue
            return response
        if last_exc is not None:
            raise last_exc
        raise requests.HTTPError(f"Exhausted retries fetching {url}")


_throttled: ThrottledRequests | None = None


def get_throttled() -> ThrottledRequests:
    global _throttled
    if _throttled is None:
        _throttled = ThrottledRequests()
    return _throttled


def install_throttle():
    """Route all pbpstats stats.nba.com web fetches through the throttler.

    Covers the data loaders plus the enhanced-pbp modules that make direct
    boxscore requests when period starters can't be inferred from pbp.
    """
    from pbpstats.data_loader.stats_nba import web_loader
    from pbpstats.resources.enhanced_pbp import start_of_period
    from pbpstats.resources.enhanced_pbp.stats_nba import enhanced_pbp_item

    shim = get_throttled()
    for module in (web_loader, start_of_period, enhanced_pbp_item):
        if module.requests is not shim:
            module.requests = shim
