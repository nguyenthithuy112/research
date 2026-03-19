"""
Search tool for the Autonomous Research & Data Analysis Agent.

Supports DuckDuckGo (free, no API key required) by default, with an optional
Google Custom Search API backend when credentials are configured.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""

    title: str
    url: str
    snippet: str
    source: str = ""

    def __str__(self) -> str:
        return f"[{self.title}]({self.url})\n{self.snippet}"


@dataclass
class SearchResponse:
    """Aggregated response from a search query."""

    query: str
    results: List[SearchResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0

    def summary(self) -> str:
        """Return a human-readable summary of all results."""
        if self.error:
            return f"Search error for '{self.query}': {self.error}"
        if self.is_empty:
            return f"No results found for '{self.query}'."
        lines = [f"Search results for: **{self.query}**\n"]
        for i, result in enumerate(self.results, 1):
            lines.append(f"{i}. **{result.title}**")
            lines.append(f"   URL: {result.url}")
            lines.append(f"   {result.snippet}\n")
        return "\n".join(lines)


class SearchTool:
    """
    Web search tool that uses DuckDuckGo by default.

    Falls back to Google Custom Search API when the environment variables
    ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID`` are both set.

    Parameters
    ----------
    max_results:
        Maximum number of results to return per query (default: 5).
    sleep_between_requests:
        Seconds to wait between consecutive requests to avoid rate limiting
        (default: 1.0).
    backend:
        Force a specific backend: ``"duckduckgo"`` or ``"google"``.
        Defaults to auto-selection based on available credentials.
    """

    def __init__(
        self,
        max_results: int = 5,
        sleep_between_requests: float = 1.0,
        backend: Optional[str] = None,
    ) -> None:
        self.max_results = max_results
        self.sleep_between_requests = sleep_between_requests
        self._last_request_time: float = 0.0
        self._backend = backend or self._detect_backend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> SearchResponse:
        """
        Execute a web search and return structured results.

        Parameters
        ----------
        query:
            The search query string.

        Returns
        -------
        SearchResponse
            Contains the original query and a list of :class:`SearchResult`.
        """
        if not query or not query.strip():
            return SearchResponse(query=query, error="Empty query provided.")

        self._rate_limit()

        logger.info("Searching [%s]: %s", self._backend, query)
        if self._backend == "google":
            return self._search_google(query)
        return self._search_duckduckgo(query)

    def search_multiple(self, queries: List[str]) -> List[SearchResponse]:
        """Execute multiple searches sequentially, respecting rate limits."""
        responses: List[SearchResponse] = []
        for query in queries:
            responses.append(self.search(query))
        return responses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend() -> str:
        has_google = bool(
            os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID")
        )
        return "google" if has_google else "duckduckgo"

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.sleep_between_requests:
            time.sleep(self.sleep_between_requests - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # DuckDuckGo backend
    # ------------------------------------------------------------------

    def _search_duckduckgo(self, query: str) -> SearchResponse:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import]
        except ImportError:
            return SearchResponse(
                query=query,
                error=(
                    "duckduckgo-search is not installed. "
                    "Run: pip install duckduckgo-search"
                ),
            )

        try:
            results: List[SearchResult] = []
            with DDGS() as ddgs:
                for item in ddgs.text(
                    query, max_results=self.max_results
                ):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("href", ""),
                            snippet=item.get("body", ""),
                            source="duckduckgo",
                        )
                    )
            return SearchResponse(query=query, results=results)
        except Exception as exc:  # pragma: no cover
            logger.warning("DuckDuckGo search failed: %s", exc)
            return SearchResponse(query=query, error=str(exc))

    # ------------------------------------------------------------------
    # Google Custom Search backend
    # ------------------------------------------------------------------

    def _search_google(self, query: str) -> SearchResponse:
        try:
            import requests  # type: ignore[import]
        except ImportError:
            return SearchResponse(
                query=query,
                error="requests is not installed. Run: pip install requests",
            )

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        cse_id = os.environ.get("GOOGLE_CSE_ID", "")

        if not api_key or not cse_id:
            return SearchResponse(
                query=query,
                error=(
                    "Google Custom Search requires GOOGLE_API_KEY and "
                    "GOOGLE_CSE_ID environment variables."
                ),
            )

        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": min(self.max_results, 10),
        }
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            results: List[SearchResult] = []
            for item in data.get("items", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                    )
                )
            return SearchResponse(query=query, results=results)
        except Exception as exc:  # pragma: no cover
            logger.warning("Google search failed: %s", exc)
            return SearchResponse(query=query, error=str(exc))
