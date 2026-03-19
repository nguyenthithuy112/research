"""Tests for the SearchTool."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest

from tools.search_tool import SearchResult, SearchResponse, SearchTool


class TestSearchResult:
    def test_str(self):
        r = SearchResult(title="Test", url="https://example.com", snippet="A snippet")
        assert "Test" in str(r)
        assert "https://example.com" in str(r)
        assert "A snippet" in str(r)


class TestSearchResponse:
    def test_is_empty_no_results(self):
        resp = SearchResponse(query="test")
        assert resp.is_empty

    def test_is_empty_with_results(self):
        resp = SearchResponse(
            query="test",
            results=[SearchResult("T", "http://x.com", "S")],
        )
        assert not resp.is_empty

    def test_summary_with_error(self):
        resp = SearchResponse(query="q", error="network error")
        assert "network error" in resp.summary()

    def test_summary_no_results(self):
        resp = SearchResponse(query="q")
        assert "No results" in resp.summary()

    def test_summary_with_results(self):
        resp = SearchResponse(
            query="climate change",
            results=[
                SearchResult(
                    title="NASA Climate",
                    url="https://nasa.gov",
                    snippet="Facts about climate.",
                )
            ],
        )
        summary = resp.summary()
        assert "NASA Climate" in summary
        assert "nasa.gov" in summary


class TestSearchTool:
    def test_empty_query_returns_error(self):
        tool = SearchTool()
        resp = tool.search("")
        assert resp.error is not None

    def test_whitespace_query_returns_error(self):
        tool = SearchTool()
        resp = tool.search("   ")
        assert resp.error is not None

    def test_backend_auto_select_no_google_creds(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
        tool = SearchTool()
        assert tool._backend == "duckduckgo"

    def test_backend_auto_select_with_google_creds(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "key123")
        monkeypatch.setenv("GOOGLE_CSE_ID", "cx123")
        tool = SearchTool()
        assert tool._backend == "google"

    def test_force_backend(self):
        tool = SearchTool(backend="duckduckgo")
        assert tool._backend == "duckduckgo"

    def test_search_duckduckgo_missing_library(self, monkeypatch):
        """Gracefully handle missing duckduckgo-search library."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "duckduckgo_search":
                raise ImportError("No module named 'duckduckgo_search'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        tool = SearchTool(backend="duckduckgo")
        resp = tool.search("test query")
        assert resp.error is not None
        assert "duckduckgo-search" in resp.error

    def test_search_multiple(self):
        tool = SearchTool(backend="duckduckgo")
        # Patch the underlying search method to avoid network calls
        with patch.object(tool, "search") as mock_search:
            mock_search.return_value = SearchResponse(
                query="q", results=[SearchResult("T", "http://x.com", "S")]
            )
            responses = tool.search_multiple(["q1", "q2"])
        assert len(responses) == 2
        assert mock_search.call_count == 2

    def test_rate_limit_respected(self):
        """Verify _rate_limit doesn't error and updates last request time."""
        import time
        tool = SearchTool(sleep_between_requests=0)
        before = tool._last_request_time
        tool._rate_limit()
        assert tool._last_request_time >= before
