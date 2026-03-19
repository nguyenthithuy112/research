"""Integration tests for the ResearchAgent."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest

from agent import ResearchAgent
from tools.search_tool import SearchResponse, SearchResult, SearchTool
from tools.code_executor import CodeExecutionResult, CodeExecutor
from tools.report_generator import ReportGenerator


def _make_mock_search_tool(num_results: int = 2) -> SearchTool:
    """Return a SearchTool that never makes real network calls."""
    tool = MagicMock(spec=SearchTool)
    result = SearchResult(
        title="Example Article",
        url="https://example.com/article",
        snippet="This is a snippet about the topic.",
    )
    tool.search.return_value = SearchResponse(
        query="mock query", results=[result] * num_results
    )
    tool.search_multiple.return_value = [
        SearchResponse(query=f"query {i}", results=[result] * num_results)
        for i in range(3)
    ]
    return tool


class TestResearchAgentPlan:
    def test_heuristic_plan_returns_list(self):
        agent = ResearchAgent()
        queries = agent._heuristic_plan("electric vehicles adoption")
        assert isinstance(queries, list)
        assert 1 <= len(queries) <= 5

    def test_heuristic_plan_contains_query_terms(self):
        agent = ResearchAgent()
        queries = agent._heuristic_plan("solar energy")
        combined = " ".join(queries).lower()
        assert "solar energy" in combined

    def test_heuristic_plan_respects_max(self):
        agent = ResearchAgent(max_search_queries=2)
        queries = agent._heuristic_plan("something")
        assert len(queries) <= 2

    def test_plan_uses_llm_when_available(self):
        llm = MagicMock(return_value="1. query one\n2. query two\n3. query three")
        agent = ResearchAgent(llm_callable=llm)
        queries = agent._plan("climate change")
        assert len(queries) == 3
        assert "query one" in queries

    def test_parse_numbered_list(self):
        text = "1. First item\n2. Second item\n3. Third item"
        items = ResearchAgent._parse_numbered_list(text)
        assert items == ["First item", "Second item", "Third item"]

    def test_parse_numbered_list_empty(self):
        assert ResearchAgent._parse_numbered_list("") == []


class TestResearchAgentVerify:
    def test_verify_deduplicates_urls(self):
        agent = ResearchAgent()
        result = SearchResult("T", "https://dup.com", "Snippet")
        responses = [
            SearchResponse(query="q1", results=[result]),
            SearchResponse(query="q2", results=[result]),
        ]
        findings = agent._verify(responses, "topic")
        # URL should appear only once
        assert findings.count("https://dup.com") == 1

    def test_verify_handles_errors(self):
        agent = ResearchAgent()
        responses = [SearchResponse(query="q", error="timeout")]
        findings = agent._verify(responses, "topic")
        assert "timeout" in findings

    def test_verify_empty_responses(self):
        agent = ResearchAgent()
        findings = agent._verify([], "topic")
        assert isinstance(findings, str)


class TestResearchAgentAnalyse:
    def test_heuristic_analysis_code_is_valid_python(self):
        agent = ResearchAgent()
        code = agent._heuristic_analysis_code("### Query: test\n- result1\n- result2")
        # If the generated code has syntax errors, compile() will raise SyntaxError
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"Generated code has syntax error: {exc}")

    def test_fallback_analysis_code(self):
        agent = ResearchAgent()
        code = agent._fallback_analysis_code("findings text")
        executor = CodeExecutor()
        result = executor.execute(code)
        assert "completed successfully" in result.stdout.lower()


class TestResearchAgentRun:
    def test_run_returns_markdown(self):
        mock_search = _make_mock_search_tool()
        agent = ResearchAgent(search_tool=mock_search)
        report = agent.run("What are the latest AI trends?")
        assert "# Research Report" in report
        assert "What are the latest AI trends?" in report

    def test_run_contains_all_sections(self):
        mock_search = _make_mock_search_tool()
        agent = ResearchAgent(search_tool=mock_search)
        report = agent.run("renewable energy statistics")
        assert "Research Plan" in report
        assert "Search Findings" in report
        assert "Data Analysis" in report
        assert "Key Insights" in report

    def test_run_and_save(self, tmp_path):
        mock_search = _make_mock_search_tool()
        agent = ResearchAgent(search_tool=mock_search)
        output = str(tmp_path / "test_report")
        path = agent.run_and_save("AI in healthcare", output)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "# Research Report" in content

    def test_run_no_results(self):
        """Agent should still produce a report when search returns nothing."""
        mock_search = MagicMock(spec=SearchTool)
        mock_search.search_multiple.return_value = [
            SearchResponse(query="q1", results=[])
        ]
        agent = ResearchAgent(search_tool=mock_search)
        report = agent.run("obscure topic with no results")
        assert "# Research Report" in report

    def test_insights_counts(self):
        result = SearchResult("T", "https://example.com", "S")
        responses = [SearchResponse(query="q", results=[result, result])]
        exec_result = CodeExecutionResult(code="x=1", stdout="done")
        insights = ResearchAgent._extract_insights(responses, exec_result)
        assert "2" in insights  # total results
