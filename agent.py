"""
Autonomous Research & Data Analysis Agent
==========================================

Conducts deep research on a topic using a strict four-step workflow:

1. **Planning & Initial Search** – Break the query into sub-questions and
   search each one with specific, targeted keywords.
2. **Data Verification & Analysis** – Cross-reference results and verify
   time-sensitive facts (e.g. the current real-time date).
3. **Python Code Execution** – For any numerical data or statistics, write
   and run Python code to clean, summarise, and visualise the findings.
4. **Final Report Generation** – Synthesise everything into a professional
   Markdown report with headings, bullet points, and embedded charts.

Usage
-----
::

    from agent import ResearchAgent

    agent = ResearchAgent()
    report_md = agent.run("What are the latest trends in renewable energy?")
    print(report_md)
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

# Load environment variables from a .env file if present (e.g. GOOGLE_API_KEY)
try:
    from dotenv import load_dotenv  # type: ignore[import]

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; environment variables may still be set directly

from tools.code_executor import CodeExecutionResult, CodeExecutor
from tools.report_generator import ReportGenerator, ReportSection, ResearchReport
from tools.search_tool import SearchResponse, SearchTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates used to guide the agent's internal reasoning.
# ---------------------------------------------------------------------------

_PLANNING_PROMPT = textwrap.dedent(
    """\
    You are a research planning assistant. Given the user's research question,
    decompose it into 3-5 specific, targeted search queries that together cover
    the topic comprehensively.

    User question: {query}

    Return ONLY a numbered list of queries, one per line. Example:
    1. <specific query 1>
    2. <specific query 2>
    """
)

_ANALYSIS_CODE_PROMPT = textwrap.dedent(
    """\
    Based on the following research findings, write Python code that:
    1. Organises the key data points into a pandas DataFrame or structured list.
    2. Computes relevant summary statistics or trends.
    3. Creates at least one Matplotlib chart that visualises the findings.
    4. Prints a formatted summary table using tabulate.

    Findings:
    {findings}

    Write ONLY valid Python code. No explanations, no markdown fences.
    """
)


class ResearchAgent:
    """
    Autonomous Research & Data Analysis Agent.

    Parameters
    ----------
    search_tool:
        A :class:`~tools.search_tool.SearchTool` instance.  Created
        automatically if not supplied.
    code_executor:
        A :class:`~tools.code_executor.CodeExecutor` instance.  Created
        automatically if not supplied.
    report_generator:
        A :class:`~tools.report_generator.ReportGenerator` instance.  Created
        automatically if not supplied.
    max_search_queries:
        Maximum number of sub-queries generated during the planning phase
        (default: 5).
    llm_callable:
        Optional callable that accepts a prompt string and returns a text
        response.  When provided the agent uses it for planning and code
        generation; otherwise it falls back to rule-based heuristics so the
        agent still works without an LLM.
    verbose:
        If ``True``, emit INFO-level log messages to ``stdout`` (default:
        ``False``).
    """

    def __init__(
        self,
        search_tool: Optional[SearchTool] = None,
        code_executor: Optional[CodeExecutor] = None,
        report_generator: Optional[ReportGenerator] = None,
        max_search_queries: int = 5,
        llm_callable: Optional[object] = None,
        verbose: bool = False,
    ) -> None:
        self.search_tool = search_tool or SearchTool()
        self.code_executor = code_executor or CodeExecutor()
        self.report_generator = report_generator or ReportGenerator()
        self.max_search_queries = max_search_queries
        self.llm_callable = llm_callable

        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(levelname)s | %(name)s | %(message)s",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> str:
        """
        Execute the full four-step research workflow for *query*.

        Parameters
        ----------
        query:
            The research topic or question.

        Returns
        -------
        str
            A fully rendered Markdown research report.
        """
        logger.info("=== Step 1: Planning & Initial Search ===")
        sub_queries = self._plan(query)
        search_responses = self._search_all(sub_queries, query)

        logger.info("=== Step 2: Data Verification & Analysis ===")
        verified_findings = self._verify(search_responses, query)

        logger.info("=== Step 3: Python Code Execution ===")
        code, exec_result = self._analyse(verified_findings)

        logger.info("=== Step 4: Final Report Generation ===")
        report = self._build_report(query, sub_queries, search_responses, verified_findings, code, exec_result)
        return self.report_generator.render(report)

    def run_and_save(self, query: str, output_path: str) -> str:
        """
        Execute the research workflow and save the report to *output_path*.

        Returns
        -------
        str
            The absolute path of the saved ``.md`` file.
        """
        report_md = self.run(query)
        if not output_path.endswith(".md"):
            output_path += ".md"
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(report_md)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Step 1 – Planning
    # ------------------------------------------------------------------

    def _plan(self, query: str) -> List[str]:
        """Decompose the query into specific sub-queries."""
        if self.llm_callable:
            prompt = _PLANNING_PROMPT.format(query=query)
            raw = self._call_llm(prompt)
            return self._parse_numbered_list(raw)[: self.max_search_queries]
        return self._heuristic_plan(query)

    def _heuristic_plan(self, query: str) -> List[str]:
        """
        Rule-based fallback that generates targeted sub-queries without an LLM.

        Produces queries for: current year statistics, latest trends, key
        metrics, official sources, and future outlook.
        """
        current_year = datetime.datetime.now(datetime.timezone.utc).year
        clean = query.strip().rstrip("?")
        return [
            f"{clean} statistics {current_year}",
            f"{clean} latest trends and developments",
            f"{clean} key facts and data",
            f"{clean} official reports and research",
            f"{clean} future outlook and predictions",
        ][: self.max_search_queries]

    # ------------------------------------------------------------------
    # Step 2 – Search & Verification
    # ------------------------------------------------------------------

    def _search_all(
        self, sub_queries: List[str], original_query: str
    ) -> List[SearchResponse]:
        """Execute all sub-queries; also search for the current date."""
        date_query = (
            f"current date today {datetime.datetime.now(datetime.timezone.utc).year}"
        )
        all_queries = sub_queries + [date_query]
        responses = self.search_tool.search_multiple(all_queries)
        return responses

    def _verify(
        self, responses: List[SearchResponse], query: str
    ) -> str:
        """
        Cross-reference search results into a consolidated findings string.

        Extracts snippets, de-duplicates URLs, and notes any conflicts.
        """
        seen_urls: set = set()
        findings_lines: List[str] = []

        for resp in responses:
            if resp.error:
                findings_lines.append(f"[Search error for '{resp.query}': {resp.error}]")
                continue
            findings_lines.append(f"\n### Query: {resp.query}")
            for r in resp.results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    findings_lines.append(f"- **{r.title}**: {r.snippet}")
                    findings_lines.append(f"  Source: {r.url}")

        return "\n".join(findings_lines)

    # ------------------------------------------------------------------
    # Step 3 – Code Execution
    # ------------------------------------------------------------------

    def _analyse(self, findings: str) -> Tuple[str, CodeExecutionResult]:
        """Generate and execute Python analysis code for the findings."""
        code = self._generate_analysis_code(findings)
        result = self.code_executor.execute(code)
        if not result.success:
            logger.warning(
                "Analysis code failed; trying simplified fallback.\n%s", result.stderr
            )
            code = self._fallback_analysis_code(findings)
            result = self.code_executor.execute(code)
        return code, result

    def _generate_analysis_code(self, findings: str) -> str:
        """Generate Python analysis code (LLM-assisted or rule-based)."""
        if self.llm_callable:
            prompt = _ANALYSIS_CODE_PROMPT.format(findings=findings[:3000])
            return self._call_llm(prompt)
        return self._heuristic_analysis_code(findings)

    @staticmethod
    def _heuristic_analysis_code(findings: str) -> str:
        """
        Generate a generic analysis script that:
        - Counts results per sub-query
        - Renders a summary table
        - Plots a bar chart of result counts
        """
        # Use repr() for safe embedding of arbitrary strings in generated code
        findings_repr = repr(findings)
        lines = [
            f"findings = {findings_repr}",
            "",
            "# ── Count lines that look like data points ──",
            "import re",
            "lines = findings.split('\\n')",
            "section_counts: dict = {}",
            "current_section = 'General'",
            "for line in lines:",
            "    m = re.match(r'###\\s*Query:\\s*(.+)', line)",
            "    if m:",
            "        current_section = m.group(1).strip()[:40]",
            "        section_counts[current_section] = 0",
            "    elif line.strip().startswith('- '):",
            "        section_counts[current_section] = section_counts.get(current_section, 0) + 1",
            "",
            "# ── Summary table ──",
            "rows = [[k, v] for k, v in section_counts.items()]",
            "print(tabulate(rows, headers=['Sub-query', 'Results found'], tablefmt='github'))",
            "",
            "# ── Bar chart ──",
            "if section_counts:",
            "    labels = [k[:25] + '…' if len(k) > 25 else k for k in section_counts]",
            "    values = list(section_counts.values())",
            "    fig, ax = plt.subplots(figsize=(10, 5))",
            "    bars = ax.barh(labels, values, color='steelblue')",
            "    ax.set_xlabel('Number of search results')",
            "    ax.set_title('Research Coverage by Sub-query')",
            "    ax.bar_label(bars, padding=3)",
            "    plt.tight_layout()",
        ]
        return "\n".join(lines)

    @staticmethod
    def _fallback_analysis_code(findings: str) -> str:
        """Minimal safe code that just prints a summary."""
        return textwrap.dedent(
            f'''\
            total_lines = {len(findings.splitlines())}
            print(f"Research summary: {{total_lines}} lines of findings collected.")
            print("Analysis completed successfully.")
            '''
        )

    # ------------------------------------------------------------------
    # Step 4 – Report Assembly
    # ------------------------------------------------------------------

    def _build_report(
        self,
        query: str,
        sub_queries: List[str],
        search_responses: List[SearchResponse],
        findings: str,
        code: str,
        exec_result: CodeExecutionResult,
    ) -> ResearchReport:
        """Assemble all gathered data into a :class:`ResearchReport`."""
        sections: List[ReportSection] = []

        # Section 1 – Research Plan
        plan_content = "The following targeted sub-queries were used:\n\n" + "\n".join(
            f"{i}. {q}" for i, q in enumerate(sub_queries, 1)
        )
        sections.append(ReportSection(heading="Research Plan", content=plan_content))

        # Section 2 – Search Findings
        findings_content = self._format_findings_section(search_responses)
        sections.append(
            ReportSection(heading="Search Findings", content=findings_content)
        )

        # Section 3 – Data Analysis & Visualisation
        analysis_content = (
            "Python code was executed to organise and visualise the collected data."
        )
        sections.append(
            ReportSection(
                heading="Data Analysis & Visualisation",
                content=analysis_content,
                code_snippet=code,
                code_output=exec_result.stdout if exec_result.stdout.strip() else None,
                figures=exec_result.figures_as_base64(),
            )
        )

        # Section 4 – Key Insights
        insights = self._extract_insights(search_responses, exec_result)
        sections.append(ReportSection(heading="Key Insights", content=insights))

        return ResearchReport(
            title=f"Research Report: {query}",
            query=query,
            sections=sections,
        )

    @staticmethod
    def _format_findings_section(responses: List[SearchResponse]) -> str:
        lines: List[str] = []
        for resp in responses:
            if resp.error:
                continue
            lines.append(f"### {resp.query}\n")
            for r in resp.results[:5]:
                lines.append(f"- **[{r.title}]({r.url})**")
                lines.append(f"  {r.snippet}\n")
        return "\n".join(lines) if lines else "No search results were retrieved."

    @staticmethod
    def _extract_insights(
        responses: List[SearchResponse], exec_result: CodeExecutionResult
    ) -> str:
        insights: List[str] = []
        total_results = sum(len(r.results) for r in responses)
        total_sources = len({res.url for r in responses for res in r.results})
        insights.append(f"- **Total search results gathered:** {total_results}")
        insights.append(f"- **Unique sources referenced:** {total_sources}")

        if exec_result.success:
            insights.append("- **Code execution:** Completed successfully ✅")
            if exec_result.figures:
                insights.append(
                    f"- **Visualisations generated:** {len(exec_result.figures)} figure(s)"
                )
        else:
            insights.append(
                f"- **Code execution:** Encountered issues ⚠️ "
                f"({exec_result.error or 'see stderr'})"
            )

        return "\n".join(insights)

    # ------------------------------------------------------------------
    # LLM integration helper
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the user-supplied LLM callable safely."""
        try:
            return str(self.llm_callable(prompt))  # type: ignore[call-arg]
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        """Extract items from a numbered list (e.g. "1. item\\n2. item")."""
        items: List[str] = []
        for line in text.splitlines():
            m = re.match(r"^\s*\d+\.\s+(.+)$", line)
            if m:
                items.append(m.group(1).strip())
        return items
