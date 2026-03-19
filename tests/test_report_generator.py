"""Tests for the ReportGenerator."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import base64

import pytest

from tools.report_generator import ReportGenerator, ReportSection, ResearchReport


def _make_report(num_sections: int = 2) -> ResearchReport:
    sections = [
        ReportSection(
            heading=f"Section {i}",
            content=f"Content of section {i}.",
        )
        for i in range(1, num_sections + 1)
    ]
    return ResearchReport(title="Test Report", query="test query", sections=sections)


class TestReportSection:
    def test_defaults(self):
        s = ReportSection(heading="Heading", content="Content")
        assert s.code_snippet is None
        assert s.code_output is None
        assert s.figures == {}


class TestResearchReport:
    def test_generated_at_set(self):
        r = ResearchReport(title="T", query="q")
        assert "UTC" in r.generated_at

    def test_sections_default_empty(self):
        r = ResearchReport(title="T", query="q")
        assert r.sections == []


class TestReportGenerator:
    def test_render_contains_title(self):
        gen = ReportGenerator()
        report = _make_report()
        md = gen.render(report)
        assert "# Test Report" in md

    def test_render_contains_query(self):
        gen = ReportGenerator()
        report = _make_report()
        md = gen.render(report)
        assert "test query" in md

    def test_render_contains_section_headings(self):
        gen = ReportGenerator()
        report = _make_report(3)
        md = gen.render(report)
        assert "## Section 1" in md
        assert "## Section 2" in md
        assert "## Section 3" in md

    def test_render_toc_for_multiple_sections(self):
        gen = ReportGenerator()
        report = _make_report(3)
        md = gen.render(report)
        assert "Table of Contents" in md

    def test_render_no_toc_for_single_section(self):
        gen = ReportGenerator()
        report = _make_report(1)
        md = gen.render(report)
        assert "Table of Contents" not in md

    def test_render_code_snippet(self):
        gen = ReportGenerator()
        section = ReportSection(
            heading="Analysis",
            content="Analysis done.",
            code_snippet="print('hello')",
            code_output="hello",
        )
        report = ResearchReport(title="T", query="q", sections=[section])
        md = gen.render(report)
        assert "```python" in md
        assert "print('hello')" in md
        assert "hello" in md

    def test_render_embedded_figure(self):
        gen = ReportGenerator(embed_figures=True)
        fake_png = base64.b64encode(b"fake_png_data").decode()
        section = ReportSection(
            heading="Viz",
            content="Chart below.",
            figures={"chart.png": fake_png},
        )
        report = ResearchReport(title="T", query="q", sections=[section])
        md = gen.render(report)
        assert "data:image/png;base64," in md
        assert fake_png in md

    def test_render_saved_figure(self, tmp_path):
        gen = ReportGenerator(embed_figures=False, output_dir=str(tmp_path))
        fake_png = base64.b64encode(b"fake_png_data").decode()
        section = ReportSection(
            heading="Viz",
            content="Chart below.",
            figures={"chart.png": fake_png},
        )
        report = ResearchReport(title="T", query="q", sections=[section])
        md = gen.render(report)
        saved = tmp_path / "chart.png"
        assert saved.exists()
        assert "chart.png" in md

    def test_save_creates_file(self, tmp_path):
        gen = ReportGenerator()
        report = _make_report()
        path = gen.save(report, str(tmp_path / "output"))
        assert path.endswith(".md")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Test Report" in content

    def test_save_appends_md_extension(self, tmp_path):
        gen = ReportGenerator()
        report = _make_report()
        path = gen.save(report, str(tmp_path / "myreport"))
        assert path.endswith(".md")

    def test_to_anchor(self):
        assert ReportGenerator._to_anchor("Key Insights") == "key-insights"
        assert ReportGenerator._to_anchor("Data Analysis & Viz") == "data-analysis--viz"
        assert ReportGenerator._to_anchor("Step (1)") == "step-1"

    def test_footer_in_report(self):
        gen = ReportGenerator()
        md = gen.render(_make_report())
        assert "Autonomous Research" in md
