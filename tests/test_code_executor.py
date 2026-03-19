"""Tests for the CodeExecutor."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from tools.code_executor import CodeExecutionResult, CodeExecutor


class TestCodeExecutionResult:
    def test_success_true_when_no_errors(self):
        result = CodeExecutionResult(code="x=1", stdout="1\n")
        assert result.success

    def test_success_false_on_timeout(self):
        result = CodeExecutionResult(code="x=1", timed_out=True)
        assert not result.success

    def test_success_false_on_error(self):
        result = CodeExecutionResult(code="x=1", error="SyntaxError")
        assert not result.success

    def test_success_false_on_stderr(self):
        result = CodeExecutionResult(code="x=1", stderr="warning: something")
        assert not result.success

    def test_summary_timeout(self):
        result = CodeExecutionResult(code="x=1", timed_out=True)
        assert "timed out" in result.summary().lower()

    def test_summary_error(self):
        result = CodeExecutionResult(code="x=1", error="ZeroDivisionError")
        assert "ZeroDivisionError" in result.summary()

    def test_summary_no_output(self):
        result = CodeExecutionResult(code="x=1")
        assert "successfully" in result.summary().lower()

    def test_figures_as_base64_missing_file(self, tmp_path):
        result = CodeExecutionResult(code="x=1", figures={})
        b64 = result.figures_as_base64()
        assert b64 == {}


class TestCodeExecutor:
    def test_empty_code_returns_error(self):
        executor = CodeExecutor()
        result = executor.execute("")
        assert result.error == "Empty code provided."

    def test_whitespace_code_returns_error(self):
        executor = CodeExecutor()
        result = executor.execute("   \n  ")
        assert result.error == "Empty code provided."

    def test_simple_print(self):
        executor = CodeExecutor()
        result = executor.execute("print('hello world')")
        assert "hello world" in result.stdout

    def test_math_computation(self):
        executor = CodeExecutor()
        result = executor.execute("print(2 + 2)")
        assert "4" in result.stdout

    def test_syntax_error(self):
        executor = CodeExecutor()
        result = executor.execute("def broken(:")
        assert not result.success

    def test_runtime_error(self):
        executor = CodeExecutor()
        result = executor.execute("raise ValueError('test error')")
        assert not result.success

    def test_timeout(self):
        executor = CodeExecutor(timeout=1)
        result = executor.execute("import time; time.sleep(60)")
        assert result.timed_out

    def test_matplotlib_figure_saved(self):
        executor = CodeExecutor()
        code = (
            "import matplotlib.pyplot as plt\n"
            "plt.figure()\n"
            "plt.plot([1, 2, 3], [4, 5, 6])\n"
            "plt.title('Test Chart')\n"
        )
        result = executor.execute(code)
        assert len(result.figures) == 1
        name = next(iter(result.figures))
        assert name.endswith(".png")

    def test_figures_as_base64(self):
        executor = CodeExecutor()
        code = (
            "import matplotlib.pyplot as plt\n"
            "plt.figure()\n"
            "plt.plot([1, 2], [3, 4])\n"
        )
        result = executor.execute(code)
        if result.figures:
            b64 = result.figures_as_base64()
            assert len(b64) == 1
            name, data = next(iter(b64.items()))
            assert name.endswith(".png")
            assert len(data) > 100  # non-trivial base64 content

    def test_pandas_dataframe(self):
        executor = CodeExecutor()
        code = (
            "import pandas as pd\n"
            "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})\n"
            "print(df.to_string())\n"
        )
        result = executor.execute(code)
        assert "a" in result.stdout
        assert "b" in result.stdout

    def test_max_output_truncation(self):
        executor = CodeExecutor(max_output_bytes=50)
        code = "print('x' * 200)"
        result = executor.execute(code)
        # stdout should be truncated to 50 bytes
        assert len(result.stdout.encode()) <= 50

    def test_build_script_contains_preamble(self):
        full = CodeExecutor._build_script("print(1)", "/tmp/out")
        assert "matplotlib" in full
        assert "pandas" in full
        assert "numpy" in full
        assert "_OUTPUT_DIR" in full
