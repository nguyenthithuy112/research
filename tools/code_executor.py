"""
Safe Python code execution tool for the Autonomous Research & Data Analysis Agent.

Runs user-supplied code in an isolated subprocess with a configurable timeout
so that long-running or infinite loops cannot block the agent.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default execution limits
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024  # 1 MB

# Preamble injected before user code to set up the environment.
_PREAMBLE = textwrap.dedent(
    """\
    import sys, os, warnings, io, base64
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from tabulate import tabulate

    _FIGURE_PATHS = []

    def _save_current_figures(output_dir):
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for idx, fig in enumerate(figs):
            path = os.path.join(output_dir, f"figure_{idx}.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            _FIGURE_PATHS.append(path)
        plt.close("all")

    """
)

_POSTAMBLE = textwrap.dedent(
    """\

    # Save any figures that were created
    import os as _os
    _save_current_figures(_OUTPUT_DIR)
    # Write figure paths to a manifest
    with open(_os.path.join(_OUTPUT_DIR, "figures.txt"), "w") as _f:
        _f.write("\\n".join(_FIGURE_PATHS))
    """
)


@dataclass
class CodeExecutionResult:
    """Result of a code execution."""

    code: str
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: Optional[str] = None
    figures: Dict[str, bytes] = field(default_factory=dict)
    """Mapping of figure filename → raw PNG bytes captured during execution."""

    @property
    def success(self) -> bool:
        return not self.timed_out and self.error is None and not self.stderr.strip()

    def summary(self) -> str:
        """Return a concise human-readable summary of the execution result."""
        lines: List[str] = []
        if self.timed_out:
            lines.append("⚠️ **Execution timed out.**")
        if self.error:
            lines.append(f"❌ **Error:** {self.error}")
        if self.stderr.strip():
            lines.append(f"```\n{self.stderr.strip()}\n```")
        if self.stdout.strip():
            lines.append(f"**Output:**\n```\n{self.stdout.strip()}\n```")
        if self.figures:
            lines.append(f"**Generated {len(self.figures)} figure(s).**")
        if not lines:
            lines.append("✅ Code executed successfully with no output.")
        return "\n".join(lines)

    def figures_as_base64(self) -> Dict[str, str]:
        """Return a mapping of figure filename → base64-encoded PNG string."""
        return {
            name: base64.b64encode(data).decode()
            for name, data in self.figures.items()
        }


class CodeExecutor:
    """
    Executes Python code snippets in a sandboxed subprocess.

    The executor injects a standard preamble that imports commonly used
    data-science libraries (``matplotlib``, ``pandas``, ``numpy``,
    ``tabulate``) and configures ``matplotlib`` to use the non-interactive
    ``Agg`` backend so that figures can be saved to files.

    Parameters
    ----------
    timeout:
        Maximum wall-clock seconds allowed per execution (default: 30).
    max_output_bytes:
        Maximum size (in bytes) of captured stdout/stderr (default: 1 MB).
    python_executable:
        Path to the Python interpreter to use (defaults to the current one).
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        python_executable: Optional[str] = None,
    ) -> None:
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes
        self.python_executable = python_executable or sys.executable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, code: str) -> CodeExecutionResult:
        """
        Execute a Python code snippet and return the result.

        Parameters
        ----------
        code:
            The Python source code to execute.

        Returns
        -------
        CodeExecutionResult
            Contains stdout, stderr, any generated figures, and error info.
        """
        if not code or not code.strip():
            return CodeExecutionResult(code=code, error="Empty code provided.")

        with tempfile.TemporaryDirectory(prefix="agent_exec_") as tmpdir:
            script_path = os.path.join(tmpdir, "user_code.py")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            full_code = self._build_script(code, output_dir)
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(full_code)

            result = self._run_script(script_path, code)
            # Read figure bytes while the temp directory still exists
            result.figures = self._collect_figures(output_dir)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_script(user_code: str, output_dir: str) -> str:
        """Wrap the user code with the standard preamble and postamble."""
        return (
            _PREAMBLE
            + f"_OUTPUT_DIR = {repr(output_dir)}\n\n"
            + user_code
            + "\n"
            + _POSTAMBLE
        )

    def _run_script(self, script_path: str, original_code: str) -> CodeExecutionResult:
        """Run the script as a subprocess and capture output."""
        try:
            proc = subprocess.run(
                [self.python_executable, script_path],
                capture_output=True,
                timeout=self.timeout,
            )
            stdout = proc.stdout[: self.max_output_bytes].decode("utf-8", errors="replace")
            stderr = proc.stderr[: self.max_output_bytes].decode("utf-8", errors="replace")
            error: Optional[str] = None
            if proc.returncode != 0:
                error = f"Process exited with code {proc.returncode}"
            return CodeExecutionResult(
                code=original_code,
                stdout=stdout,
                stderr=stderr,
                error=error,
            )
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                code=original_code,
                timed_out=True,
                error=f"Execution timed out after {self.timeout} seconds.",
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Unexpected error during code execution: %s", exc)
            return CodeExecutionResult(code=original_code, error=str(exc))

    @staticmethod
    def _collect_figures(output_dir: str) -> Dict[str, bytes]:
        """Read figure PNG files from the output directory and return their bytes."""
        manifest = os.path.join(output_dir, "figures.txt")
        figures: Dict[str, bytes] = {}
        if os.path.exists(manifest):
            with open(manifest, encoding="utf-8") as fh:
                for line in fh:
                    path = line.strip()
                    if path and os.path.exists(path):
                        try:
                            with open(path, "rb") as img:
                                figures[Path(path).name] = img.read()
                        except OSError as exc:
                            logger.warning("Could not read figure %s: %s", path, exc)
        return figures
