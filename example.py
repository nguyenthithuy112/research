"""
Example usage of the Autonomous Research & Data Analysis Agent.

Run this script to see the agent in action:

    python example.py

The generated Markdown report is printed to stdout and also saved to
``report.md`` in the current directory.
"""

import logging
import sys

from agent import ResearchAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the latest trends in renewable energy adoption globally?"
    )

    print(f"🔍  Starting research agent for query:\n    {query}\n")
    print("=" * 70)

    agent = ResearchAgent(verbose=True)
    report_md = agent.run(query)

    # Print to console
    print("\n" + "=" * 70)
    print("GENERATED REPORT:")
    print("=" * 70 + "\n")
    print(report_md)

    # Save to file
    output_path = "report.md"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)
    print(f"\n✅  Report saved to: {output_path}")


if __name__ == "__main__":
    main()
