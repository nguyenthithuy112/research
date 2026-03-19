# Autonomous Research & Data Analysis Agent

An AI-powered agent that conducts deep, evidence-based research on any topic
using a strict four-step workflow.

## Workflow

### Step 1 – Planning & Initial Search
Breaks the user's query into 3–5 specific, targeted sub-questions and searches
each one using DuckDuckGo (free, no API key required) or the Google Custom
Search API (when credentials are configured).

### Step 2 – Data Verification & Analysis
Cross-references all search results, de-duplicates sources, and verifies
time-sensitive facts such as the current real-time date.

### Step 3 – Python Code Execution
For any numerical data or statistics, the agent automatically writes and
executes Python code to:
- Organise data into pandas DataFrames
- Compute summary statistics and trends
- Create Matplotlib charts saved as PNG figures
- Format tables with `tabulate`

### Step 4 – Final Report Generation
Synthesises all findings into a professional Markdown document with:
- A table of contents
- Structured headings and bullet points
- Embedded charts (base64-encoded PNGs)
- Cited sources

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the example (uses DuckDuckGo – no API key needed)
python example.py "What are the latest trends in renewable energy?"
```

The report is printed to the console and saved as `report.md`.

---

## Installation

```bash
pip install -r requirements.txt
```

Optional – Google Custom Search API (more reliable):

```bash
export GOOGLE_API_KEY="your-key"
export GOOGLE_CSE_ID="your-cse-id"
```

---

## Usage

```python
from agent import ResearchAgent

agent = ResearchAgent(verbose=True)
report_md = agent.run("What are the latest global AI trends?")
print(report_md)

# Or save directly to a file
agent.run_and_save("Global AI trends", "my_report.md")
```

### With an LLM backend

Pass any callable that accepts a prompt string and returns text.
For example, with the OpenAI SDK:

```python
from openai import OpenAI
from agent import ResearchAgent

client = OpenAI()

def call_gpt(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

agent = ResearchAgent(llm_callable=call_gpt, verbose=True)
report_md = agent.run("Impact of AI on global employment 2024")
```

---

## Project Structure

```
research/
├── agent.py                  # Core autonomous agent (orchestrator)
├── example.py                # Example usage script
├── requirements.txt          # Python dependencies
├── tools/
│   ├── __init__.py
│   ├── search_tool.py        # Web search (DuckDuckGo / Google)
│   ├── code_executor.py      # Safe subprocess-based code runner
│   └── report_generator.py  # Markdown report renderer
└── tests/
    ├── test_search_tool.py
    ├── test_code_executor.py
    ├── test_report_generator.py
    └── test_agent.py
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Configuration

| Environment Variable | Description                             | Default          |
|----------------------|-----------------------------------------|------------------|
| `GOOGLE_API_KEY`     | Google Custom Search API key            | *(not set)*      |
| `GOOGLE_CSE_ID`      | Google Custom Search Engine ID          | *(not set)*      |

When neither variable is set the agent uses DuckDuckGo automatically.
