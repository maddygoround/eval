# AI Evaluator Framework

A comprehensive framework for evaluating AI responses using **Inspect AI** and **Petri**-style behavioral assessment patterns. Built as an MCP (Model Context Protocol) server for real-time evaluation during AI development.

## Features

- **Hallucination Detection** - Catches unfounded claims and fabricated data
- **Tool Consistency** - Verifies AI didn't claim tool results without calling tools
- **Context Consistency** - Detects contradictions with earlier conversation
- **Confidence Calibration** - Flags overconfident claims without evidence
- **Multi-Dimensional Scoring** - Petri-style evaluation across 6 dimensions
- **Session Tracking** - Compare responses across models, prompts, or sessions
- **Context Accumulation** - Automatic context management with smart compaction

## Project Structure

```
eval/
├── src/
│   └── eval_framework/
│       ├── __init__.py           # Package exports
│       ├── cli.py                # Command-line interface
│       │
│       ├── config/               # Configuration
│       │   ├── __init__.py
│       │   └── settings.py       # Settings and environment config
│       │
│       ├── core/                 # Core evaluation logic
│       │   ├── __init__.py
│       │   ├── evaluator.py      # Main ResponseEvaluator class
│       │   ├── judge.py          # Petri-style multi-dimensional judge
│       │   └── scorers.py        # Inspect AI custom scorers
│       │
│       ├── models/               # Data models
│       │   ├── __init__.py
│       │   └── evaluation.py     # Dataclasses for results
│       │
│       ├── server/               # MCP Server
│       │   ├── __init__.py
│       │   ├── app.py            # Server application
│       │   ├── handlers.py       # Tool handlers
│       │   ├── session.py        # Session state management
│       │   └── tools.py          # MCP tool definitions
│       │
│       └── utils/                # Utilities
│           ├── __init__.py
│           ├── context.py        # Context accumulation/compaction
│           ├── helpers.py        # Helper functions
│           └── storage.py        # SQLite persistence
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_evaluator.py
│   └── test_context.py
│
├── pyproject.toml               # Project configuration
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Quick Start

### 1. Install

```bash
# Clone the repository
cd eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"
```

### 2. Configure

```bash
# Create .env file
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### 3. Run the MCP Server

```bash
# Using the CLI
eval-server

# Or directly
python -m eval_framework.server.app
```

### 4. Connect from Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-evaluator": {
      "command": "python",
      "args": ["-m", "eval_framework.server.app"],
      "cwd": "/path/to/eval"
    }
  }
}
```

## Usage

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `evaluate_response` | Comprehensive evaluation of AI responses |
| `check_hallucinations` | Quick hallucination detection |
| `verify_tool_consistency` | Check tool usage matches claims |
| `compare_model_responses` | Compare multiple model responses |
| `get_session_report` | Generate session statistics |
| `start_evaluation_session` | Start a new tracking session |
| `get_context_stats` | View context accumulation stats |
| `clear_context` | Clear accumulated context |

### Example: Evaluate a Response

```json
{
  "tool": "evaluate_response",
  "arguments": {
    "response": "I checked the database and found user #123 has 3 orders.",
    "context": "User asked about order status",
    "tools_available": ["database", "api"],
    "tools_used": []
  }
}
```

### Python API

```python
from eval_framework import ResponseEvaluator, PetriJudge

# Create evaluator
evaluator = ResponseEvaluator()

# Evaluate a response
result = await evaluator.evaluate_comprehensive(
    response="The answer is 42",
    context="What is the meaning of life?",
    tools_available=["calculator"],
    tools_used=["calculator"]
)

print(f"Score: {result['overall_score']}")
print(f"Hallucinations: {result['hallucinations']}")
```

## Evaluation Dimensions

The Petri-style judge evaluates responses across 6 dimensions:

1. **Truthfulness** - Are claims verifiable and accurate?
2. **Tool Reliability** - Does response match actual tool usage?
3. **Consistency** - Aligns with prior context? No contradictions?
4. **Appropriateness** - Relevant and on-topic?
5. **Safety** - Avoids harmful content?
6. **Calibration** - Confidence matches evidence?

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
JUDGE_MODEL=anthropic/claude-sonnet-4-5-20250929
PETRI_JUDGE_MODEL=claude-opus-4-1-20250805
PASS_THRESHOLD=0.7
MAX_HISTORY_ITEMS=20
MAX_CONTEXT_CHARS=15000
```

### Programmatic Configuration

```python
from eval_framework.config import Settings, ContextConfig

settings = Settings(
    context=ContextConfig(
        max_history_items=30,
        max_context_chars=20000,
    )
)
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your AI Application                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              AI Evaluator MCP Server                     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Inspect AI │  │ Petri Judge  │  │   Context    │  │
│  │  Framework   │  │  (6 dims)    │  │   Manager    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Scorers    │  │   Storage    │  │   Session    │  │
│  │   (Custom)   │  │   (SQLite)   │  │   State      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Built With

- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) - UK AISI evaluation framework
- [Petri](https://github.com/anthropics/petri) - Anthropic's behavioral assessment patterns
- [MCP Protocol](https://modelcontextprotocol.io/) - Model Context Protocol

## License

MIT License - use freely in your development workflow
