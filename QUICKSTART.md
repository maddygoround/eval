# 🚀 Quick Start Guide

Get the AI Evaluator up and running in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- Anthropic API key

## Installation

```bash
# 1. Clone or download the repository
cd ai-evaluator-mcp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Test It Out

### Option 1: Run Example Client

```bash
python example_client.py
```

This will run through several examples showing different evaluation scenarios.

### Option 2: Use in Claude Desktop

1. Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-evaluator": {
      "command": "python",
      "args": ["/full/path/to/ai-evaluator-mcp/server.py"]
    }
  }
}
```

2. Restart Claude Desktop

3. Try asking Claude: "Can you evaluate this response: 'I checked the database and found 3 orders' - but no database tool was called"

### Option 3: Direct Python Integration

```python
import asyncio
from example_client import AIEvaluatorClient

async def test():
    client = AIEvaluatorClient()
    await client.connect()
    
    result = await client.evaluate(
        response="The answer is definitely 42",
        context="What is 2+2?",
        tools_available=[],
        tools_used=[]
    )
    
    print(f"Score: {result['overall_score']}")
    print(f"Pass: {result['pass']}")
    print(f"Warnings: {result['warnings']}")
    
    await client.disconnect()

asyncio.run(test())
```

## Common Use Cases

### 1. Testing Your Agent

```python
# Your agent code
response = await my_agent.process(user_input)

# Evaluate it
result = await evaluator.evaluate(
    response=response,
    tools_used=my_agent.tools_called
)

if not result["pass"]:
    print("⚠️ Quality issues detected!")
```

### 2. Comparing Models

```python
result = await client.compare_models(
    context="Calculate 10 * 15",
    responses=[
        {"model": "gpt-4", "response": gpt4_response},
        {"model": "claude", "response": claude_response},
        {"model": "gemini", "response": gemini_response}
    ]
)

print(f"Best model: {result['best_model']}")
```

### 3. CI/CD Quality Gate

```bash
# In your CI pipeline
python -c "
from example_client import example_ci_cd_integration
import asyncio
exit_code = asyncio.run(example_ci_cd_integration())
exit(exit_code)
"
```

## What Gets Checked

✅ **Hallucinations** - Unfounded claims, made-up stats  
✅ **Tool Mismatches** - Claims tool results without calling tools  
✅ **Contradictions** - Conflicts with earlier context  
✅ **Overconfidence** - Absolute claims without evidence  
✅ **Multi-dimensional** - 6 Petri-style quality dimensions  

## Interpreting Results

### Scores

- **0.8-1.0**: Excellent quality ✅
- **0.6-0.8**: Good, minor issues ⚠️
- **0.4-0.6**: Significant issues 🟡
- **0.0-0.4**: Major problems ❌

### Risk Levels

- **Low**: Safe to use
- **Medium**: Review suggested
- **High**: Should not use without fixes

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [example_client.py](example_client.py) for integration patterns
3. Run tests: `pytest test_evaluator.py`
4. Customize evaluation criteria in `judge.py`

## Troubleshooting

**"ModuleNotFoundError: No module named 'mcp'"**
→ Run `pip install -r requirements.txt`

**"Error: ANTHROPIC_API_KEY not found"**
→ Create `.env` file with your API key

**"Judge model timeout"**
→ Try switching to Sonnet instead of Opus in `.env`:
```
JUDGE_MODEL=claude-sonnet-4-5-20250929
```

## Support

- GitHub Issues: Report bugs
- Examples: Check `example_client.py`
- Tests: See `test_evaluator.py`

Happy evaluating! 🎯
