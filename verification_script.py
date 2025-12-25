import inspect
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

import sys
from unittest.mock import MagicMock

# Mock nest_asyncio
sys.modules["nest_asyncio"] = MagicMock()
sys.modules["nest_asyncio"].apply = MagicMock()

# Mock inspect_ai and submodules
mock_inspect = MagicMock()
sys.modules["inspect_ai"] = mock_inspect
sys.modules["inspect_ai.model"] = MagicMock()
sys.modules["inspect_ai.solver"] = MagicMock()
sys.modules["inspect_ai.scorer"] = MagicMock()
sys.modules["inspect_ai.dataset"] = MagicMock()
sys.modules["inspect_ai.core"] = MagicMock() # For Task, etc if needed

# Mock anthropic
sys.modules["anthropic"] = MagicMock()

# Mock dotenv
sys.modules["dotenv"] = MagicMock()

# Mock mcp
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.types"] = MagicMock()

try:
    from eval_framework.core.evaluator import ResponseEvaluator
    from eval_framework.server.tools import get_tools
    print("Imports successful.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

# Check evaluate_comprehensive signature
sig = inspect.signature(ResponseEvaluator.evaluate_comprehensive)
print(f"evaluate_comprehensive signature: {sig}")

if "modified_files" in sig.parameters:
    print("SUCCESS: modified_files parameter found in evaluate_comprehensive.")
else:
    print("FAILURE: modified_files parameter NOT found in evaluate_comprehensive.")
    sys.exit(1)

# Verify context injection logic using mock
from unittest.mock import MagicMock, patch, mock_open
import asyncio

async def test_context_injection():
    evaluator = ResponseEvaluator()
    evaluator.eval = MagicMock() # Mock the inspect_ai eval function if possible, or just the call
    
    # We can't easily mock the internal 'eval' call without patching imports in the module
    # So we will check if the code *looks* correct by inspecting the source or just running it and seeing if it crashes
    
    # Let's try to run it with a mock eval
    # We use AsyncMock for the eval function since it's awaited
    with patch("eval_framework.core.evaluator.eval", new_callable=MagicMock) as mock_eval:
        # Create a future for the return value
        f = asyncio.Future()
        f.set_result([
            MagicMock(results=[MagicMock(scores={"hallucination_scorer": MagicMock(value=1.0, explanation="ok")})]),
            MagicMock(results=[MagicMock(scores={"tool_consistency_scorer": MagicMock(value=1.0, explanation="ok")})]),
            MagicMock(results=[MagicMock(scores={"petri_scorer": MagicMock(value=1.0, explanation="ok", metadata={})})])
        ])
        mock_eval.return_value = f
        
        # Mock file opening
        with patch("builtins.open", mock_open(read_data="mock file content")):
            modified_files = ["/tmp/test_file.txt"]
            await evaluator.evaluate_comprehensive(
                response="test response",
                context="original context",
                modified_files=modified_files
            )
            
            # Check if eval was called
            if mock_eval.called:
                print("SUCCESS: eval() was called.")
                
                # Inspect arguments to see if context was updated in the Task
                # We check the calls to Sample() since Task receives a dataset of Samples
                # Note: Sample is imported from inspect_ai.dataset
                
                # Retrieve the mocked Sample class
                # Since we mocked sys.modules["inspect_ai.dataset"], we need to get it from there
                # However, evaluator.py does: from inspect_ai.dataset import Sample
                # So we can check the mock used in evaluator.
                
                # Better yet, since we don't have reference to the exact mock object effectively used by evaluator 
                # (since it imports it), we can try to find it via sys.modules or just inspect what was passed to eval keys?
                # No, simpler: check mock_inspect.Task.call_args_list or similar if we can.
                
                # Actually, failing that, let's look at the Task constructor args.
                # Task(dataset=[Sample(...)], ...)
                # dataset is a list. The element 0 is the Sample object (which is a mock).
                # But we can't get the 'input' attribute from a Mock object created by the constructor unless we attach it.
                
                # Strategy: inspect the ARGUMENTS passed to the calls we mocked.
                # Only way is if we have access to the mock object that was called.
                
                # In verification_script, we have:
                # sys.modules["inspect_ai.dataset"] = MagicMock()
                mock_dataset_module = sys.modules["inspect_ai.dataset"]
                mock_Sample = mock_dataset_module.Sample
                
                found_context = False
                for call in mock_Sample.call_args_list:
                    # Sample(input=..., target=..., metadata=...)
                    # Check kwargs 'input' or args[0]
                    args, kwargs = call
                    input_arg = kwargs.get('input') or (args[0] if args else "")
                    
                    if "[MODIFIED FILES CONTENT]" in str(input_arg) and "mock file content" in str(input_arg):
                        found_context = True
                        break
                
                if found_context:
                    print("SUCCESS: Context correctly contains injected file content.")
                else:
                    print(f"FAILURE: Context missing injected content in Sample calls.")
                    # print calls for debugging
                    # print(f"Calls: {mock_Sample.call_args_list}")
            else:
                print("FAILURE: eval() was not called.")

# Run the async test
try:
    asyncio.run(test_context_injection())
except Exception as e:
    print(f"FAILURE: Test raised exception: {e}")
    sys.exit(1)

print("Verification complete.")
