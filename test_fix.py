import asyncio
import os
import nest_asyncio
nest_asyncio.apply()

import sys

# Add project root to path
sys.path.append(os.getcwd())

from evaluator import ResponseEvaluator

async def main():
    print("Initializing ResponseEvaluator...")
    try:
        evaluator = ResponseEvaluator()
        print("ResponseEvaluator initialized.")
    except Exception as e:
        print(f"FAILED to initialize ResponseEvaluator: {e}")
        return

    print("Testing evaluate_comprehensive...")
    try:
        # Call evaluate_comprehensive instead
        result = await evaluator.evaluate_comprehensive("This is a test response", "This is context")
        print("evaluate_comprehensive executed successfully.")
        print(f"Result keys: {result.keys()}")
    except TypeError as e:
        if "Dataset() takes no arguments" in str(e):
             print(f"FAILED: Still getting Dataset error: {e}")
        else:
             print(f"TypeError occurred (check if related): {e}")
    except Exception as e:
        print(f"Other error occurred (expected if no API key): {e}")
        # Check if it was an inspect-related error or API error
        if "api_key" in str(e) or "authentication" in str(e).lower():
             print("SUCCESS: Evaluator ran but failed on API auth (expected).")
        else:
             print(f"WARNING: Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
