"""
Example Client for AI Evaluator MCP Server
Shows how to integrate the evaluator into your development workflow
"""

import asyncio
import json
from typing import Any, Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class AIEvaluatorClient:
    """
    Client wrapper for the AI Evaluator MCP Server
    Makes it easy to use in your development code
    """
    
    def __init__(self):
        self.session: ClientSession | None = None
    
    async def connect(self):
        """Connect to the MCP server"""
        
        server_params = StdioServerParameters(
            command="python",
            args=["server.py"]
        )
        
        stdio_transport = await stdio_client(server_params)
        self.session = ClientSession(*stdio_transport)
        await self.session.initialize()
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.session:
            await self.session.__aexit__(None, None, None)
    
    async def evaluate(
        self,
        response: str,
        context: str = "",
        tools_available: List[str] = None,
        tools_used: List[str] = None,
        model: str = "unknown"
    ) -> Dict[str, Any]:
        """Evaluate an AI response"""
        
        result = await self.session.call_tool(
            "evaluate_response",
            {
                "response": response,
                "context": context,
                "tools_available": tools_available or [],
                "tools_used": tools_used or [],
                "model": model
            }
        )
        
        return json.loads(result.content[0].text)
    
    async def check_hallucinations(
        self,
        response: str,
        strict: bool = False
    ) -> Dict[str, Any]:
        """Quick hallucination check"""
        
        result = await self.session.call_tool(
            "check_hallucinations",
            {
                "response": response,
                "strict_mode": strict
            }
        )
        
        return json.loads(result.content[0].text)
    
    async def verify_tools(
        self,
        response: str,
        tools_available: List[str],
        tools_used: List[str]
    ) -> Dict[str, Any]:
        """Verify tool usage consistency"""
        
        result = await self.session.call_tool(
            "verify_tool_consistency",
            {
                "response": response,
                "tools_available": tools_available,
                "tools_used": tools_used
            }
        )
        
        return json.loads(result.content[0].text)
    
    async def compare_models(
        self,
        responses: List[Dict[str, str]],
        context: str = ""
    ) -> Dict[str, Any]:
        """Compare responses from different models"""
        
        result = await self.session.call_tool(
            "compare_model_responses",
            {
                "responses": responses,
                "context": context
            }
        )
        
        return json.loads(result.content[0].text)
    
    async def get_report(
        self,
        session_id: str = None,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """Get session report"""
        
        result = await self.session.call_tool(
            "get_session_report",
            {
                "session_id": session_id,
                "detailed": detailed
            }
        )
        
        return json.loads(result.content[0].text)
    
    async def start_session(
        self,
        name: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """Start new evaluation session"""
        
        result = await self.session.call_tool(
            "start_evaluation_session",
            {
                "name": name,
                "description": description
            }
        )
        
        return json.loads(result.content[0].text)


# ============================================================================
# Usage Examples
# ============================================================================

async def example_basic_evaluation():
    """Example 1: Basic response evaluation"""
    
    client = AIEvaluatorClient()
    await client.connect()
    
    try:
        # Evaluate a response
        result = await client.evaluate(
            response="I checked the database and user #123 has 3 active orders.",
            context="User asked about order status",
            tools_available=["database", "api"],
            tools_used=[],  # Oops! Database wasn't called
            model="claude-sonnet-4"
        )
        
        print("Evaluation Result:")
        print(f"  Overall Score: {result['overall_score']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Pass: {result['pass']}")
        
        if result['warnings']:
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  {warning}")
        
        if result['suggestions']:
            print("\nSuggestions:")
            for suggestion in result['suggestions']:
                print(f"  {suggestion}")
    
    finally:
        await client.disconnect()


async def example_hallucination_check():
    """Example 2: Quick hallucination check"""
    
    client = AIEvaluatorClient()
    await client.connect()
    
    try:
        result = await client.check_hallucinations(
            response="""
            Our API handles 50 billion requests per second with 99.999% uptime.
            Studies show this is 40% more efficient than competitors.
            The built-in super_cache() function makes this possible.
            """,
            strict=True
        )
        
        print(f"Found {result['hallucinations_found']} hallucination(s):")
        for h in result['hallucinations']:
            print(f"  - {h['claim']} (confidence: {h['confidence']})")
    
    finally:
        await client.disconnect()


async def example_model_comparison():
    """Example 3: Compare different models"""
    
    client = AIEvaluatorClient()
    await client.connect()
    
    try:
        result = await client.compare_models(
            context="Calculate the compound interest on $10,000 at 5% for 10 years",
            responses=[
                {
                    "model": "gpt-4",
                    "response": "The final amount will be approximately $16,289"
                },
                {
                    "model": "claude-sonnet-4",
                    "response": "Let me calculate that using the compound interest formula. [calls calculator] The final amount is $16,288.95"
                },
                {
                    "model": "gemini-pro",
                    "response": "You'll definitely get $20,000 after 10 years"
                }
            ]
        )
        
        print("Model Comparison:")
        for item in result['comparison']:
            print(f"\n{item['model']}:")
            print(f"  Score: {item['score']}")
            print(f"  Hallucinations: {item['hallucination_count']}")
        
        print(f"\nRecommendation: {result['recommendation']}")
    
    finally:
        await client.disconnect()


async def example_testing_workflow():
    """Example 4: Integration into testing workflow"""
    
    client = AIEvaluatorClient()
    await client.connect()
    
    try:
        # Start a testing session
        session = await client.start_session(
            name="testing-payment-integration",
            description="Testing new payment processing prompts"
        )
        print(f"Started session: {session['session_id']}\n")
        
        # Test multiple scenarios
        test_cases = [
            {
                "prompt": "Process refund for order #123",
                "response": "I've processed the refund. The customer will receive $99.99 back.",
                "tools_used": ["payment_api"]  # Good - actually called the tool
            },
            {
                "prompt": "Check payment status",
                "response": "The payment was successful and the funds are in your account.",
                "tools_used": []  # Bad - claims knowledge without checking
            },
            {
                "prompt": "Cancel subscription",
                "response": "I've cancelled the subscription. No refund will be issued.",
                "tools_used": ["subscription_api"]  # Good
            }
        ]
        
        results = []
        for i, test in enumerate(test_cases, 1):
            print(f"Testing case {i}/{len(test_cases)}...")
            
            result = await client.evaluate(
                response=test["response"],
                context=test["prompt"],
                tools_available=["payment_api", "subscription_api"],
                tools_used=test["tools_used"]
            )
            
            results.append({
                "case": i,
                "passed": result["pass"],
                "score": result["overall_score"]
            })
            
            if not result["pass"]:
                print(f"  ❌ FAILED - Score: {result['overall_score']}")
                for warning in result['warnings']:
                    print(f"     {warning}")
            else:
                print(f"  ✅ PASSED - Score: {result['overall_score']}")
        
        # Get session report
        print("\n" + "="*60)
        report = await client.get_report(detailed=True)
        print(f"Session Report:")
        print(f"  Total Tests: {report['total_evaluations']}")
        print(f"  Pass Rate: {report['pass_rate']*100:.1f}%")
        print(f"  Average Score: {report['average_score']:.3f}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  {rec}")
    
    finally:
        await client.disconnect()


async def example_ci_cd_integration():
    """Example 5: CI/CD quality gate"""
    
    client = AIEvaluatorClient()
    await client.connect()
    
    try:
        # Evaluate your agent's responses
        responses_to_test = [
            "Process this payment...",
            "Check user balance...",
            "Generate report..."
        ]
        
        all_passed = True
        total_score = 0
        
        for response in responses_to_test:
            result = await client.evaluate(
                response=response,
                tools_available=["payment", "database", "api"]
            )
            
            total_score += result["overall_score"]
            
            if not result["pass"]:
                all_passed = False
                print(f"❌ Quality gate FAILED for: {response[:50]}...")
        
        avg_score = total_score / len(responses_to_test)
        
        if all_passed and avg_score >= 0.75:
            print("✅ Quality gate PASSED - deployment allowed")
            return 0  # Success exit code
        else:
            print(f"❌ Quality gate FAILED - avg score {avg_score:.2f} < 0.75")
            return 1  # Failure exit code
    
    finally:
        await client.disconnect()


# ============================================================================
# Main - Run Examples
# ============================================================================

async def main():
    print("AI Evaluator Client Examples\n")
    print("="*60)
    
    print("\n1. Basic Evaluation")
    print("-"*60)
    await example_basic_evaluation()
    
    print("\n\n2. Hallucination Check")
    print("-"*60)
    await example_hallucination_check()
    
    print("\n\n3. Model Comparison")
    print("-"*60)
    await example_model_comparison()
    
    print("\n\n4. Testing Workflow")
    print("-"*60)
    await example_testing_workflow()


if __name__ == "__main__":
    asyncio.run(main())
