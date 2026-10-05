#!/usr/bin/env python3
"""
Example script to test the LLM-as-judge evaluation system.

This script demonstrates how to use both simple and detailed evaluation modes
to assess agent responses.

Usage:
    python examples/test_evaluation.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.evaluation import evaluate_response, evaluate_response_detailed


# Test cases
TEST_CASES = [
    {
        "name": "Good Financial Response",
        "message": "My bill is wrong, it charged $150 more",
        "work_result": (
            "Hello! I checked your bill and identified a duplicate charge of "
            "$150. We are already processing the refund which will be credited within 2 "
            "business days. You will receive a confirmation email shortly."
        ),
        "category": "financial",
        "priority": "high",
    },
    {
        "name": "Incomplete Technical Response",
        "work_result": "The problem has been resolved. Try again.",
        "message": "I cannot access the system, getting error 500",
        "category": "software",
        "priority": "urgent",
    },
    {
        "name": "Excellent Software Response",
        "message": "The system is slow today",
        "work_result": (
            "We identified a usage spike that was affecting performance. "
            "We have already scaled additional resources and the system is operating normally. "
            "If the slowness persists, please notify us through support."
        ),
        "category": "software",
        "priority": "medium",
    },
    {
        "name": "Off-topic Response",
        "message": "What are the support hours?",
        "work_result": (
            "Your payment has been successfully processed and you will receive a "
            "confirmation email within 24 hours."
        ),
        "category": "financial",
        "priority": "low",
    },
]


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"\n{'-' * 80}\n")


def test_simple_evaluation():
    """Test simple evaluation mode."""
    print_separator("SIMPLE EVALUATION MODE")
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print(f"Request: {test_case['message']}")
        print(f"Response: {test_case['work_result']}")
        print()
        
        result = evaluate_response(
            message=test_case["message"],
            work_result=test_case["work_result"],
            category=test_case["category"],
            priority=test_case.get("priority", "medium"),
            mode="simple",
        )
        
        status = "✓ APPROVED" if result.approved else "✗ REJECTED"
        print(f"Result: {status}")
        
        if result.overall_score > 0:
            print(f"Score: {result.overall_score:.2f}/5.0")
        
        if result.rejection_reason:
            print(f"Rejection reason: {result.rejection_reason}")
        
        print_separator()


def test_detailed_evaluation():
    """Test detailed evaluation mode."""
    print_separator("DETAILED EVALUATION MODE")
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print(f"Request: {test_case['message']}")
        print(f"Response: {test_case['work_result']}")
        print()
        
        result = evaluate_response_detailed(
            message=test_case["message"],
            work_result=test_case["work_result"],
            category=test_case["category"],
            priority=test_case.get("priority", "medium"),
        )
        
        status = "✓ APPROVED" if result.approved else "✗ REJECTED"
        print(f"Result: {status}")
        print(f"Overall Score: {result.overall_score:.2f}/5.0")
        print()
        
        print("Scores by Criterion:")
        print(f"  - Relevance:    {result.relevance.score}/5 - {result.relevance.justification}")
        print(f"  - Accuracy:     {result.accuracy.score}/5 - {result.accuracy.justification}")
        print(f"  - Completeness: {result.completeness.score}/5 - {result.completeness.justification}")
        print(f"  - Clarity:      {result.clarity.score}/5 - {result.clarity.justification}")
        print(f"  - Tone:         {result.tone.score}/5 - {result.tone.justification}")
        print(f"  - Conciseness:  {result.conciseness.score}/5 - {result.conciseness.justification}")
        print()
        
        print(f"Summary: {result.summary}")
        
        if result.critical_issues:
            print("\nCritical Issues:")
            for issue in result.critical_issues:
                print(f"  • {issue}")
        
        if result.improvement_suggestions:
            print("\nImprovement Suggestions:")
            for suggestion in result.improvement_suggestions:
                print(f"  • {suggestion}")
        
        print_separator()


def main():
    """Run all evaluation tests."""
    print("\n" + "=" * 80)
    print("  LLM-as-Judge Evaluation System Test")
    print("=" * 80)
    print("\nThis script tests the evaluation system with various response quality levels.")
    print("Expected: Good responses approved, poor responses rejected with feedback.")
    
    try:
        # Test simple evaluation
        test_simple_evaluation()
        
        # Test detailed evaluation
        test_detailed_evaluation()
        
        print_separator("TEST COMPLETE")
        print("All tests executed successfully!")
        print("\nNote: Evaluation quality depends on the judge model configuration.")
        print("See docs/EVALUATION.md for more information.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
