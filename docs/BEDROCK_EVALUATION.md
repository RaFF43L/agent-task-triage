# AWS Bedrock Native Evaluation Integration

This document describes the **Bedrock native evaluator** integration that judges agent responses in the `finalize_node` before they are sent to users.

## Overview

The system now includes **two evaluation layers**:

1. **Quality Node Evaluation** (LLM-as-judge) - Evaluates work during the retry loop
2. **Finalize Node Evaluation** (Bedrock native) - **Final judge before user response**

The Bedrock native evaluator provides a last line of defense, catching any quality issues that might have passed the quality node.

## Architecture

```
┌─────────────┐
│ Tool Result │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   quality_node      │ ← LLM-as-judge evaluation (retry loop)
│   (eval_mode)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   finalize_node     │ ← Bedrock native evaluation (final judge)
│   (bedrock_judge)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Final Response    │ → User
└─────────────────────┘
```

## How It Works

### 1. Bedrock Native Evaluator

The [`BedrockEvaluator`](src/ai/evaluation/bedrock_evaluator.py:20) class uses AWS Bedrock's Converse API to evaluate responses:

```python
from ai.evaluation.bedrock_evaluator import bedrock_evaluator

result = bedrock_evaluator.evaluate_response(
    prompt="User request",
    response="Agent response to evaluate",
    category="software",
    evaluation_type="quality"
)

# result = {
#     "approved": bool,
#     "score": float (0-1),
#     "reasoning": str,
#     "dimensions": dict
# }
```

### 2. Integration in Finalize Node

The [`finalize_node`](src/ai/agent.py:137) applies Bedrock evaluation as the final quality check:

```python
def finalize_node(state: TriageState) -> dict:
    work_result = sanitize_customer_message(state.get("work_result", ""))
    approved = state.get("quality_approved", False)
    
    # Initial response
    response = work_result if approved else work_result + NOT_APPROVED_NOTE
    
    # Apply Bedrock native evaluation (final judge)
    if envs.enable_bedrock_judge and approved:
        bedrock_eval_result = bedrock_evaluator.evaluate_response(
            prompt=state["message"],
            response=work_result,
            category=state["category"],
            evaluation_type=envs.bedrock_eval_type,
        )
        
        # If Bedrock judge rejects, add warning
        if not bedrock_eval_result.get("approved", True):
            response += "\n\n⚠️ Quality check warning..."
    
    return {"response": response, "metadata": {"bedrock_evaluation": ...}}
```

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Enable Bedrock native evaluation in finalize_node
ENABLE_BEDROCK_JUDGE=true

# Bedrock judge model ID (uses Converse API)
BEDROCK_JUDGE_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Evaluation type: 'quality', 'accuracy', or 'helpfulness'
BEDROCK_EVAL_TYPE=quality
```

### Evaluation Types

#### 1. Quality Evaluation (default)
Comprehensive multi-dimensional assessment:
- **Relevance**: Does it address the user's request?
- **Accuracy**: Is the information correct?
- **Completeness**: Are all necessary details provided?
- **Helpfulness**: Does it provide actionable next steps?
- **Professionalism**: Is the tone appropriate?

Approves if overall score >= 3.5/5.0

#### 2. Accuracy Evaluation
Focused on factual correctness:
- Checks if response information is factually correct
- Validates appropriateness of the information provided

#### 3. Helpfulness Evaluation
Focused on user problem resolution:
- Checks if the response helps solve the user's problem
- Validates presence of actionable guidance

## Comparison: Quality Node vs. Finalize Node

| Aspect | Quality Node | Finalize Node (Bedrock Judge) |
|--------|-------------|------------------------------|
| **Purpose** | Validate work during retry loop | Final quality check before user |
| **When** | After each tool execution | Once, before final response |
| **Action on Fail** | Retry with feedback (up to 3 attempts) | Add warning to response |
| **Configuration** | `EVAL_MODE` (simple/detailed) | `ENABLE_BEDROCK_JUDGE` + `BEDROCK_EVAL_TYPE` |
| **Model** | `MODEL_JUDGE` (via LLM-as-judge) | `BEDROCK_JUDGE_MODEL_ID` (Converse API) |
| **Use Case** | Iterative improvement | Safety net / quality gate |

## Response Flow Example

### Scenario: Response passes quality_node but Bedrock judge identifies an issue

```
1. Tool generates response → "We fixed the bug. It should work now."

2. quality_node evaluates → APPROVED (simple mode passes)

3. finalize_node applies Bedrock evaluation:
   - Bedrock judge evaluates the response
   - Identifies lack of specific details
   - Returns: approved=false, reasoning="Missing specific actions for user"

4. Final response to user:
   "We fixed the bug. It should work now.
   
   ---
   ⚠️ This response was flagged by the final quality evaluation and will be
   reviewed by our team. Reason: Missing specific actions for user"
```

This gives visibility to users while still providing the information, and flags the case for human review.

## Benefits

1. **Defense in Depth**: Two-layer evaluation catches more quality issues
2. **No User Disruption**: Responses still sent, but flagged if problematic
3. **Quality Metrics**: Bedrock evaluation results stored in metadata for monitoring
4. **Flexible Configuration**: Can enable/disable independently of quality_node
5. **AWS Native**: Leverages Bedrock's built-in evaluation capabilities

## Monitoring and Logging

The system logs Bedrock evaluation results:

```python
logger.info("Applying Bedrock native evaluation in finalize_node...")
logger.warning(f"Bedrock judge rejected response: {reasoning}")
```

Evaluation results are also stored in state metadata:

```python
"metadata": {
    "bedrock_evaluation": {
        "approved": bool,
        "score": float,
        "reasoning": str,
        "dimensions": dict
    }
}
```

This enables:
- Tracking rejection rates
- Identifying quality patterns
- Comparing quality_node vs. Bedrock judge decisions
- Building quality dashboards

## Best Practices

### 1. When to Enable Bedrock Judge

**Enable when:**
- High-stakes use cases (financial, healthcare, legal)
- New agent deployments (extra safety)
- Quality issues have been observed
- Building quality metrics and dashboards

**Disable when:**
- Ultra-low latency requirements
- Cost optimization is critical
- Quality node is highly tuned and reliable

### 2. Choosing Evaluation Type

- **`quality`** - Best for general use, comprehensive assessment
- **`accuracy`** - Best when factual correctness is critical
- **`helpfulness`** - Best when problem resolution is the priority

### 3. Model Selection

Use the **same or more capable model** than your agent models:

```bash
# Agent uses Sonnet → Judge uses Sonnet or Opus
MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
BEDROCK_JUDGE_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Agent uses Haiku → Judge uses Sonnet
MODEL_FAST=us.anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_JUDGE_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### 4. Handling Evaluation Failures

The system gracefully handles evaluation errors:
```python
try:
    bedrock_eval_result = bedrock_evaluator.evaluate_response(...)
except Exception as e:
    logger.error(f"Bedrock evaluation failed: {e}")
    # Response is still sent to user (fail open)
```

This ensures service continuity even if Bedrock evaluation encounters issues.

## Testing

### Manual Testing

```bash
# 1. Start the API with Bedrock judge enabled
export ENABLE_BEDROCK_JUDGE=true
export BEDROCK_EVAL_TYPE=quality
PYTHONPATH=src uv run uvicorn api:app --port 8000

# 2. Send a test request
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message":"My account is locked"}'

# 3. Check logs for Bedrock evaluation
# Look for: "Applying Bedrock native evaluation in finalize_node..."
```

### Test Cases

Create test cases in [`examples/test_bedrock_evaluation.py`](examples/test_bedrock_evaluation.py):

```python
from ai.evaluation.bedrock_evaluator import bedrock_evaluator

# Test 1: Good response (should pass)
result = bedrock_evaluator.evaluate_response(
    prompt="I need help with my bill",
    response="I identified the problem with your bill and I'm already processing the adjustment...",
    category="financial",
)
assert result["approved"] == True

# Test 2: Poor response (should fail)
result = bedrock_evaluator.evaluate_response(
    prompt="I need help with my bill",
    response="Ok.",  # Too brief, unhelpful
    category="financial",
)
assert result["approved"] == False
```

## Cost Considerations

Each Bedrock evaluation adds:
- 1 additional API call per final response
- ~500-1000 tokens per evaluation (prompt + response)

**Example costs** (Claude Sonnet):
- Input: ~300 tokens @ $0.003/1K = $0.0009
- Output: ~200 tokens @ $0.015/1K = $0.003
- **Total per evaluation: ~$0.004**

For 1000 requests/day:
- Additional cost: ~$4/day
- Monthly: ~$120/month

**Cost optimization:**
- Use quality evaluation type only (faster than detailed)
- Use Haiku for judge model (cheaper)
- Enable only for high-priority categories
- Sample evaluation (e.g., 10% of traffic)

## Troubleshooting

### Issue: Bedrock evaluation always approves/rejects

**Solution**: Review the evaluation prompts in [`bedrock_evaluator.py`](src/ai/evaluation/bedrock_evaluator.py:89). Adjust the approval threshold or prompt instructions.

### Issue: Evaluation is too slow

**Solution**: 
- Use a faster model (Haiku instead of Sonnet)
- Reduce evaluation prompt complexity
- Consider disabling for low-priority requests

### Issue: JSON parsing errors

**Solution**: The evaluator handles non-JSON responses gracefully, but if issues persist:
- Check model response format
- Review prompt structure
- Enable debug logging: `LOG_LEVEL=DEBUG`

### Issue: High rejection rate

**Solution**:
- Review rejection reasons in logs
- Adjust evaluation type or prompts
- Consider if quality_node needs tuning
- Check if threshold is too strict

## Future Enhancements

Potential improvements:
1. **Conditional evaluation** - Only evaluate certain categories
2. **Sampling** - Evaluate X% of responses for cost savings
3. **A/B testing** - Compare evaluation strategies
4. **Human feedback loop** - Collect user feedback on flagged responses
5. **Evaluation caching** - Cache results for similar requests
6. **Custom metrics** - Add domain-specific evaluation criteria

## Additional Resources

- [AWS Bedrock Evaluation Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation.html)
- [Bedrock Converse API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [LLM-as-Judge Evaluation Guide](docs/EVALUATION.md)

## Summary

The Bedrock native evaluator provides a robust final quality check before responses reach users. It complements the quality_node evaluation by adding a safety net that catches edge cases and maintains high quality standards. The system is flexible, well-monitored, and designed to fail gracefully if evaluation encounters issues.
