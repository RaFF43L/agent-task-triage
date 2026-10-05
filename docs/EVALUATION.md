# LLM-as-Judge Evaluation System

This repository includes a comprehensive LLM-as-judge evaluation system for assessing agent response quality using Amazon Bedrock models.

## Overview

The evaluation system uses a judge LLM to systematically evaluate agent responses across multiple dimensions, providing detailed scoring and actionable feedback. This follows the LLM-as-judge pattern, which is widely recognized as an effective approach for automated quality assessment in AI systems.

## Features

### Two Evaluation Modes

1. **Simple Mode** (default, fast)
   - Basic pass/fail evaluation
   - Compatible with existing quality node
   - Uses the fast model for efficiency
   - Returns approval status and rejection reason

2. **Detailed Mode** (comprehensive)
   - Multi-dimensional scoring (1-5 scale)
   - Evaluates 6 key criteria:
     - **Relevance**: Does it address the user's request?
     - **Accuracy**: Is the information correct?
     - **Completeness**: Are all necessary details included?
     - **Clarity**: Is it well-structured and easy to understand?
     - **Tone**: Is it professional and empathetic?
     - **Conciseness**: Is the length appropriate?
   - Provides justification for each score
   - Lists critical issues and improvement suggestions
   - Returns overall score and summary

## Architecture

```
src/ai/evaluation/
├── __init__.py          # Public API exports
├── models.py            # Pydantic models for evaluation results
├── prompts.py           # LLM-as-judge prompts
└── judge.py             # Evaluation logic and orchestration
```

### Key Components

- **[`EvaluationCriteria`](src/ai/evaluation/models.py)**: Individual criterion with score and justification
- **[`DetailedEvaluation`](src/ai/evaluation/models.py)**: Multi-dimensional evaluation result
- **[`EvaluationResult`](src/ai/evaluation/models.py)**: Simplified result compatible with existing system
- **[`evaluate_response()`](src/ai/evaluation/judge.py)**: Main evaluation function
- **[`evaluate_response_detailed()`](src/ai/evaluation/judge.py)**: Detailed multi-dimensional evaluation

## Configuration

### Environment Variables

```bash
# Judge model (defaults to main model if not specified)
MODEL_JUDGE=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Evaluation mode: 'simple' or 'detailed'
EVAL_MODE=simple
```

Add these to your `.env` file (see `.env.example` for reference).

### Choosing a Judge Model

The judge model should be:
- **More capable** or equal to the models being evaluated
- **Consistent**: Use temperature=0 for reproducible evaluations
- **Cost-effective**: Consider using the same model as your main model to avoid additional costs

Recommended models:
- `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (default, balanced)
- `us.anthropic.claude-opus-4-5-20250514-v1:0` (most capable, higher cost)

## Usage

### Integration in Quality Node

The evaluation system is automatically integrated into the [`quality_node`](src/ai/agent.py) in the triage graph:

```python
from ai.evaluation import evaluate_response

def quality_node(state: TriageState) -> dict:
    result = evaluate_response(
        message=state["message"],
        work_result=state["work_result"],
        category=state["category"],
        priority=state.get("priority", "medium"),
        attempts=state.get("attempts", 1),
        mode="simple",  # or "detailed"
    )
    return {
        "quality_approved": result.approved,
        "rejection_reason": result.rejection_reason,
        "metadata": {
            "quality_score": result.overall_score,
        },
    }
```

### Direct API Usage

#### Simple Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My bill is wrong, it charged $150 more",
    "work_result": "Hello! I checked your bill and identified a duplicate charge of $150. We are already processing the refund which will be credited within 2 business days. You will receive a confirmation email.",
    "category": "financial",
    "priority": "high",
    "attempts": 1,
    "mode": "simple"
  }'
```

Response:
```json
{
  "approved": true,
  "rejection_reason": "",
  "overall_score": 4.5,
  "detailed_feedback": "Response is clear, addresses the issue, and provides next steps."
}
```

#### Detailed Evaluation

```bash
curl -X POST http://localhost:8000/evaluate/detailed \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I cannot access my account, getting error 500",
    "work_result": "We identified a problem in the authentication server that has already been fixed. Try logging in again. If the error persists, wait 10 minutes and try again.",
    "category": "software",
    "priority": "urgent",
    "attempts": 1
  }'
```

Response includes detailed scores for all criteria:
```json
{
  "relevance": {
    "score": 5,
    "justification": "Directly addresses the login error issue"
  },
  "accuracy": {
    "score": 4,
    "justification": "Correct diagnosis and resolution steps"
  },
  "completeness": {
    "score": 4,
    "justification": "Includes clear next steps and fallback option"
  },
  "clarity": {
    "score": 5,
    "justification": "Clear, concise explanation with actionable steps"
  },
  "tone": {
    "score": 4,
    "justification": "Professional and helpful tone"
  },
  "conciseness": {
    "score": 5,
    "justification": "Appropriately brief without missing information"
  },
  "overall_score": 4.5,
  "approved": true,
  "critical_issues": [],
  "improvement_suggestions": [
    "Consider adding an apology for the inconvenience"
  ],
  "summary": "High-quality response that effectively addresses the technical issue with clear resolution steps."
}
```

### Programmatic Usage

```python
from ai.evaluation import evaluate_response, evaluate_response_detailed

# Simple evaluation
result = evaluate_response(
    message="Customer request...",
    work_result="Agent response...",
    category="software",
    mode="simple"
)

if result.approved:
    print(f"✓ Approved (score: {result.overall_score}/5.0)")
else:
    print(f"✗ Rejected: {result.rejection_reason}")

# Detailed evaluation
detailed = evaluate_response_detailed(
    message="Customer request...",
    work_result="Agent response...",
    category="financial",
    priority="high",
    attempts=1
)

print(f"Overall: {detailed.overall_score}/5.0")
print(f"Relevance: {detailed.relevance.score}/5")
print(f"Accuracy: {detailed.accuracy.score}/5")
# ... other criteria
```

## Evaluation Criteria Explained

### 1. Relevance (1-5)
Does the response directly address what the customer asked? Is it on-topic and focused?

**Example scores:**
- 5: Perfectly addresses the request
- 3: Addresses most aspects but misses some details
- 1: Completely off-topic or doesn't address the request

### 2. Accuracy (1-5)
Is the information provided correct and factual? Are there any errors or misleading statements?

**Example scores:**
- 5: All information is correct
- 3: Mostly correct with minor inaccuracies
- 1: Contains significant errors or misinformation

### 3. Completeness (1-5)
Does the response provide all necessary information? Are there clear next steps?

**Example scores:**
- 5: Complete with all details and clear action items
- 3: Covers basics but missing some important details
- 1: Incomplete, missing critical information

### 4. Clarity (1-5)
Is the response well-structured and easy to understand? Is the language appropriate?

**Example scores:**
- 5: Crystal clear, well-organized, easy to follow
- 3: Understandable but could be clearer
- 1: Confusing, poorly structured, or unclear

### 5. Tone (1-5)
Is the tone professional, courteous, and empathetic? Does it match customer service standards?

**Example scores:**
- 5: Perfectly professional and empathetic
- 3: Professional but could be more empathetic
- 1: Inappropriate, rude, or unprofessional

### 6. Conciseness (1-5)
Is the response appropriately concise? Not too brief, not too verbose?

**Example scores:**
- 5: Perfect length for the situation
- 3: Slightly too long or too brief
- 1: Way too verbose or missing essential information

## Approval Threshold

Responses are approved if:
- **Simple mode**: The judge LLM determines the response adequately solves the request
- **Detailed mode**: `overall_score >= 3.5` (out of 5.0)

You can adjust the threshold in [`judge.py`](src/ai/evaluation/judge.py) if needed.

## Performance Considerations

### Simple Mode
- Uses `model_fast` (Haiku)
- ~1-2 seconds per evaluation
- Suitable for production use in the quality loop
- Lower cost

### Detailed Mode
- Uses `model_judge` (Sonnet or Opus)
- ~2-4 seconds per evaluation
- Best for testing, monitoring, or quality assurance
- Higher cost but more comprehensive feedback

## Monitoring and Logging

The evaluation system logs detailed metrics:

```python
logger.info(f"Quality: approved={result.approved}")
logger.info(f"Overall score: {result.overall_score:.2f}/5.0")
logger.debug(f"Detailed feedback:\n{result.detailed_feedback}")
```

Metadata is also stored in the state for tracking:

```python
"metadata": {
    "quality_approved": result.approved,
    "quality_score": result.overall_score,
    "eval_mode": "simple" or "detailed"
}
```

## Best Practices

1. **Use simple mode in production** for fast, cost-effective evaluation
2. **Use detailed mode for**:
   - Quality assurance testing
   - Model comparison studies
   - Understanding failure patterns
   - Training data generation

3. **Monitor evaluation metrics** to:
   - Track quality trends over time
   - Identify common rejection patterns
   - Tune prompts and improve agent responses

4. **Adjust thresholds** based on your quality requirements:
   - Higher threshold (≥4.0) for critical responses
   - Lower threshold (≥3.0) for less critical use cases

5. **Review rejections regularly** to improve the agent system

## Testing

Run evaluation tests:

```bash
# Test simple evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/evaluation_request.json

# Test detailed evaluation
curl -X POST http://localhost:8000/evaluate/detailed \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/evaluation_request.json
```

## Further Reading

- [LLM-as-Judge Pattern](https://arxiv.org/abs/2306.05685) - Academic research on using LLMs for evaluation
- [Amazon Bedrock Evaluation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation.html) - AWS documentation
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - Best practices for evaluation prompts

## Troubleshooting

### Issue: Evaluation is too strict/lenient

**Solution**: Adjust the prompts in [`prompts.py`](src/ai/evaluation/prompts.py) to provide clearer guidelines or adjust the approval threshold in [`judge.py`](src/ai/evaluation/judge.py).

### Issue: Evaluations are inconsistent

**Solution**: Ensure you're using temperature=0 for the judge model. This is set by default in [`llm.py`](src/ai/llm.py).

### Issue: Evaluation is too slow

**Solution**: 
- Use simple mode instead of detailed mode
- Consider using a faster judge model
- Cache evaluation results for identical requests

### Issue: High AWS costs

**Solution**:
- Use simple mode which leverages the fast model
- Set MODEL_JUDGE to use Haiku instead of Sonnet/Opus
- Implement caching for repeated evaluations

## Contributing

When extending the evaluation system:

1. Add new criteria to [`DetailedEvaluation`](src/ai/evaluation/models.py)
2. Update the prompts in [`prompts.py`](src/ai/evaluation/prompts.py)
3. Document the new criteria in this README
4. Add tests for the new functionality

## License

This evaluation system is part of the 4dev-lang-graph project.
