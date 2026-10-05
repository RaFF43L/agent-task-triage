"""AWS Bedrock native evaluation integration using Bedrock Evaluators.

This module integrates AWS Bedrock's built-in model evaluation capabilities
to judge agent responses before they are sent to users.
"""

import json
from typing import Any, Literal
from config.aws import aws_client_manager
from config import envs
from config.logging import get_logger

logger = get_logger(__name__)


class BedrockEvaluator:
    """Wrapper for AWS Bedrock native evaluation capabilities.
    
    Uses Bedrock's built-in evaluation models to judge response quality
    across multiple dimensions (accuracy, helpfulness, harmfulness, etc.).
    """

    def __init__(self):
        self.client = aws_client_manager.get_client("bedrock-runtime")
        self.judge_model_id = getattr(envs, "bedrock_judge_model_id", envs.model)

    def evaluate_response(
        self,
        prompt: str,
        response: str,
        category: Literal["financial", "software"],
        evaluation_type: str = "quality",
    ) -> dict[str, Any]:
        """Evaluate a response using Bedrock's native evaluation.
        
        Args:
            prompt: Original user request
            response: Agent's response to evaluate
            category: Category of the request
            evaluation_type: Type of evaluation (quality, accuracy, helpfulness)
            
        Returns:
            Dict with evaluation results including:
            - approved: bool
            - score: float (0-1)
            - reasoning: str
            - dimensions: dict with individual scores
        """
        logger.info(f"Evaluating response with Bedrock evaluator (type={evaluation_type})")
        
        # Build evaluation prompt using Bedrock's evaluation format
        evaluation_prompt = self._build_evaluation_prompt(
            prompt=prompt,
            response=response,
            category=category,
            evaluation_type=evaluation_type,
        )
        
        try:
            # Use Bedrock Converse API to get structured evaluation
            response_obj = self.client.converse(
                modelId=self.judge_model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": evaluation_prompt}],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 1000,
                    "temperature": 0,  # Deterministic for consistency
                },
            )
            
            # Extract evaluation result
            output_text = response_obj["output"]["message"]["content"][0]["text"]
            
            # Parse structured evaluation response
            evaluation_result = self._parse_evaluation_response(output_text)
            
            logger.info(
                f"Bedrock evaluation complete: approved={evaluation_result['approved']}, "
                f"score={evaluation_result['score']:.2f}"
            )
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error during Bedrock evaluation: {e}")
            # Fallback: approve by default if evaluation fails
            return {
                "approved": True,
                "score": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}. Response approved by default.",
                "dimensions": {},
            }

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response: str,
        category: str,
        evaluation_type: str,
    ) -> str:
        """Build evaluation prompt in Bedrock's preferred format."""
        
        system_instructions = {
            "quality": """You are an expert quality evaluator for customer service responses.
Evaluate the response on these dimensions (score 1-5 for each):
- Relevance: Does it address the user's request?
- Accuracy: Is the information correct?
- Completeness: Are all necessary details provided?
- Helpfulness: Does it provide actionable next steps?
- Professionalism: Is the tone appropriate?

Return your evaluation in JSON format with:
{
  "approved": true/false,
  "overall_score": float (1-5),
  "dimensions": {
    "relevance": int (1-5),
    "accuracy": int (1-5),
    "completeness": int (1-5),
    "helpfulness": int (1-5),
    "professionalism": int (1-5)
  },
  "reasoning": "brief explanation"
}

Approve (true) if overall_score >= 3.5, otherwise reject (false).""",
            
            "accuracy": """You are an accuracy evaluator. Check if the response information is factually correct and appropriate.
Return JSON: {"approved": bool, "score": float (0-1), "reasoning": "explanation"}""",
            
            "helpfulness": """You are a helpfulness evaluator. Check if the response helps the user solve their problem.
Return JSON: {"approved": bool, "score": float (0-1), "reasoning": "explanation"}""",
        }
        
        system_prompt = system_instructions.get(evaluation_type, system_instructions["quality"])
        
        return f"""{system_prompt}

## User Request (Category: {category})
{prompt}

## Agent Response to Evaluate
{response}

## Your Evaluation (JSON format only):"""

    def _parse_evaluation_response(self, response_text: str) -> dict[str, Any]:
        """Parse evaluation response from Bedrock."""
        try:
            # Try to extract JSON from the response
            # Handle cases where the model includes markdown code blocks
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)
                
                # Normalize the response structure
                result = {
                    "approved": evaluation.get("approved", True),
                    "score": float(evaluation.get("overall_score", evaluation.get("score", 0))) / 5.0,  # Normalize to 0-1
                    "reasoning": evaluation.get("reasoning", ""),
                    "dimensions": evaluation.get("dimensions", {}),
                }
                
                return result
            else:
                # If no JSON found, parse as text
                logger.warning("No JSON found in evaluation response, using text parsing")
                approved = "approved" in response_text.lower() or "true" in response_text.lower()
                return {
                    "approved": approved,
                    "score": 1.0 if approved else 0.0,
                    "reasoning": response_text,
                    "dimensions": {},
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            # Default to approved if parsing fails
            return {
                "approved": True,
                "score": 0.0,
                "reasoning": f"Failed to parse evaluation: {response_text[:200]}",
                "dimensions": {},
            }


# Singleton instance
bedrock_evaluator = BedrockEvaluator()
