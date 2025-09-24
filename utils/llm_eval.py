import ast
import re
from datetime import datetime
import json
import os
import sys

# Import basic model functions from centralized providers
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_with_retry,
    sanitize_input,
    extract_json_from_markdown_fence,
    APIError,
    RateLimitError,
    ModelNotAvailableError,
)

def answer_match(model, query, gt_answer, pred_answer):
    """
    Use the model to judge whether pred_answer matches gt_answer for the given query.

    Returns:
        dict: {"match": True/False/None, "reason": str}
    """
    # Assume inputs are already strings
    gt = gt_answer
    pred = pred_answer

    # Trivial case
    if gt.strip() == "" and pred.strip() == "":
        return {"match": True, "reason": "Both answers are empty"}

    # Build evaluation prompt
    prompt = f"""You are an expert evaluator. Given a user query, a ground-truth answer, and a candidate answer,
judge whether the candidate matches the ground-truth in meaning (allow paraphrase and different phrasing).
Focus on semantic equivalence, factual consistency, and coverage of key points required by the query.

Query:
{query}

Ground-Truth Answer (GT):
{gt}

Candidate Answer (Pred):
{pred}

Respond with STRICT JSON only:
```json
{{
  "match": true/false,
  "reason": "Brief justification"
}}
```
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = generate_content_with_retry(model=model, prompt=prompt)
            if not response or not isinstance(response, str):
                raise ValueError("Empty or invalid response from model")

            parsed = extract_json_from_markdown_fence(response)
            if not isinstance(parsed, dict) or "match" not in parsed:
                raise ValueError("Missing 'match' field in parsed JSON")

            match_val = parsed.get("match")
            reason_val = parsed.get("reason", "No reason provided")

            if isinstance(match_val, bool):
                return {"match": match_val, "reason": reason_val}
            if isinstance(match_val, str):
                lowered = match_val.strip().lower()
                if lowered in ("true", "yes", "y", "1"):
                    return {"match": True, "reason": reason_val}
                if lowered in ("false", "no", "n", "0"):
                    return {"match": False, "reason": reason_val}
                raise ValueError(f"Unrecognized match string: {match_val}")
            if isinstance(match_val, (int, float)):
                return {"match": bool(match_val), "reason": reason_val}

            raise ValueError("'match' has unsupported type")
        except Exception as e:
            if attempt == max_retries - 1:
                return {"match": None, "reason": f"Failed to obtain valid model judgment after {max_retries} attempts"}
            continue

def traj_win(model, query, gt_tool_list, pred_tool_list, max_retries=3):
    """
    Compare two tool-using trajectories and determine which is better.
    
    Args:
        model: Model name for LLM evaluation
        query: The original query/task
        gt_tool_list: Ground truth tool list (list of dicts)
        pred_tool_list: Predicted tool list (list of dicts)
        max_retries: Maximum number of retry attempts for LLM calls
        
    Returns:
        dict: Contains "better solution" ("S1"/"S2") and "reason"
    """
    # Input validation
    if not query or not isinstance(query, str):
        return {"better solution": "ERROR", "reason": "Invalid query input"}
    
    if not isinstance(gt_tool_list, list) or not isinstance(pred_tool_list, list):
        return {"better solution": "ERROR", "reason": "Tool lists must be lists"}
    
    if not gt_tool_list and not pred_tool_list:
        return {"better solution": "TIE", "reason": "Both solutions are empty"}
    
    if not gt_tool_list:
        return {"better solution": "S2", "reason": "S1 is empty, S2 has tools"}
    
    if not pred_tool_list:
        return {"better solution": "S1", "reason": "S2 is empty, S1 has tools"}
    
    # Format tool lists more robustly
    def format_tool_list(tool_list, name):
        """Format tool list for display with error handling"""
        try:
            if not tool_list:
                return f"{name}: (Empty tool list)"
            
            formatted_tools = []
            for i, tool in enumerate(tool_list):
                if isinstance(tool, dict):
                    tool_name = tool.get('tool name', f'Tool {i+1}')
                    tool_desc = tool.get('tool description', 'No description')
                    formatted_tools.append(f"- {tool_name}: {tool_desc[:100]}{'...' if len(tool_desc) > 100 else ''}")
                else:
                    formatted_tools.append(f"- {str(tool)[:100]}{'...' if len(str(tool)) > 100 else ''}")
            
            return f"{name}:\n" + "\n".join(formatted_tools)
        except Exception as e:
            return f"{name}: (Error formatting tools: {str(e)})"
    
    # Create robust prompt with proper escaping
    try:
        solution1_text = format_tool_list(gt_tool_list, "S1")
        solution2_text = format_tool_list(pred_tool_list, "S2")
        
        prompt = f"""I have a query:
{query}

I have two tool calling solutions, please help me select the better one.

{solution1_text}

{solution2_text}

Please evaluate both solutions based on:
1. Relevance to the query
2. Completeness of tool coverage
3. Appropriateness of tool selection
4. Logical tool ordering (if applicable)

Select the better solution and provide a clear reason. Respond in the following JSON format:

```json
{{
    "better solution": "S1/S2/TIE",
    "reason": "Detailed explanation of why this solution is better"
}}
```"""
    except Exception as e:
        return {"better solution": "ERROR", "reason": f"Error formatting prompt: {str(e)}"}
    
    # Retry logic for LLM calls
    for attempt in range(max_retries):
        try:
            response = generate_content_with_retry(
                model=model,
                prompt=prompt,
            )
            
            if not response or not isinstance(response, str):
                raise ValueError("Empty or invalid response from model")
            
            # Extract and validate JSON
            result_json = extract_json_from_markdown_fence(response)
            
            if not result_json or not isinstance(result_json, dict):
                raise ValueError("Invalid JSON structure")
            
            # Validate required fields
            if "better solution" not in result_json:
                raise ValueError("Missing 'better solution' field")
            
            better_solution = result_json["better solution"]
            if better_solution not in ["S1", "S2", "TIE"]:
                raise ValueError(f"Invalid 'better solution' value: {better_solution}")
            
            # Ensure reason field exists
            if "reason" not in result_json:
                result_json["reason"] = "No reason provided"
            
            return result_json
            
        except (ValueError, KeyError, TypeError) as e:
            if attempt == max_retries - 1:
                return {
                    "better solution": "ERROR", 
                    "reason": f"Failed to parse LLM response after {max_retries} attempts: {str(e)}"
                }
            continue
        except (APIError, RateLimitError, ModelNotAvailableError) as e:
            if attempt == max_retries - 1:
                return {
                    "better solution": "ERROR", 
                    "reason": f"API error after {max_retries} attempts: {str(e)}"
                }
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "better solution": "ERROR", 
                    "reason": f"Unexpected error after {max_retries} attempts: {str(e)}"
                }
            continue
    
    # Fallback (should never reach here)
    return {"better solution": "ERROR", "reason": "Maximum retries exceeded"}


def traj_satisfaction(model, query, pred_tool_list, max_retries=3):
    """
    Score how well a tool trajectory can solve the query.

    Args:
        model: Model name for LLM judging
        query: The original question/query string
        pred_tool_list: Predicted tool list (list of dicts)
        max_retries: Max attempts for LLM call and parsing

    Returns:
        dict with keys:
          - score: float in [0, 10]
          - reason: brief justification
    """
    # Validate inputs
    if not isinstance(query, str) or not query.strip():
        return {"score": 0.0, "reason": "Invalid query input"}
    if not isinstance(pred_tool_list, list):
        pred_tool_list = []

    # Reuse formatter from traj_win
    def format_tool_list(tool_list, name):
        try:
            if not tool_list:
                return f"{name}: (Empty tool list)"
            formatted = []
            for i, tool in enumerate(tool_list):
                if isinstance(tool, dict):
                    tool_name = tool.get('tool name', f'Tool {i+1}')
                    tool_desc = tool.get('tool description', 'No description')
                    formatted.append(f"- {tool_name}: {tool_desc[:120]}{'...' if len(tool_desc) > 120 else ''}")
                else:
                    s = str(tool)
                    formatted.append(f"- {s[:120]}{'...' if len(s) > 120 else ''}")
            return f"{name}:\n" + "\n".join(formatted)
        except Exception as e:
            return f"{name}: (Error formatting tools: {str(e)})"

    tools_text = format_tool_list(pred_tool_list, "Proposed Tool Trajectory")

    prompt = f"""You are an expert judge of tool-using plans.
Given a user query and a proposed tool trajectory, rate how well the plan can solve the query.

Query:
{query}

{tools_text}

Scoring rubric (0-10):
- 0-2: Irrelevant or unusable plan
- 3-4: Partially relevant but unlikely to solve core needs
- 5-6: Reasonably relevant; may solve some parts but missing key steps/tools
- 7-8: Strong plan that likely solves most requirements with minor gaps
- 9-10: Excellent plan; comprehensive, appropriate tools, clear coverage of needs

Return STRICT JSON only:
```json
{{
  "score": 0-10 number,
  "reason": "Brief, specific justification"
}}
```
"""

    for attempt in range(max_retries):
        try:
            response = generate_content_with_retry(model=model, prompt=prompt)
            if not response or not isinstance(response, str):
                raise ValueError("Empty or invalid response from model")

            result = extract_json_from_markdown_fence(response)
            if not isinstance(result, dict):
                raise ValueError("Parsed JSON is not a dict")

            raw_score = result.get("score", None)
            reason = result.get("reason", "No reason provided")

            # Coerce score to float and clamp to [0, 10]
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
            elif isinstance(raw_score, str):
                # extract first number if present
                try:
                    score = float(raw_score.strip())
                except Exception:
                    raise ValueError(f"Non-numeric score: {raw_score}")
            else:
                raise ValueError("Missing or invalid 'score'")

            if score != score:  # NaN check
                raise ValueError("Score is NaN")
            score = max(0.0, min(10.0, score))

            if not isinstance(reason, str):
                reason = str(reason)

            return {"score": score, "reason": reason}

        except (ValueError, KeyError, TypeError) as e:
            if attempt == max_retries - 1:
                return {"score": 0.0, "reason": f"Parsing error after {max_retries} attempts: {str(e)}"}
            continue
        except (APIError, RateLimitError, ModelNotAvailableError) as e:
            if attempt == max_retries - 1:
                return {"score": 0.0, "reason": f"API error after {max_retries} attempts: {str(e)}"}
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                return {"score": 0.0, "reason": f"Unexpected error after {max_retries} attempts: {str(e)}"}
            continue

    return {"score": 0.0, "reason": "Maximum retries exceeded"}