import ast
import re
from datetime import datetime
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_with_retry,
    sanitize_input,
    extract_json_from_markdown_fence,
    APIError,
    RateLimitError,
    ModelNotAvailableError
)



def normalize_numeric(value):
    """Convert numeric strings to appropriate numeric types"""
    if isinstance(value, (int, float)):
        return value
    
    if isinstance(value, str):
        value = value.strip()
        # Try integer conversion
        if value.isdigit():
            return int(value)
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
    
    return value

def normalize_whitespace(value):
    """Normalize whitespace in strings"""
    if not isinstance(value, str):
        return value
    
    # Remove extra spaces around commas: "Rome, Italy" -> "Rome,Italy"
    result = re.sub(r'\s*,\s*', ',', value)
    # Normalize multiple spaces to single space
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def normalize_location(location_str):
    """Normalize location strings to handle country code variations"""
    if not isinstance(location_str, str) or ',' not in location_str:
        return location_str
    
    # Common country mappings
    country_mappings = {
        'FR': 'France',
        'US': 'United States', 'USA': 'United States',
        'UK': 'United Kingdom', 'GB': 'United Kingdom',
        'DE': 'Germany', 'Deutschland': 'Germany',
        'IT': 'Italy', 'Italia': 'Italy',
        'ES': 'Spain', 'España': 'Spain',
        'JP': 'Japan',
        'CN': 'China',
        'AU': 'Australia',
        'CA': 'Canada',
        'IN': 'India',
        'BR': 'Brazil',
        'MX': 'Mexico',
        'RU': 'Russia',
        'KR': 'South Korea',
        'NL': 'Netherlands',
        'SE': 'Sweden',
        'NO': 'Norway',
        'DK': 'Denmark',
        'FI': 'Finland',
        'CH': 'Switzerland',
        'AT': 'Austria',
        'BE': 'Belgium',
        'PT': 'Portugal',
        'GR': 'Greece',
        'TR': 'Turkey',
        'PL': 'Poland',
        'CZ': 'Czech Republic',
        'HU': 'Hungary',
        'RO': 'Romania',
        'BG': 'Bulgaria',
        'HR': 'Croatia',
        'SI': 'Slovenia',
        'SK': 'Slovakia',
        'LT': 'Lithuania',
        'LV': 'Latvia',
        'EE': 'Estonia'
    }
    
    # Split by comma and process parts
    parts = [part.strip() for part in location_str.split(',')]
    normalized_parts = []
    
    for part in parts:
        # Check if this part is a country code that should be expanded
        if part.upper() in country_mappings:
            normalized_parts.append(country_mappings[part.upper()])
        else:
            # Check if this part is a full country name that has a common code
            part_lower = part.lower()
            for code, full_name in country_mappings.items():
                if full_name.lower() == part_lower:
                    normalized_parts.append(full_name)
                    break
            else:
                normalized_parts.append(part)
    
    return ','.join(normalized_parts)

def normalize_date(value):
    """Normalize date strings to YYYY-MM-DD format"""
    if not isinstance(value, str):
        return value
    
    # Check if it matches date pattern
    if re.match(r'\d{4}-\d{1,2}-\d{1,2}', value):
        try:
            # Parse and reformat to ensure consistency
            dt = datetime.strptime(value, '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except:
            pass
    
    return value

def normalize_boolean(value):
    """Normalize boolean-like strings"""
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ['true', '1', 'yes', 'on']:
            return True
        elif value_lower in ['false', '0', 'no', 'off']:
            return False
    
    return value

def normalize_value(value):
    """Enhanced value normalization using specialized normalization functions"""
    if value is None:
        return None
    
    # Convert to string first for initial processing
    str_val = str(value).strip()
    
    # Try ast.literal_eval first for safe evaluation
    try:
        result = ast.literal_eval(str_val)
    except:
        result = str_val
    
    # Apply normalization functions in sequence
    result = normalize_whitespace(result)  # Clean up whitespace first
    result = normalize_location(result)    # Handle location codes
    result = normalize_date(result)        # Handle date formats
    result = normalize_boolean(result)     # Handle boolean values
    result = normalize_numeric(result)     # Convert numeric strings last
    
    return result

def is_empty(value):
    return value is None or (isinstance(value, str) and value.strip() == '')

def normalize_param_list(param_list):
    """Normalize a list of {'name': ..., 'value': ...} dicts for comparison."""
    normalized = []
    for p in param_list:
        if is_empty(p.get('value')):  # Skip parameters with empty values
            continue
        name = p['name'].strip()
        value = normalize_value(p['value'])
        normalized.append((name, value))
    # Sort by name so order doesn't affect comparison
    return sorted(normalized, key=lambda x: x[0])


def compare_tool_parameters(pred, gt, debug=False):
    # Add default empty lists to handle missing keys gracefully
    pred_req = pred.get("required_parameters") or pred.get("required parameters") or []
    gt_req = gt.get("required_parameters") or gt.get("required parameters") or []
    
    pred_opt = pred.get("optional_parameters") or pred.get("optional parameters") or []
    gt_opt = gt.get("optional_parameters") or gt.get("optional parameters") or []
    
    pred_required = normalize_param_list(pred_req)
    gt_required = normalize_param_list(gt_req)
    pred_optional = normalize_param_list(pred_opt)
    gt_optional = normalize_param_list(gt_opt)
    
    required_match = pred_required == gt_required
    optional_match = pred_optional == gt_optional
    
    if debug and not (required_match and optional_match):
        print(f"Required mismatch: {pred_required} vs {gt_required}")
        print(f"Optional mismatch: {pred_optional} vs {gt_optional}")
    
    return required_match and optional_match

def exact_match_tools(gt_tool_list, pred_tool_list, order=False):
    """
    Compare two tool lists for exact matching.
    
    Args:
        gt_tool_list: Ground truth tool list
        pred_tool_list: Predicted tool list  
        order: If True, consider tool order; if False, ignore order
        
    Returns:
        bool: True if tools match exactly according to order parameter
    """
    gt_tool_names = [item['tool name'] for item in gt_tool_list]
    pred_tool_names = [item['tool name'] for item in pred_tool_list]
    
    if order:
        # Consider order: lists must be identical
        return gt_tool_names == pred_tool_names
    else:
        # Ignore order: sets must be identical
        return set(gt_tool_names) == set(pred_tool_names)

def inclusion_tools(gt_tool_list, pred_tool_list):
    """
    Calculate the inclusion ratio of ground truth tools in predicted tools.
    
    This function measures how many of the ground truth tools are present 
    in the predicted tool list, regardless of order or parameters.
    
    Args:
        gt_tool_list: Ground truth tool list (reference)
        pred_tool_list: Predicted tool list to evaluate
        
    Returns:
        float: Ratio of included tools (0.0 to 1.0)
               - 1.0 means all GT tools are found in predictions
               - 0.0 means no GT tools are found in predictions
               - Returns 0.0 if GT list is empty (to avoid division by zero)
    
    Example:
        gt_tools = [{"tool name": "A"}, {"tool name": "B"}, {"tool name": "C"}]
        pred_tools = [{"tool name": "A"}, {"tool name": "C"}, {"tool name": "D"}]
        inclusion_tools(gt_tools, pred_tools)  # Returns 0.67 (2 out of 3 tools found)
    """
    gt_tool_name = [item['tool name'] for item in gt_tool_list]
    pred_tool_name = [item['tool name'] for item in pred_tool_list]
    return len(set(gt_tool_name) & set(pred_tool_name))/max(1,len(gt_tool_name))

def retrieval_rate(gt_tool_list, retrieved_tool_list):
    """
    Calculate the retrieval rate - fraction of ground truth tools found in retrieved tools.
    
    This function measures the recall/coverage of ground truth tools in the retrieved
    tool list. It's similar to inclusion_tools but focuses on retrieval effectiveness.
    
    Args:
        gt_tool_list: Ground truth tool list (reference)
        retrieved_tool_list: Retrieved tool list to evaluate
        
    Returns:
        float: Retrieval rate (0.0 to 1.0)
               - 1.0 means all GT tools are successfully retrieved
               - 0.0 means no GT tools are retrieved
               - Returns 0.0 if GT list is empty (to avoid division by zero)
    
    Example:
        gt_tools = [{"tool name": "Search"}, {"tool name": "Filter"}, {"tool name": "Sort"}]
        retrieved_tools = [{"tool name": "Search"}, {"tool name": "Filter"}]
        retrieval_rate(gt_tools, retrieved_tools)  # Returns 0.67 (2 out of 3 GT tools retrieved)
    """
    gt_tool_names = [item['tool name'] for item in gt_tool_list]
    retrieved_tool_names = [item['tool name'] for item in retrieved_tool_list]
    
    # Find intersection of GT tools and retrieved tools
    retrieved_gt_tools = set(gt_tool_names) & set(retrieved_tool_names)
    
    # Calculate fraction of GT tools that were retrieved
    return len(retrieved_gt_tools) / max(1, len(gt_tool_names))

def tool_traj_usage(gt_tool_list, pred_tool_list):
    """
    Evaluate the correctness of tool usage by comparing parameters between 
    ground truth and predicted tool lists.
    
    This function finds tools that exist in both lists and compares their 
    parameter configurations to assess how correctly the tools are being used.
    
    Args:
        gt_tool_list: Ground truth tool list with correct parameters
        pred_tool_list: Predicted tool list to evaluate
        
    Returns:
        list: List of parameter comparison results for each common tool
              Each result indicates how well the predicted tool parameters 
              match the ground truth parameters
              
    Process:
        1. Find tools that exist in both GT and predicted lists
        2. For each common tool, compare parameter configurations
        3. Return detailed comparison results for analysis
        
    Example:
        gt_tools = [{"tool name": "Search", "required parameters": [{"name": "query", "value": "test"}]}]
        pred_tools = [{"tool name": "Search", "required parameters": [{"name": "query", "value": "test"}]}]
        tool_traj_usage(gt_tools, pred_tools)  # Returns [True] indicating perfect parameter match
    """
    tool_usage = []
    gt_tool_name = [item['tool name'] for item in gt_tool_list]
    pred_tool_name = [item['tool name'] for item in pred_tool_list]
    included = list(set(gt_tool_name) & set(pred_tool_name))
    for item in included:
        gt_target = next((p for p in gt_tool_list if p["tool name"] == item), None)
        pred_target = next((p for p in pred_tool_list if p["tool name"] == item), None)
        tool_usage.append(compare_tool_parameters(pred_target, gt_target))
    return tool_usage


#  following metrics require execution first
def tool_traj_win(model, query, gt_tool_list, pred_tool_list):
    """
    Use LLM to determine which trajectory (ground truth vs predicted) is better 
    for solving a given query.
    
    This function leverages the LLM's reasoning capabilities to evaluate two tool 
    trajectories and determine which one would be more effective for the given query.
    
    Args:
        model: LLM model identifier for evaluation
        query: The user query that needs to be solved
        gt_tool_list: Ground truth tool trajectory (reference)
        pred_tool_list: Predicted tool trajectory to compare
        
    Returns:
        dict: Evaluation result with format:
              {"judge": "gt" or "pred", "reason": "brief explanation"}
              
    Evaluation Criteria (Weighted):
        - TOOL SELECTION (40%): Relevance, completeness, efficiency
        - PARAMETER ACCURACY (35%): Correctness, completeness of parameters  
        - OVERALL EFFECTIVENESS (25%): Query satisfaction, practical utility
        
    Process:
        1. Construct detailed evaluation prompt with criteria
        2. Send to LLM for analysis
        3. Parse response to determine winner
        4. Fallback to GT if parsing fails
        
    Example:
        result = tool_traj_win("claude_v37", "Find hotels in Paris", gt_tools, pred_tools)
        # Returns: {"judge": "gt", "reason": "GT has more relevant tools for hotel search"}
    """
    try:
        prompt = f"""Compare two tool trajectories for solving this query:

Query: {query}

Ground Truth Trajectory: {json.dumps(gt_tool_list, indent=2)}

Predicted Trajectory: {json.dumps(pred_tool_list, indent=2)}

Which trajectory is better for solving the query? Evaluate using these detailed criteria:

TOOL SELECTION (40% weight):
- Relevance: Are the selected tools appropriate for the query?
- Completeness: Are all necessary tools included?
- Efficiency: Are there unnecessary or redundant tools?

PARAMETER ACCURACY (35% weight):
- Correctness: Are parameter values accurate and properly formatted?
- Completeness: Are all required parameters provided?

OVERALL EFFECTIVENESS (25% weight):
- Query satisfaction: Does the trajectory fully address the query?
- Practical utility: Would this trajectory produce useful results?

Respond in JSON format: 
```json
{{"judge": "gt" or "pred", "reason": "brief explanation"}}
```
"""
        
        response = generate_content_with_retry(
            model=model,
            prompt=prompt,
            max_tokens=300
        )
        
        result_json = extract_json_from_markdown_fence(response)
        if result_json:
            return result_json
        
        try:
            return json.loads(response)
        except:
            return {"judge": "gt", "reason": "Failed to parse LLM response"}
            
    except Exception as e:
        return {"judge": "gt", "reason": f"Error in evaluation: {str(e)}"}

def tool_traj_judge(model, query, pred_tool_list):
    """
    Use LLM to evaluate how well a predicted tool trajectory solves a given query,
    providing a score from 0-10 with detailed reasoning.
    
    This function assesses the quality and effectiveness of a single tool trajectory
    by analyzing its relevance, completeness, and practical utility for the query.
    
    Args:
        model: LLM model identifier for evaluation
        query: The user query that needs to be solved
        pred_tool_list: Predicted tool trajectory to evaluate
        
    Returns:
        dict: Evaluation result with format:
              {"score": 0-10, "reason": "detailed explanation"}
              
    Scoring System (0-10 scale):
        TOOL RELEVANCE (0-3 points):
        - 0: Wrong tools selected, completely irrelevant
        - 1: Some relevant tools but major gaps or inappropriate choices  
        - 2: Mostly appropriate tools with minor issues
        - 3: All tools are highly relevant and well-chosen
        
        PARAMETER ACCURACY (0-4 points):
        - 0: Missing or completely wrong parameters
        - 1-2: Some correct parameters but significant gaps
        - 3: Most parameters correct with minor issues
        - 4: All parameters accurate and properly configured
        
        OVERALL EFFECTIVENESS (0-3 points):
        - 0: Trajectory cannot solve the query
        - 1: Partial solution with major limitations
        - 2: Good solution with minor issues
        - 3: Complete, effective solution
        
    Process:
        1. Construct detailed scoring prompt with criteria
        2. Send to LLM for comprehensive evaluation
        3. Parse response to extract score and reasoning
        4. Return structured evaluation result
        
    Example:
        result = tool_traj_judge("claude_v37", "Find hotels in Paris", pred_tools)
        # Returns: {"score": 8, "reason": "Excellent tool selection with minor parameter gaps"}
    """
    try:
        prompt = f"""Evaluate how well this tool trajectory solves the given query:

Query: {query}

Tool Trajectory: {json.dumps(pred_tool_list, indent=2)}

Rate how effectively this trajectory solves the query on a scale of 0-10 using these detailed criteria:

TOOL RELEVANCE (0-3 points):
- 0: Wrong tools selected, completely irrelevant
- 1: Some relevant tools but major gaps or inappropriate choices
- 2: Mostly appropriate tools with minor issues
- 3: All tools are highly relevant and well-chosen

PARAMETER QUALITY (0-4 points):
- 0: Parameters are incorrect, missing, or malformed
- 1: Many parameter errors, some correct values
- 2: Parameters mostly correct with some minor errors
- 3: Parameters are accurate with very minor issues
- 4: All parameters are correct and well-formatted

QUERY COMPLETENESS (0-3 points):
- 0: Does not address the query at all
- 1: Addresses only small parts of the query
- 2: Addresses most aspects but misses some key elements
- 3: Fully addresses all aspects of the query

Total Score: Tool Relevance + Parameter Quality + Query Completeness (0-10)

Respond in JSON format: {{"judge": score (0-10), "reason": "brief explanation"}}"""
        
        response = generate_content_with_retry(
            model=model,
            prompt=prompt,
            max_tokens=300
        )
        
        result_json = extract_json_from_markdown_fence(response)
        if result_json:
            return result_json
        
        try:
            return json.loads(response)
        except:
            return {"judge": 0, "reason": "Failed to parse LLM response"}
            
    except Exception as e:
        return {"judge": 0, "reason": f"Error in evaluation: {str(e)}"}

def solution_match(model, query, gt_answer, pred_answer):
    """
    LLM determine if the pred_answer is the same with gt_answer given the query; provide brief reasons.
    """
    try:
        prompt = f"""Compare the predicted answer with the ground truth answer for the given query:

Query: {query}

Ground Truth Answer: {gt_answer}

Predicted Answer: {pred_answer}

Determine if the predicted answer is equivalent to the ground truth answer. Consider:
- Semantic meaning rather than exact wording
- Factual accuracy and completeness
- Whether both answers solve the query equally well

Respond in JSON format: {{"judge": 1 (if same/equivalent) or 0 (if different), "reason": "brief explanation"}}"""
        
        response = generate_content_with_retry(
            model=model,
            prompt=prompt,
            max_tokens=300
        )
        
        result_json = extract_json_from_markdown_fence(response)
        if result_json:
            return result_json
        
        try:
            return json.loads(response)
        except:
            return {"judge": 0, "reason": "Failed to parse LLM response"}
            
    except Exception as e:
        return {"judge": 0, "reason": f"Error in evaluation: {str(e)}"}