import os
import json
import re
import argparse
import time
import random
import sys
import logging
from copy import deepcopy
import concurrent.futures

# Setup lightweight logging for performance
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import basic model functions from centralized providers
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_vllm, 
    generate_content_gemini,
    generate_content_bedrock, 
    generate_content_openai,
    vllm_api_meta,
    bedrock_meta, 
    gemini_models,
    openai_models
)

parser = argparse.ArgumentParser('evaluate tool usage')
parser.add_argument('-model', type=str, default='claude_v37', help='model name', choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
parser.add_argument('-traj_file', type=str, default='simple_traj_parallel_consis_gemini-2.5-pro_v2', help='trajectory file')
parser.add_argument('-save_dir', type=str, default='/home/ec2-user/mountS3/newToolData/simple_query', help='log directory')
parser.add_argument('-chk_dir', type=str, default='./chk/traj_to_query', help='checkpoint directory')
parser.add_argument('-enable_multi_query', action='store_true', help='enable multi-query generation with LLM judge')
parser.add_argument('-num_candidates', type=int, default=3, help='number of query candidates to generate')
parser.add_argument('-judge_model', type=str, default=None, help='model for judging queries (defaults to same as generation model)')
# parser.add_argument('-save_name', type=str, default='simple_traj_parallel_claude_v37', help='save name')
args = parser.parse_args()

# Robust features preserved from original file

# generate content function
def generate_content(model, prompt, temperature=None):
    if model in vllm_api_meta:
        temp = temperature if temperature is not None else 0.7
        return generate_content_vllm(model, prompt, max_tokens=8000, temperature=temp)
    elif model in gemini_models:
        return generate_content_gemini(model, prompt)
    elif model in bedrock_meta:
        temp = temperature if temperature is not None else 0.3
        return generate_content_bedrock(model, prompt, max_tokens=8000, temperature=temp)
    elif model in openai_models:
        return generate_content_openai(model, prompt)
    else:
        raise ValueError(f"Unknown model: {model}")

# Robust JSON extraction with multiple fallback patterns
def sanitize_json_string(json_str):
    """Clean up common JSON issues"""
    # Remove control characters that cause parsing errors
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
    # Fix common quote issues
    json_str = json_str.replace('""', '"')
    # Remove trailing commas
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    return json_str.strip()

def extract_json_from_markdown_fence(text: str, max_retries=3):
    """
    Robust JSON extraction with multiple fallback patterns.
    
    Args:
        text (str): The input string containing JSON content.
        max_retries (int): Number of parsing attempts with different patterns
        
    Returns:
        dict: Parsed JSON data
        
    Raises:
        ValueError: If no valid JSON can be extracted after all attempts
    """
    if not text or not text.strip():
        raise ValueError("Empty or None input text")
    
    # Multiple patterns to try, in order of preference
    patterns = [
        r"```json\s*(\{.*?\})\s*```",           # Standard markdown fence
        r"```\s*(\{.*?\})\s*```",               # Fence without json label
        r"(\{[^{}]*\"query\"[^{}]*\})",         # Simple query JSON
        r"(\{.*?\})",                           # Any JSON-like structure
    ]
    
    for attempt in range(max_retries):
        for i, pattern in enumerate(patterns):
            try:
                matches = re.findall(pattern, text, re.DOTALL)
                if not matches:
                    continue
                
                # Try each match found by this pattern
                for match in matches:
                    try:
                        # Clean the JSON string
                        cleaned_json = sanitize_json_string(match)
                        if not cleaned_json:
                            continue
                            
                        # Attempt to parse
                        result = json.loads(cleaned_json)
                        
                        # Validate that it's a reasonable result
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                            
                    except json.JSONDecodeError as e:
                        if attempt == max_retries - 1 and i == len(patterns) - 1:
                            logger.warning(f"JSON parse error on attempt {attempt+1}, pattern {i+1}: {e}")
                        continue
                
            except Exception as e:
                if attempt == max_retries - 1 and i == len(patterns) - 1:
                    logger.warning(f"Pattern matching error: {e}")
                continue
    
    # If all else fails, try to extract just the query value
    query_patterns = [
        r'"query"\s*:\s*"([^"]*)"',
        r"'query'\s*:\s*'([^']*)'",
        r'"query"\s*:\s*([^,}\]]+)',
    ]
    
    for pattern in query_patterns:
        try:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                query_value = match.group(1).strip().strip('"').strip("'")
                if query_value:
                    return {"query": query_value}
        except Exception:
            continue
    
    # Last resort: return the original text as query if it looks reasonable
    cleaned_text = text.strip()
    if len(cleaned_text) > 10 and len(cleaned_text) < 1000:
        return {"query": cleaned_text}
    
    raise ValueError(f"Could not extract valid JSON after {max_retries} attempts with {len(patterns)} patterns")

def load_checkpoint(chk_file):
    try:
        with open(chk_file, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # no checkpoint file, start from beginning

def save_checkpoint(index, chk_file):
    with open(chk_file, "w") as f:
        f.write(str(index))

def robust_generate_content(model, prompt, temperature=None, max_retries=3, backoff_factor=1.5):
    """
    Generate content with retry logic and exponential backoff.
    
    Args:
        model: Model name
        prompt: Input prompt
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
    
    Returns:
        str: Generated content
    
    Raises:
        Exception: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Add small random delay to avoid rate limiting
            if attempt > 0:
                delay = (backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                print(f"Retry attempt {attempt + 1} after {delay:.1f}s delay...")
                time.sleep(delay)
            
            # Validate inputs
            if not prompt or not prompt.strip():
                raise ValueError("Empty prompt provided")
            
            if len(prompt) > 100000:  # Very long prompts might cause issues
                print(f"WARNING: Very long prompt ({len(prompt)} chars), truncating...")
                prompt = prompt[:100000] + "..."
            
            response = generate_content(model, prompt, temperature)
            
            # Validate response
            if not response or not response.strip():
                raise ValueError("Empty response from model")
            
            if len(response) < 5:  # Suspiciously short response
                raise ValueError(f"Suspiciously short response: '{response}'")
            
            return response
            
        except Exception as e:
            last_error = e
            print(f"WARNING: Generation attempt {attempt + 1} failed: {e}")
            
            # Don't retry on certain types of errors
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                print("WARNING: Rate limit detected, extending delay...")
                time.sleep(min(60, (backoff_factor ** attempt) * 10))  # Longer delay for rate limits
            elif "authentication" in str(e).lower() or "authorization" in str(e).lower():
                print("ERROR: Authentication error, not retrying")
                break
    
    raise Exception(f"Failed to generate content after {max_retries} attempts. Last error: {last_error}")

def robust_extract_query(response_text, context="generation"):
    """
    Robustly extract query from response with detailed error reporting.
    
    Args:
        response_text: Raw response from model
        context: Context for error reporting (e.g., "generation", "judging")
    
    Returns:
        str: Extracted query text
    
    Raises:
        Exception: If query extraction fails
    """
    try:
        if not response_text or not response_text.strip():
            raise ValueError(f"Empty response in {context}")
        
        # Try to extract JSON
        result = extract_json_from_markdown_fence(response_text)
        
        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError(f"Response is not a dictionary: {type(result)}")
        
        if "query" not in result:
            raise ValueError(f"No 'query' field found in response. Available keys: {list(result.keys())}")
        
        query = result["query"]
        if not query or not query.strip():
            raise ValueError("Empty query value found")
        
        # Basic query validation
        if len(query.strip()) < 5:
            raise ValueError(f"Query too short: '{query}'")
        
        if len(query) > 5000:
            print(f"WARNING: Very long query ({len(query)} chars)")
        
        return query.strip()
        
    except Exception as e:
        print(f"ERROR: Query extraction failed in {context}: {e}")
        print(f"DEBUG: Response preview: {response_text[:200]}...")
        raise

# Multi-query judge prompt template
judge_prompt_template = """
You are a query evaluation expert. Your task is to compare multiple generated queries and select the best one that describes a tool-calling trajectory.

<Tool Trajectory>
[tool_list]
</Tool Trajectory>

<Query Requirements>
[query_requirements]
</Query Requirements>

<Query Candidates>
[candidates]
</Query Candidates>

For each query, assess:
- **Trajectory Match (0–5):** Does the query *require* the same tools and call order as the trajectory?
- **Query Type Fit (0–5):** Does the query fulfill the query requirement?
- **Overall Score (0–10):** Based on the above, how well does the query match overall?

Respond in this JSON format:
```json
{
    "evaluations": [
        {
            "query_candidate": "each query in candidates",
            "scores": {
                trajectory_match": ,
                "type_fit": 
            },
            "total_score": ,
            "comments": "Brief explanation of assessment"
        }
    ],
    "best_query": "The selected best query text here",
    "justification": "Explain why this query was chosen as the best based on the specific requirements"
}
```
"""

def generate_multiple_queries_with_judge(model, prompt_template, tool_list, query_requirements, num_candidates=3, judge_model=None):
    """
    Generate multiple query candidates and use LLM judge to select the best one.
    
    Args:
        model: Model for query generation
        prompt_template: Template for query generation
        tool_list: Tool trajectory information
        num_candidates: Number of query candidates to generate
        judge_model: Model for judging (defaults to generation model)
        query_requirements: Specific requirements for this query type
    
    Returns:
        dict: Contains best query, all candidates, and evaluation details
    """
    if judge_model is None:
        judge_model = model
    
    candidates = []
    
    # Generate multiple candidates with higher temperature for diversity
    successful_generations = 0
    max_attempts = num_candidates * 2  # Allow some failed attempts
    
    for i in range(max_attempts):
        if successful_generations >= num_candidates:
            break
            
        try:
            # Use higher temperature for diversity
            temperature = 0.9 if model in bedrock_meta else 1.0
            
            # Silent candidate generation for performance
            
            # Use robust generation
            response = robust_generate_content(model, prompt_template, temperature=temperature)
            
            # Use robust extraction
            query = robust_extract_query(response, f"candidate_{successful_generations + 1}")
            
            candidates.append({
                "id": successful_generations + 1,
                "query": query,
                "temperature": temperature,
                "attempt": i + 1
            })
            successful_generations += 1
            
        except Exception as e:
            print(f"WARNING: Error generating candidate (attempt {i+1}): {e}")
            
            # If we've tried many times and have no candidates, something is seriously wrong
            if i >= 5 and successful_generations == 0:
                print("ERROR: Multiple generation failures, aborting candidate generation")
                break
            continue
    
    if not candidates:
        raise Exception("Failed to generate any query candidates")
    
    # If only one candidate was generated, return it directly
    if len(candidates) == 1:
        return {
            "best_query": candidates[0]["query"],
            "candidates": candidates,
            "evaluation": None,
            "selection_method": "single_candidate"
        }
    
    # Format candidates for judge
    candidates_text = ""
    for candidate in candidates:
        candidates_text += f"Query {candidate['id']}: {candidate['query']}\n\n"
    
    # Use LLM judge to select best query
    try:
        judge_prompt = judge_prompt_template.replace("[tool_list]", str(tool_list)).replace("[query_requirements]", query_requirements).replace("[candidates]", candidates_text.strip())
        
        # Silent judge evaluation for performance
        
        # Use robust generation for judge
        judge_response = robust_generate_content(judge_model, judge_prompt, temperature=0.1)
        
        # Use robust extraction for judge evaluation
        evaluation_data = extract_json_from_markdown_fence(judge_response)
        
        # Extract the best query
        best_query_id = evaluation_data.get("best_query_id", 1)
        best_query = evaluation_data.get("best_query", "")
        
        # If judge didn't provide best_query text, get it from candidates
        if not best_query:
            best_candidate = next((c for c in candidates if c["id"] == best_query_id), candidates[0])
            best_query = best_candidate["query"]
        
        return {
            "best_query": best_query,
            "candidates": candidates,
            "evaluation": evaluation_data,
            "selection_method": "llm_judge"
        }
        
    except Exception as e:
        print(f"WARNING: Error in judge evaluation: {e}")
        # Fallback: return first candidate
        return {
            "best_query": candidates[0]["query"],
            "candidates": candidates,
            "evaluation": None,
            "selection_method": "fallback_first"
        }

def generate_both_queries_parallel(record, args, requirement_simple, requirement_hard, format_tool_prompt):
    """
    Generate simple and hard queries in parallel for 2x speedup.
    
    Args:
        record: Single record with tool list
        args: Command line arguments
        requirement_simple: Simple query requirements
        requirement_hard: Hard query requirements
        format_tool_prompt: Template for formatting prompts
    
    Returns:
        tuple: (simple_result, hard_result, error)
    """
    tool_list_str = str(record['tool list'])
    
    # Prepare prompts
    prompt_query_simple = requirement_simple + format_tool_prompt.replace("[tool list]", tool_list_str)
    prompt_query_hard = requirement_hard + format_tool_prompt.replace("[tool list]", tool_list_str)
    
    def generate_simple_query():
        """Generate simple query in separate thread"""
        try:
            if args.enable_multi_query:
                return generate_multiple_queries_with_judge(
                    args.model, prompt_query_simple, record['tool list'], 
                    requirement_simple, args.num_candidates, 
                    args.judge_model if hasattr(args, 'judge_model') else None
                )
            else:
                response = robust_generate_content(args.model, prompt_query_simple)
                return {"query": robust_extract_query(response, "simple_query")}
        except Exception as e:
            raise Exception(f"Simple query generation failed: {e}")
    
    def generate_hard_query():
        """Generate hard query in separate thread"""
        try:
            if args.enable_multi_query:
                return generate_multiple_queries_with_judge(
                    args.model, prompt_query_hard, record['tool list'], 
                    requirement_hard, args.num_candidates,
                    args.judge_model if hasattr(args, 'judge_model') else None
                )
            else:
                response = robust_generate_content(args.model, prompt_query_hard)
                return {"query": robust_extract_query(response, "hard_query")}
        except Exception as e:
            raise Exception(f"Hard query generation failed: {e}")
    
    # Run both generations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Silent parallel generation for performance
        
        # Submit both tasks
        simple_future = executor.submit(generate_simple_query)
        hard_future = executor.submit(generate_hard_query)
        
        # Wait for both to complete with timeout
        try:
            simple_result = simple_future.result(timeout=180)  # 3 min timeout per query
            hard_result = hard_future.result(timeout=180)
            
            # Silent success for performance
            return simple_result, hard_result, None
            
        except concurrent.futures.TimeoutError:
            # Cancel any remaining tasks
            simple_future.cancel()
            hard_future.cancel()
            return None, None, "Timeout: Query generation took too long"
            
        except Exception as e:
            # Cancel any remaining tasks
            simple_future.cancel()
            hard_future.cancel()
            return None, None, str(e)

#### data loading
# domain list
with open('/home/ec2-user/mountS3/newToolData/selected_category.json', 'r') as f:
    select_cate = json.load(f)
domain_list = list(select_cate.keys())[9:]
num_domain = len(domain_list)

# main
format_tool_prompt = """
Given the used tools, including tool description and detailed parameters:
[tool list]

Please provide the query in the following json format:

```json
{
    "query": "..."
}
```
"""
with open('/home/ec2-user/mountS3/newToolData/query_gen/query_gen_prompt.json', 'r') as f:
    gen_prompt = json.load(f)

for domain in domain_list:
    print(f"=== Generating query for domain: {domain} ===")
    # load checkpoint
    save_file = os.path.join(args.save_dir, f"{domain}/{args.traj_file}.json")
    chk_file = os.path.join(args.chk_dir, f"{args.traj_file}_{args.model}_{domain}.txt")
    chk_index = load_checkpoint(chk_file)
    requirement_simple = gen_prompt['simple']
    requirement_hard = gen_prompt[f'{domain}_hard']
    # if os.path.isfile(log_file):
    with open(save_file, 'r') as file:
        records = json.load(file)
    # evaluate
    for i in range(chk_index, len(records)):
        try:
            print(f"Processing record {i+1}/{len(records)} for domain {domain}")
            
            # Time the query generation for this trajectory
            trajectory_start_time = time.time()
            
            # Generate both simple and hard queries in parallel
            simple_result, hard_result, error = generate_both_queries_parallel(
                records[i], args, requirement_simple, requirement_hard, format_tool_prompt
            )
            
            trajectory_time = time.time() - trajectory_start_time
            
            if error:
                print(f"❌ Error processing record {i}: {error} (took {trajectory_time:.2f}s)")
                continue
            
            # Print timing information
            print(f"✅ Record {i+1} completed in {trajectory_time:.2f} seconds")
            
            # Store results based on generation method
            if args.enable_multi_query:
                # Multi-query results with metadata
                records[i][f"query_simple_{args.model}"] = simple_result['best_query']
                records[i][f"query_simple_{args.model}_candidates"] = simple_result.get('candidates', [])
                records[i][f"query_simple_{args.model}_evaluation"] = simple_result.get('evaluation')
                records[i][f"query_simple_{args.model}_selection_method"] = simple_result.get('selection_method')
                
                records[i][f"query_hard_{args.model}"] = hard_result['best_query']
                records[i][f"query_hard_{args.model}_candidates"] = hard_result.get('candidates', [])
                records[i][f"query_hard_{args.model}_evaluation"] = hard_result.get('evaluation')
                records[i][f"query_hard_{args.model}_selection_method"] = hard_result.get('selection_method')
            else:
                # Single query results
                records[i][f"query_simple_{args.model}"] = simple_result['query']
                records[i][f"query_hard_{args.model}"] = hard_result['query']
            
            # Store timing metadata
            records[i][f"query_generation_time_{args.model}"] = trajectory_time
            
            # save checkpoint
            save_checkpoint(i+1, chk_file)
            
            # Save results periodically (every 10 records) to avoid data loss
            if (i + 1) % 10 == 0:
                with open(save_file, 'w') as f:
                    json.dump(records, f, indent=4)
                print(f"Saved checkpoint at record {i+1}")
                
        except Exception as e:
            print(f"ERROR: Error processing record {i}: {e}")
            continue
    
    # Final save for this domain
    with open(save_file, 'w') as f:
        json.dump(records, f, indent=4)
        # exit(0)