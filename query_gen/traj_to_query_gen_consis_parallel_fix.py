import os
import json
import argparse
import time
import sys
import logging
from copy import deepcopy
import concurrent.futures

# Setup lightweight logging for performance
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import centralized model functions
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_with_retry,
    extract_json_from_markdown_fence
)

parser = argparse.ArgumentParser('evaluate tool usage')
parser.add_argument('-model', type=str, default='claude_v37', help='model name', choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
parser.add_argument('-traj_file', type=str, default='simple_traj_parallel_consis_gemini-2.5-pro_v2.1', help='trajectory file')
parser.add_argument('-save_dir', type=str, default='/home/ec2-user/mountS3/newToolData/simple_query', help='log directory')
parser.add_argument('-enable_multi_query', action='store_true', help='enable multi-query generation with LLM judge')
parser.add_argument('-num_candidates', type=int, default=3, help='number of query candidates to generate')
parser.add_argument('-judge_model', type=str, default=None, help='model for judging queries (defaults to same as generation model)')
parser.add_argument('-chk_dir', type=str, default='./checkpoints', help='checkpoint directory')
# parser.add_argument('-save_name', type=str, default='simple_traj_parallel_claude_v37', help='save name')
args = parser.parse_args()

# Create necessary directories
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.chk_dir, exist_ok=True)
print(f"Ensured directories exist: {args.save_dir}, {args.chk_dir}")

# Robust features preserved from original file

# Use centralized generate_content function from utils.model_providers

# Use centralized extract_json_from_markdown_fence from utils.model_providers

# Redundant function removed - using centralized version

def load_checkpoint(chk_file):
    try:
        with open(chk_file, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # no checkpoint file, start from beginning

def save_checkpoint(index, chk_file):
    with open(chk_file, "w") as f:
        f.write(str(index))

def robust_generate_content(model, prompt, temperature=None):
    """
    Wrapper for centralized generate_content_with_retry with input validation.
    """
    # Validate inputs
    if not prompt or not prompt.strip():
        raise ValueError("Empty prompt provided")
    
    if len(prompt) > 100000:  # Very long prompts might cause issues
        print(f"WARNING: Very long prompt ({len(prompt)} chars), truncating...")
        prompt = prompt[:100000] + "..."
    
    response = generate_content_with_retry(model, prompt, temperature=temperature)
    
    # Validate response
    if not response or not response.strip():
        raise ValueError("Empty response from model")
    
    if len(response) < 5:  # Suspiciously short response
        raise ValueError(f"Suspiciously short response: '{response}'")
    
    return response

def robust_extract_query(response_text, context="generation"):
    """
    Robustly extract query from response with detailed error reporting.
    Uses centralized extract_json_from_markdown_fence function.
    """
    try:
        if not response_text or not response_text.strip():
            raise ValueError(f"Empty response in {context}")
        
        # Use centralized extraction function with expect_dict=True
        result = extract_json_from_markdown_fence(response_text, expect_dict=True)
        
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
            temperature = 1.0
            
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

def generate_single_query(record, model, requirement, format_tool_prompt, query_type='simple'):
    """Generate a single query (simple or hard) for a trajectory record"""
    try:
        # Extract tool information from the record
        tool_list = record.get('tool list', [])
        
        # Format the prompt
        tool_list_str = json.dumps(tool_list, indent=2)
        final_prompt = requirement.replace('[tool list]', tool_list_str) + format_tool_prompt
        
        # Generate the query
        response = robust_generate_content(model, final_prompt)
        
        # Extract the query from response
        query = robust_extract_query(response)
        result = {"query": query}
        
        if "query" not in result:
            return {"query": f"Generated {query_type} query failed", "error": "No query in result"}
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating {query_type} query: {e}")
        return {"query": f"Generated {query_type} query failed", "error": str(e)}

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
domain_list = list(select_cate.keys())
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
    save_file = os.path.join(args.save_dir, f"{domain}/{args.traj_file}.json")
    requirement_simple = gen_prompt['simple']
    requirement_hard = gen_prompt[f'{domain}_hard']
    
    with open(save_file, 'r') as file:
        records = json.load(file)
    
    # Count records missing query fields
    simple_query_key = f'query_simple_{args.model}_candidates'
    hard_query_key = f'query_hard_{args.model}_candidates'
    
    missing_count = 0
    for record in records:
        if simple_query_key not in record or hard_query_key not in record:
            missing_count += 1
    
    if missing_count == 0:
        print(f"All records in {domain} already have {simple_query_key} and {hard_query_key}")
        continue
    
    print(f"Found {missing_count} records missing query fields in domain {domain}")
    
    # Process records missing query fields
    for i in range(len(records)):
        # Skip records that already have both query fields
        if simple_query_key in records[i] and hard_query_key in records[i]:
            print(f"Skipping record {i+1}/{len(records)} - already has query fields")
            continue
            
        try:
            print(f"Processing record {i+1}/{len(records)} for domain {domain}")
            
            # Time the query generation for this trajectory
            trajectory_start_time = time.time()
            
            # Check which queries are needed
            needs_simple = simple_query_key not in records[i]
            needs_hard = hard_query_key not in records[i]
            
            # Generate only the missing queries
            if needs_simple and needs_hard:
                # Generate both queries in parallel
                simple_result, hard_result, error = generate_both_queries_parallel(
                    records[i], args, requirement_simple, requirement_hard, format_tool_prompt
                )
            elif needs_simple:
                # Generate only simple query
                simple_result = generate_single_query(
                    records[i], args.model, requirement_simple, format_tool_prompt, query_type='simple'
                )
                hard_result = None
                error = None
            elif needs_hard:
                # Generate only hard query
                hard_result = generate_single_query(
                    records[i], args.model, requirement_hard, format_tool_prompt, query_type='hard'
                )
                simple_result = None
                error = None
            else:
                # This shouldn't happen due to our skip logic above
                continue
            
            trajectory_time = time.time() - trajectory_start_time
            
            if error:
                print(f"❌ Error processing record {i}: {error} (took {trajectory_time:.2f}s)")
                continue
            
            # Print timing information
            print(f"✅ Record {i+1} completed in {trajectory_time:.2f} seconds")
            
            # Store results based on generation method - only store generated queries
            if args.enable_multi_query:
                # Multi-query results with metadata
                if simple_result is not None:
                    records[i][f"query_simple_{args.model}"] = simple_result['best_query']
                    records[i][f"query_simple_{args.model}_candidates"] = simple_result.get('candidates', [])
                    records[i][f"query_simple_{args.model}_evaluation"] = simple_result.get('evaluation')
                    records[i][f"query_simple_{args.model}_selection_method"] = simple_result.get('selection_method')
                
                if hard_result is not None:
                    records[i][f"query_hard_{args.model}"] = hard_result['best_query']
                    records[i][f"query_hard_{args.model}_candidates"] = hard_result.get('candidates', [])
                    records[i][f"query_hard_{args.model}_evaluation"] = hard_result.get('evaluation')
                    records[i][f"query_hard_{args.model}_selection_method"] = hard_result.get('selection_method')
            else:
                # Single query results - only store generated queries
                if simple_result is not None:
                    records[i][f"query_simple_{args.model}"] = simple_result['query']
                if hard_result is not None:
                    records[i][f"query_hard_{args.model}"] = hard_result['query']
            
            # Store timing metadata
            records[i][f"query_generation_time_{args.model}"] = trajectory_time
            
            # Save immediately after each generation to prevent data loss
            with open(save_file, 'w') as f:
                json.dump(records, f, indent=4)
            print(f"💾 Saved updated record {i+1}")
                
        except Exception as e:
            print(f"ERROR: Error processing record {i}: {e}")
            continue
    
    # Final save for this domain
    with open(save_file, 'w') as f:
        json.dump(records, f, indent=4)
    print(f"Completed domain {domain}")

print("All domains completed!")