import os
import json
import re
import argparse
import time
import logging
import sys
from copy import deepcopy
import concurrent.futures

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser('evaluate tool usage')
parser.add_argument('-model', type=str, default='claude_v37', help='model name', choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
parser.add_argument('-traj_file', type=str, default='simple_traj_parallel_selfplay_gemini-2.5-pro_v2', help='trajectory file')
parser.add_argument('-save_dir', type=str, default='/home/ec2-user/mountS3/newToolData/simple_query', help='log directory')
parser.add_argument('-chk_dir', type=str, default='./chk/traj_to_query', help='checkpoint directory')
parser.add_argument('-enable_self_play', action='store_true', help='enable self-play query improvement')
parser.add_argument('-max_iterations', type=int, default=3, help='maximum self-play improvement iterations')
parser.add_argument('-max_retries', type=int, default=5, help='maximum retry attempts per operation')
parser.add_argument('-retry_delay', type=float, default=1.0, help='initial retry delay in seconds')
# parser.add_argument('-save_name', type=str, default='simple_traj_parallel_claude_v37', help='save name')
args = parser.parse_args()

# Robust generation functions (preserved from original file)

# generate content function with retry logic
def generate_content_with_retry(model, prompt, max_retries=5, base_delay=1.0):
    """
    Generate content with exponential backoff retry logic.
    
    Args:
        model: Model name
        prompt: Prompt text
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        Generated content string
    
    Raises:
        Exception: If all retries are exhausted
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if model in vllm_api_meta:
                return generate_content_vllm(model, prompt, max_new_tokens=8000, temperature=0.7)
            elif model in gemini_models:
                return generate_content_gemini(model, prompt)
            elif model in bedrock_meta:
                return generate_content_bedrock(model, prompt, max_tokens=8000, temperature=0.3)
            elif model in openai_models:
                return generate_content_openai(model, prompt)
            else:
                raise ValueError(f"Unknown model: {model}")
                
        except Exception as e:
            last_error = e
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            
            # Log the error and retry info
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for model {model}: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed for model {model}")
                raise Exception(f"Content generation failed after {max_retries} attempts. Last error: {last_error}")

# Backward compatibility wrapper
def generate_content(model, prompt):
    return generate_content_with_retry(model, prompt, args.max_retries, args.retry_delay)

# extract json from markdown fence with enhanced error handling
def extract_json_from_markdown_fence_with_retry(text: str, max_retries=3):
    """
    Extracts and parses JSON from markdown with multiple fallback strategies.
    
    Args:
        text (str): The input string containing a markdown block with JSON content.
        max_retries (int): Maximum number of parsing attempts with different strategies
        
    Returns:
        dict: Parsed JSON data
        
    Raises:
        ValueError: If all parsing strategies fail
    """
    strategies = [
        # Strategy 1: Standard JSON block
        r"```json\s*(\{.*?\})\s*```",
        # Strategy 2: JSON block without language specifier
        r"```\s*(\{.*?\})\s*```",
        # Strategy 3: JSON without code blocks
        r"(\{[^}]*\"[^\"]*\"[^}]*\})",
        # Strategy 4: Multiline JSON
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    ]
    
    errors = []
    
    for i, pattern in enumerate(strategies):
        try:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                
                # Try to parse the JSON
                result = json.loads(json_str)
                logger.debug(f"Successfully parsed JSON using strategy {i+1}")
                return result
                
        except (json.JSONDecodeError, AttributeError) as e:
            errors.append(f"Strategy {i+1}: {e}")
            continue
    
    # If all strategies fail, raise detailed error
    error_msg = f"Failed to parse JSON from text. Attempted strategies: {'; '.join(errors)}"
    logger.error(f"{error_msg}\nOriginal text: {text[:500]}...")
    raise ValueError(error_msg)

# Backward compatibility wrapper
def extract_json_from_markdown_fence(text: str):
    return extract_json_from_markdown_fence_with_retry(text, max_retries=3)

# Merged criticism and improvement prompt
merged_criticism_improvement_prompt = """
You are a query quality evaluator and improver. Analyze the following generated query against the tool trajectory, then provide an improved version if needed.

<Generated Query>
[query]
</Generated Query>

<Tool Trajectory>
[tool_list]
</Tool Trajectory>

<Query Requirements>
[query_requirements]
</Query Requirements>

Please evaluate the query based on these criteria:
- **Trajectory Match:** Does the query *require* the same tools and call order as the trajectory?
- **Query Type Fit:** Does the query fulfill the query requirements?

If the query needs improvement, provide a better version. If it's already good, return the original query.

Provide your response in this JSON format:
```json
{
    "needs_improvement": true/false,
    "issues": ["list of specific issues found if any"],
    "improved_query": "improved query here (or original if no improvement needed)"
}
```
"""

def load_checkpoint(chk_file):
    try:
        with open(chk_file, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # no checkpoint file, start from beginning

def save_checkpoint(index, chk_file):
    with open(chk_file, "w") as f:
        f.write(str(index))

def generate_both_queries_parallel_selfplay(record, args, requirement_simple, requirement_hard, format_tool_prompt):
    """
    Generate simple and hard queries in parallel for self-play version.
    
    Args:
        record: Single record with tool list
        args: Command line arguments
        requirement_simple: Simple query requirements
        requirement_hard: Hard query requirements
        format_tool_prompt: Template for formatting prompts
    
    Returns:
        tuple: (simple_result, hard_result, error)
    """
    def generate_simple_query():
        """Generate simple query with optional self-play improvement"""
        try:
            # Generate initial simple query
            prompt_query_simple = requirement_simple + format_tool_prompt
            prompt_query_simple = prompt_query_simple.replace("[tool list]", str(record['tool list']))
            response_simple = generate_content(args.model, prompt_query_simple)
            initial_simple_query = extract_json_from_markdown_fence(response_simple)['query']
            
            # Self-play improvement for simple query (optional)
            if args.enable_self_play:
                logger.debug(f"Starting self-play improvement for simple query")
                simple_improvement = self_play_improve_query(
                    args.model, initial_simple_query, record['tool list'], 
                    requirement_simple, args.max_iterations, args.max_retries
                )
                return {
                    "query": simple_improvement['final_query'],
                    "improvement_history": simple_improvement['improvement_history'],
                    "errors": simple_improvement.get('errors', []),
                    "success": simple_improvement.get('success', False),
                    "query_changed": simple_improvement.get('query_changed', False)
                }
            else:
                return {
                    "query": initial_simple_query,
                    "improvement_history": [],
                    "errors": [],
                    "success": True,
                    "query_changed": False
                }
        except Exception as e:
            raise Exception(f"Simple query generation failed: {e}")
    
    def generate_hard_query():
        """Generate hard query with optional self-play improvement"""
        try:
            # Generate initial hard query
            prompt_query_hard = requirement_hard + format_tool_prompt
            prompt_query_hard = prompt_query_hard.replace("[tool list]", str(record['tool list']))
            response_hard = generate_content(args.model, prompt_query_hard)
            initial_hard_query = extract_json_from_markdown_fence(response_hard)['query']
            
            # Self-play improvement for hard query (optional)
            if args.enable_self_play:
                logger.debug(f"Starting self-play improvement for hard query")
                hard_improvement = self_play_improve_query(
                    args.model, initial_hard_query, record['tool list'], 
                    requirement_hard, args.max_iterations, args.max_retries
                )
                return {
                    "query": hard_improvement['final_query'],
                    "improvement_history": hard_improvement['improvement_history'],
                    "errors": hard_improvement.get('errors', []),
                    "success": hard_improvement.get('success', False),
                    "query_changed": hard_improvement.get('query_changed', False)
                }
            else:
                return {
                    "query": initial_hard_query,
                    "improvement_history": [],
                    "errors": [],
                    "success": True,
                    "query_changed": False
                }
        except Exception as e:
            raise Exception(f"Hard query generation failed: {e}")
    
    # Run both generations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        logger.info("Starting parallel generation of simple and hard queries...")
        
        # Submit both tasks
        simple_future = executor.submit(generate_simple_query)
        hard_future = executor.submit(generate_hard_query)
        
        # Wait for both to complete with timeout
        try:
            simple_result = simple_future.result(timeout=300)  # 5 min timeout per query type
            hard_result = hard_future.result(timeout=300)
            
            logger.info("Both queries generated successfully in parallel!")
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

def process_record_with_retry(record_index, record, args, requirement_simple, requirement_hard, format_tool_prompt):
    """Process a single record with retry logic."""
    max_attempts = args.max_retries
    
    for attempt in range(max_attempts):
        try:
            simple_key = f"query_simple_{args.model}"
            hard_key = f"query_hard_{args.model}"
            
            # Skip if already processed
            if simple_key in record and hard_key in record:
                logger.debug(f"Record {record_index+1} already processed, skipping")
                return True, record
            
            # Check if we need to generate either query
            need_simple = simple_key not in record
            need_hard = hard_key not in record
            
            if need_simple or need_hard:
                logger.debug(f"Generating queries for record {record_index+1} (simple: {need_simple}, hard: {need_hard})")
                
                # Generate both queries in parallel (even if only one is needed, for simplicity)
                simple_result, hard_result, error = generate_both_queries_parallel_selfplay(
                    record, args, requirement_simple, requirement_hard, format_tool_prompt
                )
                
                if error:
                    raise Exception(f"Parallel query generation failed: {error}")
                
                # Store simple query results if needed
                if need_simple:
                    record[simple_key] = simple_result['query']
                    record[f"{simple_key}_improvement_history"] = simple_result['improvement_history']
                    record[f"{simple_key}_errors"] = simple_result['errors']
                    record[f"{simple_key}_success"] = simple_result['success']
                    record[f"{simple_key}_query_changed"] = simple_result.get('query_changed', False)
                
                # Store hard query results if needed
                if need_hard:
                    record[hard_key] = hard_result['query']
                    record[f"{hard_key}_improvement_history"] = hard_result['improvement_history']
                    record[f"{hard_key}_errors"] = hard_result['errors']
                    record[f"{hard_key}_success"] = hard_result['success']
                    record[f"{hard_key}_query_changed"] = hard_result.get('query_changed', False)
            
            # Add processing metadata
            record['processing_metadata'] = {
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model': args.model,
                'attempt_count': attempt + 1,
                'self_play_enabled': args.enable_self_play
            }
            
            logger.info(f"Successfully processed record {record_index+1} on attempt {attempt+1}")
            return True, record
            
        except Exception as e:
            error_msg = f"Attempt {attempt+1}/{max_attempts} failed for record {record_index+1}: {e}"
            logger.error(error_msg)
            
            if attempt < max_attempts - 1:
                delay = args.retry_delay * (2 ** attempt)
                logger.info(f"Retrying record {record_index+1} in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Record {record_index+1} failed after {max_attempts} attempts")
                return False, record
    
    return False, record

def self_play_improve_query(model, original_query, tool_list, query_requirements, max_iterations=2, max_retries=3):
    """
    Improve query using merged self-play with comprehensive error handling.
    
    Args:
        model: The model name for generation
        original_query: Initial generated query
        tool_list: Tool trajectory information
        query_requirements: Requirements for the query
        max_iterations: Maximum number of improvement iterations
        max_retries: Maximum number of retries per iteration for parsing errors
    
    Returns:
        dict: Contains final query, improvement history, and metadata
    """
    current_query = original_query
    improvement_history = []
    total_errors = []
    
    logger.info(f"Starting self-play improvement with {max_iterations} iterations and {max_retries} retries per iteration")
    
    for iteration in range(max_iterations):
        success = False
        last_error = None
        iteration_errors = []
        
        logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
        
        for retry in range(max_retries):
            try:
                # Single step: Criticize and improve
                merged_prompt = (merged_criticism_improvement_prompt
                               .replace('[query]', current_query)  
                               .replace('[tool_list]', str(tool_list))
                               .replace('[query_requirements]', query_requirements))
                
                logger.debug(f"Generating content for iteration {iteration + 1}, retry {retry + 1}")
                response = generate_content(model, merged_prompt)
                
                logger.debug(f"Parsing JSON response for iteration {iteration + 1}, retry {retry + 1}")
                response_data = extract_json_from_markdown_fence(response)
                
                # Validate response structure
                if not isinstance(response_data, dict):
                    raise ValueError("Response is not a valid dictionary")
                
                # Debug: Log what we got from the LLM
                logger.info(f"LLM response for iteration {iteration + 1}: needs_improvement={response_data.get('needs_improvement')}, "
                          f"issues={len(response_data.get('issues', []))} issues")
                
                # Check if improvement is needed
                needs_improvement = response_data.get("needs_improvement", False)
                if not needs_improvement:
                    logger.info(f"No improvement needed at iteration {iteration + 1} - query is already good")
                    # Record this as a successful completion without change
                    iteration_result = {
                        "iteration": iteration + 1,
                        "previous_query": current_query,
                        "issues": response_data.get("issues", []),
                        "improved_query": current_query,  # Keep original since no improvement needed
                        "retry_count": retry + 1,
                        "no_improvement_needed": True
                    }
                    improvement_history.append(iteration_result)
                    success = True
                    # Break out of both retry loop AND iteration loop
                    break
                    
                # Get the improved query
                new_query = response_data.get("improved_query", current_query)
                if not new_query or new_query.strip() == "":
                    raise ValueError("Empty or missing improved_query in response")
                
                # Record this iteration
                iteration_result = {
                    "iteration": iteration + 1,
                    "previous_query": current_query,
                    "issues": response_data.get("issues", []),
                    "improved_query": new_query,
                    "retry_count": retry + 1
                }
                improvement_history.append(iteration_result)
                
                current_query = new_query
                success = True
                logger.info(f"Successfully completed iteration {iteration + 1} on retry {retry + 1}")
                break
                
            except Exception as e:
                last_error = e
                iteration_errors.append(f"Retry {retry + 1}: {str(e)}")
                logger.warning(f"Error in self-play iteration {iteration + 1}, retry {retry + 1}: {e}")
                
                if retry < max_retries - 1:
                    delay = args.retry_delay * (2 ** retry)  # Exponential backoff
                    logger.info(f"Retrying iteration {iteration + 1} in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
        
        # Record errors for this iteration
        if iteration_errors:
            total_errors.extend(iteration_errors)
        
        # If all retries failed, break the iteration loop
        if not success:
            error_msg = f"Failed to complete iteration {iteration + 1} after {max_retries} retries. Last error: {last_error}"
            logger.error(error_msg)
            total_errors.append(error_msg)
            break
        
        # If no improvement was needed, break out of iteration loop
        if success and len(improvement_history) > 0 and improvement_history[-1].get("no_improvement_needed", False):
            logger.info(f"Stopping iterations early - no improvement needed")
            break
    
    # Determine if query was actually improved
    query_changed = current_query != original_query
    no_improvement_iterations = sum(1 for h in improvement_history if h.get("no_improvement_needed", False))
    
    result = {
        "final_query": current_query,
        "improvement_history": improvement_history,
        "total_iterations": len(improvement_history),
        "errors": total_errors,
        "success": len(improvement_history) > 0,
        "query_changed": query_changed,
        "no_improvement_iterations": no_improvement_iterations
    }
    
    if query_changed:
        logger.info(f"Self-play improvement completed: Query WAS improved over {len(improvement_history)} iterations")
    else:
        logger.info(f"Self-play improvement completed: Query was NOT changed (LLM said no improvement needed in {no_improvement_iterations} iterations)")
    
    if total_errors:
        logger.warning(f"Self-play had {len(total_errors)} errors during processing")
    
    return result

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
    print(f"Generating query for domain: {domain}")
    # load checkpoint
    save_file = os.path.join(args.save_dir, f"{domain}/{args.traj_file}.json")
    chk_file = os.path.join(args.chk_dir, f"{args.traj_file}_{args.model}_{domain}.txt")
    chk_index = load_checkpoint(chk_file)
    requirement_simple = gen_prompt['simple']
    requirement_hard = gen_prompt[f'{domain}_hard']
    # if os.path.isfile(log_file):
    with open(save_file, 'r') as file:
        records = json.load(file)
    # Process records with robust error handling
    failed_records = []
    successful_records = 0
    
    for i in range(chk_index, len(records)):
        logger.info(f"Processing record {i+1}/{len(records)} for domain {domain}")
        
        # Time the query generation for this trajectory
        trajectory_start_time = time.time()
        
        # Use the retry function to process this record
        success, updated_record = process_record_with_retry(
            i, records[i], args, requirement_simple, requirement_hard, format_tool_prompt
        )
        
        trajectory_time = time.time() - trajectory_start_time
        
        # Update the record in the array
        records[i] = updated_record
        
        if success:
            successful_records += 1
            logger.info(f"✅ Record {i+1} completed in {trajectory_time:.2f} seconds")
            # Store timing metadata
            records[i][f"query_generation_time_{args.model}"] = trajectory_time
        else:
            logger.error(f"❌ Record {i+1} failed after {trajectory_time:.2f} seconds")
            failed_record = {
                'record_index': i,
                'domain': domain,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': 'Max retries exceeded',
                'processing_time': trajectory_time
            }
            failed_records.append(failed_record)
        
        # Save checkpoint after each record (successful or failed)
        save_checkpoint(i+1, chk_file)
        
        # Save results after each record for safety
        try:
            with open(save_file, 'w') as f:
                json.dump(records, f, indent=4)
        except Exception as save_error:
            logger.error(f"Failed to save results after record {i+1}: {save_error}")
    
    # Summary for this domain
    total_processed = len(records) - chk_index
    logger.info(f"Domain {domain} complete: {successful_records}/{total_processed} successful")
    
    if failed_records:
        # Save failed records for this domain
        failed_records_file = os.path.join(args.chk_dir, f"failed_records_{args.model}_{domain}.json")
        try:
            with open(failed_records_file, 'w') as f:
                json.dump(failed_records, f, indent=4)
            logger.info(f"Failed records saved to: {failed_records_file}")
        except Exception as e:
            logger.error(f"Could not save failed records: {e}")
    logger.info("All domains processed successfully!")

