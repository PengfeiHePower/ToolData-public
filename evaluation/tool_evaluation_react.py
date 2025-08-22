import os
import json
import logging
import time
import sys
from typing import Dict, List, Any, Optional
from copy import deepcopy
import argparse
import re
import subprocess
import random
import pickle

# Import basic model functions from centralized providers
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_with_retry,
    extract_json_from_markdown_fence,
    sanitize_input,
    APIError,
    RateLimitError,
    ModelNotAvailableError
)

# Import tool execution function
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))
from tool_exe import tool_exe

# API keys are already validated by utils.model_providers

# Custom exceptions (additional ones not in utils.model_providers)
class InvalidResponseError(APIError):
    """Raised when API returns invalid response"""
    pass

# All retry logic is now handled by the centralized generate_content_with_retry function

parser = argparse.ArgumentParser('evaluate tool usage')
# test setting
parser.add_argument('-model', type=str, default='gemini-2.5-pro', help='model name', choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite', 'gpt-4.1-mini', 'o4-mini', 'gpt-oss:20b', 'gpt-oss:120b'])
parser.add_argument('-tool_select', type=str, default='domain', help='tool selection mode', choices=['domain', 'all', 'fixed'])
parser.add_argument('-type', type=str, help='type', choices=['hard', 'simple'])
parser.add_argument('-k', type=int, help='tool pool size, necessary for fixed mode')
# query setting
parser.add_argument('-query_dir', type=str, help='query directory', default = '/home/ec2-user/mountS3/newToolData/simple_query')
parser.add_argument('-traj_file', type=str, default='simple_traj_parallel_consis_gemini-2.5-pro_v2', help='trajectory file')
parser.add_argument('-gen_model_query', type=str, help='generation model of queries', default = 'claude_v37', choices=['gemini-2.5-pro','claude_v37'])
# log setting
parser.add_argument('-log_dir', type=str, default='./log/simple_query/react', help='log directory')
parser.add_argument('-chk_dir', type=str, default='./chk/simple_query/react', help='checkpoint directory')
parser.add_argument('-base_data_dir', type=str, default='/home/ec2-user/mountS3/newToolData', help='base data directory')
parser.add_argument('-sleep_interval', type=float, default=0.5, help='sleep interval between API calls (seconds)')
parser.add_argument('-reset', action='store_true', help='reset and start from beginning, ignoring existing checkpoints')

args = parser.parse_args()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, 'evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP and API logs
# logging.getLogger('urllib3').setLevel(logging.WARNING)
# logging.getLogger('requests').setLevel(logging.WARNING)
# logging.getLogger('httpx').setLevel(logging.WARNING)
# logging.getLogger('google').setLevel(logging.WARNING)
# logging.getLogger('google.generativeai').setLevel(logging.WARNING)

# Create directories if they don't exist
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.chk_dir, exist_ok=True)

# All generation and utility functions are now imported from utils.model_providers

def load_checkpoint(checkpoint_file: str) -> dict:
    """Load checkpoint data from pickle file."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            return {}
    return {}

def save_checkpoint(checkpoint_file: str, data: dict) -> None:
    """Save checkpoint data to pickle file."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)

#### data loading
# domain list
with open(os.path.join(args.base_data_dir, 'selected_category.json'), 'r') as f:
    select_cate = json.load(f)
domain_list = list(select_cate.keys())
num_domain = len(domain_list)

# all tools
with open(os.path.join(args.base_data_dir, 'tools/all_tool.json'), 'r') as f:
    all_tool = json.load(f)

# ReAct instruction template
react_instruction = """You are an agent that solves a task using tools with ReAct: interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Tool[<json_object>], which call a single tool with the necessary parameters; The <json_object> strictly follows this format:
```json
{"tool name": [tool name], "tool description": [tool description], "required parameters": [{"name": xxx, value: xxx}, ...], "optional parameters": [{"name": xxx, "value": xxx}, ...], "domain name": [domain name], "parent tool name": [parent tool name], "API name": [API name]}
```
(2) Finish[answer], which returns the answer and finishes the task.

You MUST proceed ONE tool call at a time, and available tools:
<tools>
"""

def execute_action(action):
    """Execute a ReAct action with real tool execution and return an observation."""
    try:
        action = action.strip()
        
        if action.startswith("Tool[") and action.endswith("]"):
            # Extract and parse the JSON object using robust extraction
            try:
                tool_spec = extract_json_from_markdown_fence(action)
                if not tool_spec:
                    # Fallback: extract manually and parse directly
                    json_content = action[5:-1].strip()
                    tool_spec = json.loads(json_content)
                
                tool_name = tool_spec.get("tool name", "").strip()
                
                # Execute the tool directly using the JSON specification
                try:
                    result = tool_exe(tool_spec)
                    tool_spec["exe_result"] = result
                    return {
                        "type": "tool", 
                        "tool_spec": tool_spec, 
                        "observation": f"Tool '{tool_name}' executed successfully. Result: {result}"
                    }
                except Exception as exe_error:
                    tool_spec["exe_result"] = f"ERROR: {str(exe_error)}"
                    return {
                        "type": "tool", 
                        "tool_spec": tool_spec, 
                        "observation": f"Tool '{tool_name}' execution failed: {str(exe_error)}"
                    }
                    
            except json.JSONDecodeError as json_error:
                return f"Invalid JSON format in Tool action: {str(json_error)}"
        
        elif action.startswith("Finish[") and action.endswith("]"):
            # Extract and return the final answer
            answer_content = action[7:-1].strip()  # Remove "Finish[" and "]"
            return {"type": "finish", "answer": answer_content}
            
        else:
            return f"Invalid action format. Use Tool[json_object] or Finish[answer]."
            
    except Exception as e:
        return f"Error executing action: {str(e)}"


def run_react_conversation(query, tool_list, model, max_turns=8, to_print=False):
    """
    Run a complete ReAct conversation following the efficient original webthink pattern.
    
    Args:
        query (str): The user query to solve
        tool_list (list): List of available tools  
        model (str): Model name for generation
        max_turns (int): Maximum number of ReAct turns (default 8 like original)
        to_print (bool): Whether to print debug info
    
    Returns:
        dict: Result containing conversation history, final answer, and call counts
    """
    # Initialize prompt with ReAct instruction template (like original webthink_prompt)
    query_text = sanitize_input(query)
    tools_text = sanitize_input(str(tool_list))
    
    prompt = react_instruction.replace("<query>", query_text).replace("<tools>", tools_text)
    prompt += query_text + "\n"
    
    if to_print:
        print(f"Query: {query_text}")
    
    n_calls, n_badcalls = 0, 0
    final_answer = ""
    called_tools = []  # Track tools called during conversation
    
    # Main ReAct loop (exactly like original webthink)
    for i in range(1, max_turns + 1):
        n_calls += 1
        
        # Generate Thought and Action with stop token (like original)
        thought_action = generate_content_with_retry(model, prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        
        try:
            # Parse thought and action (like original)
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            # Handle parsing failure (like original)
            if to_print:
                print(f'Parse error: {thought_action}')
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = generate_content_with_retry(model, prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()
        
        # Execute action and get observation (like original step function)
        obs_result = execute_action(action)
        
        # Handle observation result
        if isinstance(obs_result, dict) and obs_result.get("type") == "finish":
            final_answer = obs_result.get("answer", "")
            obs = f"Task completed. Final answer: {final_answer}"
            done = True
        elif isinstance(obs_result, dict) and obs_result.get("type") == "tool":
            # Tool was called - collect it and use observation
            tool_spec = obs_result.get("tool_spec", {})
            called_tools.append(tool_spec)
            obs = obs_result.get("observation", "").replace('\\n', '')  # Clean obs like original
            done = False
        else:
            obs = str(obs_result).replace('\\n', '')  # Clean obs like original
            done = False
        
        # Build step string (exactly like original)
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        
        if to_print:
            print(step_str)
        
        if done:
            break
    
    # Force completion if not done (like original)
    if not done:
        obs_result = execute_action("Finish[]")
        if isinstance(obs_result, dict) and obs_result.get("type") == "finish":
            final_answer = obs_result.get("answer", "")
    
    if to_print:
        print(f"Final answer: {final_answer}")
        print(f"Calls: {n_calls}, Bad calls: {n_badcalls}\n")
    
    # Return info similar to original webthink
    return {
        'conversation_history': prompt,
        'final_answer': final_answer,
        'called_tools': called_tools,  # List of tool_spec dicts with exe_result
        'n_calls': n_calls,
        'n_badcalls': n_badcalls
    }

# Create checkpoint file path
if args.tool_select == 'fixed':
    checkpoint_file = os.path.join(args.chk_dir, f"{args.traj_file}_{args.type}_{args.model}_{args.tool_select}_{str(args.k)}.pkl")
else:
    checkpoint_file = os.path.join(args.chk_dir, f"{args.traj_file}_{args.type}_{args.model}_{args.tool_select}.pkl")

# Load checkpoint data (or start fresh if reset flag is used)
if args.reset:
    logger.info("Reset flag enabled - starting from beginning")
    checkpoint_data = {}
else:
    checkpoint_data = load_checkpoint(checkpoint_file)

# main
for domain in domain_list:
    logger.info(f"Evaluating domain: {domain}")
    
    # Check if domain is already completed
    if domain in checkpoint_data.get('completed_domains', []):
        logger.info(f"Skipping already completed domain: {domain}")
        continue
    
    # Create domain-specific directories
    domain_log_dir = os.path.join(args.log_dir, domain)
    os.makedirs(domain_log_dir, exist_ok=True)
    
    # Create log file path
    if args.tool_select == 'fixed':
        log_file = os.path.join(domain_log_dir, f"{args.traj_file}_{args.type}_{args.model}_{args.tool_select}_{str(args.k)}.json")
    else:
        log_file = os.path.join(domain_log_dir, f"{args.traj_file}_{args.type}_{args.model}_{args.tool_select}.json")
    
    # Load existing results or start fresh
    domain_key = f"{domain}_{args.type}_{args.model}_{args.tool_select}" + (f"_{args.k}" if args.tool_select == 'fixed' else "")
    chk_index = checkpoint_data.get('processed_queries', {}).get(domain_key, 0)
    
    if chk_index > 0:
        try:
            with open(log_file, 'r') as file:
                records = json.load(file)
        except FileNotFoundError:
            records = []
    else:
        records = []

    # load data
    data_file = os.path.join(args.query_dir, f'{domain}/{args.traj_file}.json')
    with open(data_file, 'r') as f:
        test_query = json.load(f)
    # load domain tool
    with open(os.path.join(args.base_data_dir, f'tools/{domain}/true_mix_subtool.json'), 'r') as f:
        domain_tool = json.load(f)
    if args.tool_select == 'domain':
        tool_list = domain_tool
    elif args.tool_select == 'all':
        tool_list = all_tool

    # evaluate
    logger.info(f"Processing {len(test_query) - chk_index} queries starting from index {chk_index}")

    for i in range(chk_index, len(test_query)):
    # for i in range(chk_index, 30):
        max_retries = 3
        retry_count = 0
        success = False

        if args.tool_select == 'fixed':
            gt_tool_list = test_query[i]['tool list']
            gt_names = [d['tool name'] for d in gt_tool_list]
            matched = [d for d in domain_tool if d['tool name'] in gt_names]
            unmatched = [d for d in domain_tool if d['tool name'] not in gt_names]
            tool_list = matched + random.sample(unmatched, args.k-len(gt_names))

        
        while retry_count < max_retries and not success:
            try:
                logger.info(f"Processing query {i+1}/{len(test_query)} (index {i}) - attempt {retry_count + 1}")
                
                # Validate required fields exist
                if f"query_{args.type}_{args.gen_model_query}" not in test_query[i]:
                    raise KeyError(f"Missing query field: query_{args.type}_{args.gen_model_query}")
                if 'tool list' not in test_query[i]:
                    raise KeyError("Missing 'tool list' field in test data")
                if 'num_tools_used' not in test_query[i]:
                    raise KeyError("Missing 'num_tools_used' field in test data")
                
                # Run ReAct conversation using the wrapped function
                query_text = test_query[i][f"query_{args.type}_{args.gen_model_query}"]
                
                try:
                    # Use the wrapped ReAct function
                    react_result = run_react_conversation(query_text, tool_list, args.model, max_turns=8)
                    
                    # Extract results
                    conversation = react_result['conversation_history']
                    n_calls = react_result['n_calls']
                    final_answer = react_result['final_answer']
                    called_tools = react_result['called_tools']
                    
                except Exception as parse_error:
                    logger.warning(f"ReAct conversation failed for query {i}, attempt {retry_count + 1}: {str(parse_error)}")
                    if retry_count == max_retries - 1:
                        final_answer = ""
                        called_tools = []
                        conversation = ""
                        n_calls = 0
                        logger.error(f"Final ReAct conversation failed for query {i}.")
                    else:
                        raise ValueError(str(parse_error))
                
                results = {
                    'query': test_query[i][f"query_{args.type}_{args.gen_model_query}"], 
                    'final_answer': final_answer,
                    'tool list': called_tools,  # List of tool_spec dicts with exe_result
                    'gt tool list': test_query[i]['tool list'], 
                    'num_tools_used': test_query[i]['num_tools_used']
                }
                records.append(results)
                success = True
                
                # Update checkpoint data
                if 'processed_queries' not in checkpoint_data:
                    checkpoint_data['processed_queries'] = {}
                checkpoint_data['processed_queries'][domain_key] = i + 1
                
                # Save both checkpoint and log file after each query for interrupt/resume safety
                save_checkpoint(checkpoint_file, checkpoint_data)
                try:
                    with open(log_file, 'w') as f:
                        json.dump(records, f, indent=4)
                except Exception as save_error:
                    logger.error(f"Failed to save log file after query {i+1}: {str(save_error)}")
                    
                logger.info(f"Successfully processed query {i+1}/{len(test_query)}")
                
            except (APIError, RateLimitError) as e:
                retry_count += 1
                logger.warning(f"API error processing query {i}, attempt {retry_count}/{max_retries}: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to process query {i} after {max_retries} attempts due to API error: {str(e)}")
                    # Add failed result to maintain index consistency
                    results = {
                        'query': test_query[i].get(f"query_{args.type}_{args.gen_model_query}", "QUERY_ERROR"), 
                        'final_answer': "",
                        'tool list': [],
                        'gt tool list': test_query[i].get('tool list', []), 
                        'num_tools_used': test_query[i].get('num_tools_used', 0),
                        'error': f"API_ERROR: {str(e)}"
                    }
                    records.append(results)
                    success = True  # Mark as "success" to continue processing
                    
                    # Update checkpoint and save both files
                    if 'processed_queries' not in checkpoint_data:
                        checkpoint_data['processed_queries'] = {}
                    checkpoint_data['processed_queries'][domain_key] = i + 1
                    save_checkpoint(checkpoint_file, checkpoint_data)
                    try:
                        with open(log_file, 'w') as f:
                            json.dump(records, f, indent=4)
                    except Exception as save_error:
                        logger.error(f"Failed to save log file after error in query {i+1}: {str(save_error)}")
                    
            except (ValueError, KeyError) as e:
                retry_count += 1
                logger.warning(f"Parsing/validation error processing query {i}, attempt {retry_count}/{max_retries}: {str(e)}")
                if retry_count < max_retries:
                    time.sleep(1)  # Short wait for parsing errors
                else:
                    logger.error(f"Failed to process query {i} after {max_retries} attempts due to parsing error: {str(e)}")
                    # Add failed result to maintain index consistency
                    results = {
                        'query': test_query[i].get(f"query_{args.type}_{args.gen_model_query}", "QUERY_ERROR"), 
                        'final_answer': "",
                        'tool list': [],
                        'gt tool list': test_query[i].get('tool list', []), 
                        'num_tools_used': test_query[i].get('num_tools_used', 0),
                        'error': f"PARSING_ERROR: {str(e)}"
                    }
                    records.append(results)
                    success = True  # Mark as "success" to continue processing
                    
                    # Update checkpoint and save both files
                    if 'processed_queries' not in checkpoint_data:
                        checkpoint_data['processed_queries'] = {}
                    checkpoint_data['processed_queries'][domain_key] = i + 1
                    save_checkpoint(checkpoint_file, checkpoint_data)
                    try:
                        with open(log_file, 'w') as f:
                            json.dump(records, f, indent=4)
                    except Exception as save_error:
                        logger.error(f"Failed to save log file after error in query {i+1}: {str(save_error)}")
                    
            except Exception as e:
                logger.critical(f"Unexpected error processing query {i}: {str(e)}")
                # Save checkpoint and results before re-raising
                if 'processed_queries' not in checkpoint_data:
                    checkpoint_data['processed_queries'] = {}
                checkpoint_data['processed_queries'][domain_key] = i
                save_checkpoint(checkpoint_file, checkpoint_data)
                with open(log_file, 'w') as f:
                    json.dump(records, f, indent=4)
                raise
        
        # Configurable sleep interval between queries
        time.sleep(args.sleep_interval)
    
    # Mark domain as completed and save final state
    try:
        logger.info(f"Completed processing all queries for domain {domain}")
        
        # Mark domain as completed
        if 'completed_domains' not in checkpoint_data:
            checkpoint_data['completed_domains'] = []
        checkpoint_data['completed_domains'].append(domain)
        save_checkpoint(checkpoint_file, checkpoint_data)
        
        # Final log file save (though it should already be up-to-date)
        with open(log_file, 'w') as f:
            json.dump(records, f, indent=4)
            
    except Exception as final_save_error:
        logger.error(f"Failed to save final state for domain {domain}: {str(final_save_error)}")
    # exit(0)

