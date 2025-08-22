import os
import json
import requests
import logging
import time
import sys
from typing import Dict, List, Any, Optional
from copy import deepcopy
from functools import wraps
import argparse
import subprocess
import random

# Import basic model functions from centralized providers
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_with_retry,
    sanitize_input,
    extract_json_from_markdown_fence,
    APIError,
    RateLimitError,
    ModelNotAvailableError
)

# All retry logic and exception classes are now centralized in utils.model_providers

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
parser.add_argument('-log_dir', type=str, default='./log/simple_query/model', help='log directory')
parser.add_argument('-chk_dir', type=str, default='./chk/simple_query/model', help='checkpoint directory')
parser.add_argument('-base_data_dir', type=str, default='/home/ec2-user/mountS3/newToolData', help='base data directory')
parser.add_argument('-sleep_interval', type=float, default=0.5, help='sleep interval between API calls (seconds)')
parser.add_argument('-reset', action='store_true', help='reset checkpoint and start from beginning')

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

# All generation functions with retry logic are now centralized in utils.model_providers

def load_checkpoint(chk_file: str) -> Dict[str, int]:
    """Load checkpoint data from file. Returns dict with domain -> index mapping."""
    try:
        with open(chk_file, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {}  # no checkpoint file or invalid content, start from beginning

def save_checkpoint(domain_progress: Dict[str, int], chk_file: str) -> None:
    """Save checkpoint data to file."""
    os.makedirs(os.path.dirname(chk_file), exist_ok=True)
    with open(chk_file, "w") as f:
        json.dump(domain_progress, f, indent=2)

#### data loading
# domain list
with open(os.path.join(args.base_data_dir, 'selected_category.json'), 'r') as f:
    select_cate = json.load(f)
domain_list = list(select_cate.keys())
num_domain = len(domain_list)

# all tools
with open(os.path.join(args.base_data_dir, 'tools/all_tool.json'), 'r') as f:
    all_tool = json.load(f)

# Global checkpoint file
if args.tool_select == 'fixed':
    global_chk_file = os.path.join(args.chk_dir, f"global_{args.traj_file}_{args.type}_{args.model}_{args.tool_select}_{str(args.k)}.json")
else:
    global_chk_file = os.path.join(args.chk_dir, f"global_{args.traj_file}_{args.type}_{args.model}_{args.tool_select}.json")

# Handle reset option
if args.reset:
    if os.path.exists(global_chk_file):
        os.remove(global_chk_file)
        logger.info("Reset: Removed global checkpoint file")
    # Also remove domain-specific log files
    for domain in domain_list:
        if args.tool_select == 'fixed':
            log_file = os.path.join(args.log_dir, f"{domain}/{args.traj_file}_{args.type}_{args.model}_{args.tool_select}_{str(args.k)}.json")
        else:
            log_file = os.path.join(args.log_dir, f"{domain}/{args.traj_file}_{args.type}_{args.model}_{args.tool_select}.json")
        if os.path.exists(log_file):
            os.remove(log_file)
            logger.info(f"Reset: Removed log file for domain {domain}")

# Load global checkpoint
global_checkpoint = load_checkpoint(global_chk_file)

# main
prompt = """
Given the tool list:
<tools>

Please slove the query:
<query>

Please selct proper tools and, provide me when to call and how to call, in the following json format:

```json
[
{"tool name":[fused name], "tool description":[fused description], "required parameters":[{"name": xxx,"value": xxx}, {"name": xxx,"value": xxx},...], "optional parameters":[{"name": xxx,"value": xxx},...]},
...
]
```

P.S. If the later tool needs the result of the previous tool, please place a placeholder in the corresponding parameter position, for example, {"name": xxx,"value": "{{xxx from the previous step}}"}.
"""
for domain in domain_list:
    logger.info(f"Evaluating domain: {domain}")
    if args.tool_select == 'fixed':
        log_file = os.path.join(args.log_dir, f"{domain}/{args.traj_file}_{args.type}_{args.model}_{args.tool_select}_{str(args.k)}.json")
    else:
        log_file = os.path.join(args.log_dir, f"{domain}/{args.traj_file}_{args.type}_{args.model}_{args.tool_select}.json")
    
    # Get checkpoint index for this domain
    chk_index = global_checkpoint.get(domain, 0)
    records = []
    
    # Handle resume logic with proper synchronization check
    if chk_index > 0:
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as file:
                    records = json.load(file)
                # Verify consistency between checkpoint and log file
                if len(records) != chk_index:
                    logger.warning(f"Domain {domain}: Checkpoint-log file mismatch: checkpoint={chk_index}, log_records={len(records)}")
                    # Use the smaller of the two to ensure consistency
                    chk_index = min(chk_index, len(records))
                    # Trim records if necessary
                    records = records[:chk_index]
                    # Update global checkpoint to match actual records
                    global_checkpoint[domain] = chk_index
                    save_checkpoint(global_checkpoint, global_chk_file)
                    logger.info(f"Domain {domain}: Synchronized checkpoint and log file at index {chk_index}")
            else:
                logger.warning(f"Domain {domain}: Checkpoint file exists ({chk_index}) but log file not found. Starting from beginning.")
                chk_index = 0
                global_checkpoint[domain] = 0
                save_checkpoint(global_checkpoint, global_chk_file)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Domain {domain}: Error reading log file: {str(e)}. Starting from beginning.")
            chk_index = 0
            records = []
            global_checkpoint[domain] = 0
            save_checkpoint(global_checkpoint, global_chk_file)

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
                
                prompt_query = deepcopy(prompt)
                # Sanitize inputs before template replacement
                query_text = sanitize_input(test_query[i][f"query_{args.type}_{args.gen_model_query}"])
                tools_text = sanitize_input(str(tool_list))
                
                prompt_query = prompt_query.replace("<query>", query_text)
                prompt_query = prompt_query.replace("<tools>", tools_text)
                
                # Generate content with retry logic already built-in
                response = generate_content_with_retry(args.model, prompt_query)
                
                # Validate response
                if not response or not isinstance(response, str):
                    raise ValueError("Empty or invalid response from model")
                
                logger.debug(f"Generated response for query {i}: {response[:100]}...")
                
                # Extract and validate JSON with retry for parsing errors
                try:
                    extracted_tools = extract_json_from_markdown_fence(response, expect_dict=False)
                    if not isinstance(extracted_tools, list):
                        raise ValueError("Extracted result is not a list")
                except ValueError as parse_error:
                    logger.warning(f"JSON parsing failed for query {i}, attempt {retry_count + 1}: {str(parse_error)}")
                    if retry_count == max_retries - 1:
                        # On final attempt, store empty list and continue
                        extracted_tools = []
                        logger.error(f"Final parsing attempt failed for query {i}. Storing empty tool list.")
                    else:
                        raise parse_error
                
                results = {
                    'query': test_query[i][f"query_{args.type}_{args.gen_model_query}"], 
                    'tool list': extracted_tools, 
                    'gt tool list': test_query[i]['tool list'], 
                    'num_tools_used': test_query[i]['num_tools_used']
                }
                records.append(results)
                success = True
                
                # save checkpoint and log file after successful processing
                global_checkpoint[domain] = i+1
                save_checkpoint(global_checkpoint, global_chk_file)
                try:
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    with open(log_file, 'w') as f:
                        json.dump(records, f, indent=4)
                    logger.info(f"Successfully processed and saved query {i+1}/{len(test_query)}")
                except Exception as save_error:
                    logger.error(f"Failed to save log file after query {i+1}: {str(save_error)}")
                    # If we can't save the log file, we should also revert the checkpoint
                    global_checkpoint[domain] = i
                    save_checkpoint(global_checkpoint, global_chk_file)
                    raise save_error
                
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
                        'tool list': [], 
                        'gt tool list': test_query[i].get('tool list', []), 
                        'num_tools_used': test_query[i].get('num_tools_used', 0),
                        'error': f"API_ERROR: {str(e)}"
                    }
                    records.append(results)
                    success = True  # Mark as "success" to continue processing
                    global_checkpoint[domain] = i+1
                    save_checkpoint(global_checkpoint, global_chk_file)
                    # Save log file immediately after checkpoint
                    try:
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, 'w') as f:
                            json.dump(records, f, indent=4)
                    except Exception as save_error:
                        logger.error(f"Failed to save log file after API error for query {i+1}: {str(save_error)}")
                        global_checkpoint[domain] = i
                        save_checkpoint(global_checkpoint, global_chk_file)  # Revert checkpoint
                        raise save_error
                    
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
                        'tool list': [], 
                        'gt tool list': test_query[i].get('tool list', []), 
                        'num_tools_used': test_query[i].get('num_tools_used', 0),
                        'error': f"PARSING_ERROR: {str(e)}"
                    }
                    records.append(results)
                    success = True  # Mark as "success" to continue processing
                    global_checkpoint[domain] = i+1
                    save_checkpoint(global_checkpoint, global_chk_file)
                    # Save log file immediately after checkpoint
                    try:
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, 'w') as f:
                            json.dump(records, f, indent=4)
                    except Exception as save_error:
                        logger.error(f"Failed to save log file after parsing error for query {i+1}: {str(save_error)}")
                        global_checkpoint[domain] = i
                        save_checkpoint(global_checkpoint, global_chk_file)  # Revert checkpoint
                        raise save_error
                    
            except Exception as e:
                logger.critical(f"Unexpected error processing query {i}: {str(e)}")
                # Save checkpoint and results before re-raising
                global_checkpoint[domain] = i
                save_checkpoint(global_checkpoint, global_chk_file)
                with open(log_file, 'w') as f:
                    json.dump(records, f, indent=4)
                raise
        
        # Configurable sleep interval between queries
        time.sleep(args.sleep_interval)
        
        # Note: Log file is now saved immediately after each successful query processing
        # This periodic save is kept as a backup for additional safety
        if (i + 1) % 10 == 0 or i == len(test_query) - 1:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'w') as f:
                    json.dump(records, f, indent=4)
                logger.info(f"Backup save completed after processing query {i+1}")
            except Exception as save_error:
                logger.error(f"Failed to backup save results after query {i+1}: {str(save_error)}")
    
    # Final save
    try:
        with open(log_file, 'w') as f:
            json.dump(records, f, indent=4)
        logger.info(f"Completed processing all queries for domain {domain}")
    except Exception as final_save_error:
        logger.error(f"Failed to save final results for domain {domain}: {str(final_save_error)}")
    # exit(0)

