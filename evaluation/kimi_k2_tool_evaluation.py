#!/usr/bin/env python3
"""
Kimi K2 Tool Evaluation Script

This script evaluates Kimi K2's function calling capability using the transformed K2 tools.
It supports both parallel and sequential trajectory types and domain/all tool selection modes.

Usage Examples:
    # Domain mode (default)
    python evaluation/kimi_k2_tool_evaluation.py -tool_select domain -traj_type parallel

    # All tools mode
    python evaluation/kimi_k2_tool_evaluation.py -tool_select all -traj_type sequential

    # Reset mode to clear all progress
    python evaluation/kimi_k2_tool_evaluation.py -reset -tool_select domain

Dependencies:
- utils.checkpoints: Checkpoint save/load functionality
- utils.metrics: Evaluation metric calculations
- utils.model_providers: Model generation functions

Output:
- Checkpoint files: Progress tracking for resuming interrupted evaluations
- Log files: Detailed results for each domain with metrics
- Console logging: Real-time progress and error reporting
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from utils.checkpoints import load_checkpoint_json, save_checkpoint_json
    from utils.metrics import exact_match_tools, inclusion_tools, tool_traj_usage
    from utils.model_providers import generate_content_with_retry, extract_json_from_markdown_fence, sanitize_input
except ImportError as e:
    print(f"Warning: Could not import utils modules: {e}")
    print("Please ensure utils modules are available")

import json
import logging
import argparse
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

def load_k2_tools(tool_select: str, domain: str, base_data_dir: str) -> List[Dict[str, Any]]:
    """
    Load tools based on selection mode.
    
    Args:
        tool_select: Tool selection mode ("domain" or "all")
        domain: Current domain name
        base_data_dir: Base data directory path
        
    Returns:
        List of tools in K2 format
    """
    if tool_select == "domain":
        # Load domain-specific tools
        tool_file = os.path.join(base_data_dir, "kimi-k2_tool", f"{domain}_tool.json")
        if not os.path.exists(tool_file):
            raise FileNotFoundError(f"Tool file not found: {tool_file}")
        
        with open(tool_file, "r") as f:
            tools = json.load(f)
        return tools
    
    elif tool_select == "all":
        # Load all tools
        tool_file = os.path.join(base_data_dir, "kimi-k2_tool", "all_tools.json")
        if not os.path.exists(tool_file):
            raise FileNotFoundError(f"All tools file not found: {tool_file}")
        
        with open(tool_file, "r") as f:
            tools = json.load(f)
        return tools
    
    else:
        raise ValueError(f"Unsupported tool_select mode: {tool_select}")

def create_tool_selection_prompt(query: str, tools: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for tool selection using K2.
    
    Args:
        query: User query
        tools: List of tools in K2 format
        
    Returns:
        Formatted prompt for tool selection
    """
    tools_text = "\n".join([
        f"- {tool['function']['name']}: {tool['function']['description']}" 
        for tool in tools
    ])
    
    prompt = f"""Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.

Available tools:
{tools_text}

User Query: {query}

Respond with a JSON object in the following format:
{{
    "selected_tools": [
        {{
            "tool_name": "exact_tool_name_from_list",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
    ]
}}

Only select tools that are directly relevant to answering the query. If no tools are needed, return an empty array.
"""
    return prompt

def call_k2_with_tools(
    query: str, 
    tools: List[Dict[str, Any]], 
    model: str = "moonshot-v1-8k",
    max_tokens: int = 1024
) -> str:
    """
    Call K2 using model_providers for tool selection.
    
    Args:
        query: User query
        tools: List of tools in K2 format
        model: K2 model to use
        max_tokens: Maximum tokens for response
        
    Returns:
        K2's response text
    """
    prompt = create_tool_selection_prompt(query, tools)
    
    # Use the existing model_providers infrastructure
    response = generate_content_with_retry(model, prompt)
    
    return response

def extract_tool_calls_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from K2's text response.
    
    Args:
        response_text: K2's response text
        
    Returns:
        List of tool calls
    """
    try:
        # Try to extract JSON from the response
        import json
        
        # Extract JSON from markdown fence if present
        extracted_json = extract_json_from_markdown_fence(response_text)
        
        if isinstance(extracted_json, dict) and "selected_tools" in extracted_json:
            tool_calls = []
            for tool_call in extracted_json["selected_tools"]:
                tool_calls.append({
                    "tool_name": tool_call.get("tool_name", ""),
                    "parameters": tool_call.get("parameters", {}),
                    "id": f"call_{len(tool_calls)}"
                })
            return tool_calls
        else:
            # Fallback: try to parse as direct JSON
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "selected_tools" in parsed:
                tool_calls = []
                for tool_call in parsed["selected_tools"]:
                    tool_calls.append({
                        "tool_name": tool_call.get("tool_name", ""),
                        "parameters": tool_call.get("parameters", {}),
                        "id": f"call_{len(tool_calls)}"
                    })
                return tool_calls
    except Exception as e:
        print(f"Error extracting tool calls: {e}")
        return []
    
    return []

# main loop
parser = argparse.ArgumentParser(description="Kimi K2 Tool Evaluation Script")
parser.add_argument("-model", type=str, default="kimi-k2", help="K2 model to use")
parser.add_argument("-max_tokens", type=int, default=8000, help="Maximum tokens for K2 response")
parser.add_argument("-tool_select", type=str, default="domain", help="Tool selection mode", choices=["domain", "all"])
parser.add_argument("-traj_type", type=str, default="parallel", help="Trajectory type", choices=["parallel", "sequential"])
parser.add_argument("-traj_file", type=str, default="simple_ver", help="Trajectory file")
# Log settings
parser.add_argument("-log_dir", type=str, default="./log/kimi_k2_tool_evaluation", help="Log directory")
parser.add_argument("-chk_dir", type=str, default="./chk/kimi_k2_tool_evaluation", help="Checkpoint directory")
parser.add_argument("-base_data_dir", type=str, default="../public_data", help="Base data directory")
# Evaluation settings
parser.add_argument("-sleep_interval", type=float, default=1.0, help="Sleep interval between API calls (seconds)")
parser.add_argument("-max_retries", type=int, default=3, help="Maximum number of retry attempts for failed queries")
parser.add_argument("-reset", action="store_true", help="Reset checkpoint and start from beginning")
args = parser.parse_args()

# Create directories if they don't exist
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.chk_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, "k2_evaluation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# domain list
TARGET_DIRECTORY = Path(os.path.join(args.base_data_dir, args.traj_type))
domain_list = sorted([p.name for p in TARGET_DIRECTORY.iterdir() if p.is_dir()])
num_domain = len(domain_list)
print(f"Evaluating {num_domain} domains: {domain_list}.")
    
# Global checkpoint file
global_chk_file = os.path.join(
    args.chk_dir,
    f"global_{args.traj_type}_{args.traj_file}_{args.model}_{args.tool_select}.json",
)
    
# Handle reset option
if args.reset:
    logger.info("Reset mode: Removing all checkpoint and log files")
    
    # Remove global checkpoint file
    if os.path.exists(global_chk_file):
        os.remove(global_chk_file)
        logger.info("Reset: Removed global checkpoint file")
    
    # Remove domain-specific log files and checkpoint files
    for domain in domain_list:
        # Remove log files
        log_file = os.path.join(
            args.log_dir,
            f"{domain}/{args.traj_type}_{args.traj_file}_{args.model}_{args.tool_select}.json",
        )
        
        if os.path.exists(log_file):
            os.remove(log_file)
            logger.info(f"Reset: Removed log file for domain {domain}")
        
        # Remove domain-specific checkpoint files
        domain_chk_file = os.path.join(
            args.chk_dir,
            f"{domain}_{args.traj_type}_{args.traj_file}_{args.model}_{args.tool_select}.json",
        )
        
        if os.path.exists(domain_chk_file):
            os.remove(domain_chk_file)
            logger.info(f"Reset: Removed domain checkpoint file for {domain}")
    
    logger.info("Reset completed: All checkpoint and log files removed")
    global_checkpoint = {}
else:
    # Load global checkpoint
    global_checkpoint = load_checkpoint_json(global_chk_file)
    
# Main evaluation loop
for domain in domain_list:
    logger.info(f"Evaluating domain: {domain}")
    
    # Skip Finance domain for non-fixed modes due to context issues
    # if domain == "Finance" and args.tool_select != "all":
    #     logger.info(f"Skipping domain {domain} due to out-of-context issue!")
    #     continue
        
    # Determine log file path
    log_file = os.path.join(
        args.log_dir,
        f"{domain}/{args.traj_type}_{args.traj_file}_{args.model}_{args.tool_select}.json",
    )
    
    # Create the log directory for this domain
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    # Get checkpoint index for this domain
    chk_index = global_checkpoint.get(domain, 0)
    records = []
    
    # Handle resume logic
    if chk_index > 0:
        try:
            if os.path.exists(log_file):
                with open(log_file, "r") as file:
                    records = json.load(file)
                # Verify consistency between checkpoint and log file
                if len(records) != chk_index:
                    logger.warning(
                        f"Domain {domain}: Checkpoint-log file mismatch: checkpoint={chk_index}, log_records={len(records)}"
                    )
                    chk_index = min(chk_index, len(records))
                    records = records[:chk_index]
                    global_checkpoint[domain] = chk_index
                    save_checkpoint_json(global_checkpoint, global_chk_file)
                    logger.info(
                        f"Domain {domain}: Synchronized checkpoint and log file at index {chk_index}"
                    )
            else:
                logger.warning(
                    f"Domain {domain}: Checkpoint file exists ({chk_index}) but log file not found. Starting from beginning."
                )
                chk_index = 0
                global_checkpoint[domain] = 0
                save_checkpoint_json(global_checkpoint, global_chk_file)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(
                f"Domain {domain}: Error reading log file: {str(e)}. Starting from beginning."
            )
            chk_index = 0
            records = []
            global_checkpoint[domain] = 0
            save_checkpoint_json(global_checkpoint, global_chk_file)
        
    # Load test data
    data_file = os.path.join(
        args.base_data_dir, args.traj_type, f"{domain}/{args.traj_file}.json"
    )
    if not os.path.exists(data_file):
        logger.warning(f"Data file not found: {data_file}. Skipping domain {domain}.")
        continue
            
    with open(data_file, "r") as f:
        test_query = json.load(f)
    
    # Load tools
    try:
        tool_list = load_k2_tools(args.tool_select, domain, args.base_data_dir)
        logger.info(f"Loaded {len(tool_list)} tools for domain {domain}")
    except Exception as e:
        logger.error(f"Failed to load tools for domain {domain}: {e}")
        continue
        
    # Evaluate
    logger.info(
        f"Processing {len(test_query) - chk_index} queries starting from index {chk_index}"
    )
        
    for i in range(chk_index, len(test_query)):
        # Validate required fields
        if "query" not in test_query[i]:
            logger.error(f"Missing query field in test data at index {i}")
            continue
        if "tool list" not in test_query[i]:
            logger.error(f"Missing 'tool list' field in test data at index {i}")
            continue
            
        # Retry logic
        retry_count = 0
        success = False
            
        while retry_count < args.max_retries and not success:
            try:
                logger.info(
                    f"Processing query {i+1}/{len(test_query)} (index {i}) - attempt {retry_count + 1}"
                )
 
                # Call K2 with tools
                response_text = call_k2_with_tools(
                    test_query[i]["query"], 
                    tool_list, 
                    args.model,
                    args.max_tokens
                )
                
                # Extract tool calls from response
                tool_calls = extract_tool_calls_from_response(response_text)
                    
                # Convert tool calls to expected format
                pred_tool_list = []
                for tool_call in tool_calls:
                    pred_tool_list.append({
                        "tool name": tool_call["tool_name"],
                        "parameters": tool_call["parameters"]
                    })
                    
                # Create result record
                result = {
                    "query": test_query[i]["query"],
                    "pred tool list": pred_tool_list,
                    "gt tool list": test_query[i]["tool list"],
                    "gt answer": test_query[i].get("final_answer", ""),
                    "trajectory_type": test_query[i].get("trajectory_type", "unknown"),
                    "task_name": test_query[i].get("task_name", "unknown"),
                    "task_description": test_query[i].get("task_description", ""),
                    "k2_response": response_text,
                    "tool_calls": tool_calls,
                    "domain": domain,
                    "index": i
                }
                    
                # Calculate metrics
                try:
                    if args.traj_type == "parallel":
                        result["traj_exact_match"] = exact_match_tools(
                            result["gt tool list"], result["pred tool list"]
                        )
                    elif args.traj_type == "sequential":
                        result["traj_exact_match"] = exact_match_tools(
                            result["gt tool list"], result["pred tool list"], order=True
                        )
                    result["traj_inclusion"] = inclusion_tools(
                        result["gt tool list"], result["pred tool list"]
                    )
                    result["tool_traj_usage"] = tool_traj_usage(
                        result["gt tool list"], result["pred tool list"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate metrics for query {i}: {e}")
                    result.update({
                        "traj_exact_match": 0.0,
                        "traj_inclusion": 0.0,
                        "tool_traj_usage": 0.0
                    })
                    
                records.append(result)
                success = True
                
                logger.info(
                    f"Query {i+1} completed - Tool calls: {len(tool_calls)}, "
                    f"Exact match: {result.get('traj_exact_match', 0.0):.3f}, "
                    f"Inclusion: {result.get('traj_inclusion', 0.0):.3f}"
                )
                    
            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Error processing query {i}, attempt {retry_count}: {str(e)}"
                )
                    
                if retry_count >= args.max_retries:
                    # Store failed result
                    result = {
                        "query": test_query[i]["query"],
                        "pred tool list": [],
                        "gt tool list": test_query[i]["tool list"],
                        "gt answer": test_query[i].get("final_answer", ""),
                        "trajectory_type": test_query[i].get("trajectory_type", "unknown"),
                        "task_name": test_query[i].get("task_name", "unknown"),
                        "task_description": test_query[i].get("task_description", ""),
                        "k2_response": "",
                        "tool_calls": [],
                        "domain": domain,
                        "index": i,
                        "error": str(e),
                        "traj_exact_match": 0.0,
                        "traj_inclusion": 0.0,
                        "tool_traj_usage": 0.0
                    }
                    records.append(result)
                    logger.error(f"Failed to process query {i} after {args.max_retries} attempts")
                else:
                    # Wait before retry
                    time.sleep(args.sleep_interval * (2 ** retry_count))
            
        # Save progress after each query
        with open(log_file, "w") as f:
            json.dump(records, f, indent=2)
        
        # Update checkpoint
        global_checkpoint[domain] = i + 1
        save_checkpoint_json(global_checkpoint, global_chk_file)
        
        # Sleep between queries
        time.sleep(args.sleep_interval)
        
    # Calculate domain summary
    if records:
        # Calculate average tool_traj_usage score
        tool_usage_scores = []
        for r in records:
            tool_usage = r.get("tool_traj_usage", [])
            if isinstance(tool_usage, list) and tool_usage:  # If it's a non-empty list
                # Calculate average of boolean values (True=1, False=0)
                avg_score = sum(tool_usage) / len(tool_usage)
                tool_usage_scores.append(avg_score)
            elif isinstance(tool_usage, (int, float)):  # If it's already a number
                tool_usage_scores.append(tool_usage)
        
        domain_metrics = {
            "traj_exact_match": sum(r.get("traj_exact_match", 0) for r in records) / len(records),
            "traj_inclusion": sum(r.get("traj_inclusion", 0) for r in records) / len(records),
            "tool_traj_usage": sum(tool_usage_scores) / len(tool_usage_scores) if tool_usage_scores else 0.0,
            "total_queries": len(records),
            "successful_queries": len([r for r in records if "error" not in r])
        }
        
        logger.info(f"Domain {domain} completed - Metrics: {domain_metrics}")
        
        # Add summary to records
        records.append({"domain_summary": domain_metrics})
        
        # Save final results
        with open(log_file, "w") as f:
            json.dump(records, f, indent=2)

logger.info("Kimi K2 tool evaluation completed!")
