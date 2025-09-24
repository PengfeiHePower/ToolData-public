"""
Tool Evaluation Script - Unified Evaluation for Multiple Tool Selection Modes

This script provides a comprehensive evaluation framework for testing tool usage prediction
across different domains and tool selection strategies. It supports multiple evaluation
modes through the -tool_select parameter and handles both parallel and sequential tool
execution patterns.

Supported Tool Selection Modes:
- domain: Uses domain-specific tools for each evaluation domain
- all: Uses all available tools across all domains
- fixed: Uses a fixed-size tool pool with specified size (-k parameter)
- retrieval: Uses retrieval-based tool selection with embedding models

Key Features:
- Multi-domain evaluation (Travel, Music, Weather, Restaurant, Shopping, Movie, Book, Game, News, Sports)
- Robust checkpointing and resume functionality
- Comprehensive error handling with retry logic
- Multiple evaluation metrics (exact_match, inclusion, retrieval_rate, tool_traj_usage)
- Support for both parallel and sequential trajectory types
- Automatic file organization with trajectory type in filenames
- Reset mode for complete cleanup of all checkpoint and log files

Usage Examples:
    # Domain mode (default)
    python evaluation/tool_evaluation_public.py -tool_select domain -model claude_v37

    # Fixed mode with tool pool size
    python evaluation/tool_evaluation_public.py -tool_select fixed -k 20 -model claude_v37

    # Retrieval mode with embedding model
    python evaluation/tool_evaluation_public.py -tool_select retrieval -top_k 20 -emb_model all-MiniLM

    # Reset mode to clear all progress
    python evaluation/tool_evaluation_public.py -reset -tool_select domain

Dependencies:
- utils.model_providers: Centralized LLM interaction functions
- utils.checkpoints: Checkpoint save/load functionality
- utils.metrics: Evaluation metric calculations
- utils.retriever: Tool retrieval functions (for retrieval mode)

Output:
- Checkpoint files: Progress tracking for resuming interrupted evaluations
- Log files: Detailed results for each domain with metrics
- Console logging: Real-time progress and error reporting
"""

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
from pathlib import Path

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
from utils.checkpoints import load_checkpoint_json, save_checkpoint_json
from utils.metrics import (
    exact_match_tools,
    inclusion_tools,
    tool_traj_usage,
    retrieval_rate,
)

# Import retriever functions
try:
    from utils.retriever import load_retriever_model, load_encoded_tools, retrieve_tools

    RETRIEVER_AVAILABLE = True
except ImportError:
    print("Warning: Retriever functions not available. Retrieval mode will not work.")
    RETRIEVER_AVAILABLE = False

    def load_retriever_model(model_name):
        raise NotImplementedError("Retriever not available")

    def load_encoded_tools(tools, domain, model_name):
        raise NotImplementedError("Retriever not available")

    def retrieve_tools(model, query, tools, embeddings, top_k):
        raise NotImplementedError("Retriever not available")


# All retry logic and exception classes are now centralized in utils.model_providers

parser = argparse.ArgumentParser("evaluate tool usage")
# test setting
parser.add_argument(
    "-model",
    type=str,
    default="claude_v37",
    help="model name",
    choices=[
        "qwen-8b",
        "qwen-32b",
        "qwen-30b-A3B",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-8b",
        "claude_v4",
        "claude_v37",
        "nova_pro",
        "nova_lite",
        "gpt-5-mini",
        "o4-mini",
        "gpt-oss:20b",
        "gpt-oss:120b",
        "deepseek-chat",
        "qwen-235b-A22B",
        "kimi-k2"
    ],
)
parser.add_argument("-tool_select", type=str, default="domain", help="tool selection mode", choices=["domain", "all", "fixed", "retrieval"])
parser.add_argument("-method", type=str, default="direct", help="problem solving method, direct prompting, CoT, etc.", choices=["direct", "cot"])
parser.add_argument("-k", type=int, default=20, help="tool pool size for fixed pool and top_k for retrieval")
# retrieval settings (only used when tool_select is 'retrieval')
parser.add_argument("-emb_model", type=str, default="all-MiniLM", help="embedding model for retrieval, necessary for retrieval", choices=["ToolBench_IR", "bge-large", "all-MiniLM"])
parser.add_argument("-retrieve_pool", type=str, default="domain", help="retrieve pool, all means retrieve all tools, subtool means retrieve subtools", choices=["all", "domain"])
# trajectory and file settings
parser.add_argument("-traj_type", type=str, default="parallel", help="trajectory type", choices=["parallel", "sequential"])
parser.add_argument("-traj_file", type=str, default="simple_ver", help="trajectory file")
# log setting
parser.add_argument("-log_dir", type=str, default="./log/simple_query/model", help="log directory")
parser.add_argument("-chk_dir", type=str, default="./chk/simple_query/model", help="checkpoint directory")
parser.add_argument("-base_data_dir", type=str, default="../public_data", help="base data directory")
parser.add_argument("-sleep_interval", type=float, default=0.5, help="sleep interval between API calls (seconds)")
parser.add_argument(
    "-max_retries",
    type=int,
    default=1,
    help="maximum number of retry attempts for failed queries",
)
parser.add_argument("-reset", action="store_true", help="reset checkpoint and start from beginning")

args = parser.parse_args()

# Create directories if they don't exist
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.chk_dir, exist_ok=True)

# Validate required arguments based on tool_select mode
if args.tool_select == "fixed" and args.k is None:
    parser.error("Fixed mode requires -k argument for tool pool size")
if args.tool_select == "retrieval" and not RETRIEVER_AVAILABLE:
    parser.error("Retrieval mode requires retriever functions which are not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, "evaluation.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

#### data loading
# domain list
TARGET_DIRECTORY = Path(os.path.join(args.base_data_dir, args.traj_type))
domain_list = sorted([p.name for p in TARGET_DIRECTORY.iterdir() if p.is_dir()])
num_domain = len(domain_list)
print(f"Evaluating {num_domain} domains: {domain_list}.")

# load all tools
with open(os.path.join(args.base_data_dir, "tools/all_tools.json"), "r") as f:
    all_tool = json.load(f)

# Global checkpoint file
if args.tool_select == "fixed":
    global_chk_file = os.path.join(
        args.chk_dir,
        f"global_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{str(args.k)}.json",
    )
elif args.tool_select == "retrieval":
    global_chk_file = os.path.join(
        args.chk_dir,
        f"global_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{args.emb_model}_{str(args.k)}_{args.retrieve_pool}.json",
    )
else:
    global_chk_file = os.path.join(
        args.chk_dir,
        f"global_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}.json",
    )
# logger.info(f"Global checkpoint file: {global_chk_file}")
# exit(0)

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
        if args.tool_select == "fixed":
            log_file = os.path.join(
                args.log_dir,
                f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{str(args.k)}.json",
            )
        elif args.tool_select == "retrieval":
            log_file = os.path.join(
                args.log_dir,
                f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{args.emb_model}_{str(args.k)}_{args.retrieve_pool}.json",
            )
        else:
            log_file = os.path.join(
                args.log_dir,
                f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}.json",
            )

        if os.path.exists(log_file):
            os.remove(log_file)
            logger.info(f"Reset: Removed log file for domain {domain}")

        # Remove domain-specific checkpoint files if they exist
        if args.tool_select == "fixed":
            domain_chk_file = os.path.join(
                args.chk_dir,
                f"{domain}_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{str(args.k)}.json",
            )
        elif args.tool_select == "retrieval":
            domain_chk_file = os.path.join(
                args.chk_dir,
                f"{domain}_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{args.emb_model}_{str(args.k)}_{args.retrieve_pool}.json",
            )
        else:
            domain_chk_file = os.path.join(
                args.chk_dir,
                f"{domain}_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}.json",
            )

        if os.path.exists(domain_chk_file):
            os.remove(domain_chk_file)
            logger.info(f"Reset: Removed domain checkpoint file for {domain}")

    # Also remove evaluation.log if it exists
    evaluation_log = os.path.join(args.log_dir, "evaluation.log")
    if os.path.exists(evaluation_log):
        os.remove(evaluation_log)
        logger.info("Reset: Removed evaluation.log file")

    logger.info("Reset completed: All checkpoint and log files removed")

    # Initialize empty global checkpoint after reset
    global_checkpoint = {}
else:
    # Load global checkpoint
    global_checkpoint = load_checkpoint_json(global_chk_file)

# main
## load prompt
with open("evaluation/evaluation_prompt.json", "r") as f:
    prompts = json.load(f)

for domain in domain_list:
    logger.info(f"Evaluating domain: {domain}")

    if domain == "Finance":
        if args.tool_select not in ["fixed", "retrieval"]:
            logger.info(f"Skipping domain {domain} due to out-of-context issue!")
            continue

    # Determine log file path based on tool_select mode
    if args.tool_select == "fixed":
        log_file = os.path.join(
            args.log_dir,
            f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{str(args.k)}.json",
        )
    elif args.tool_select == "retrieval":
        log_file = os.path.join(
            args.log_dir,
            f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{args.emb_model}_{str(args.k)}_{args.retrieve_pool}.json",
        )
    else:
        log_file = os.path.join(
            args.log_dir,
            f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}.json",
        )

    # Create the log directory for this domain
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Get checkpoint index for this domain
    chk_index = global_checkpoint.get(domain, 0)
    records = []

    # Handle resume logic with proper synchronization check
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
                    # Use the smaller of the two to ensure consistency
                    chk_index = min(chk_index, len(records))
                    # Trim records if necessary
                    records = records[:chk_index]
                    # Update global checkpoint to match actual records
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

    # load data
    data_file = os.path.join(
        args.base_data_dir, args.traj_type, f"{domain}/{args.traj_file}.json"
    )
    with open(data_file, "r") as f:
        test_query = json.load(f)

    # load domain tool
    with open(os.path.join(args.base_data_dir, f"tools/{domain}_tool.json"), "r") as f:
        domain_tool = json.load(f)

    # Initialize tool_list based on tool_select mode
    if args.tool_select == "domain":
        tool_list = domain_tool
    elif args.tool_select == "all":
        tool_list = all_tool
    elif args.tool_select == "retrieval":
        # Load retriever model and embeddings for the selected pool
        logger.info(f"Loading retriever model: {args.emb_model}")
        retriever_model = load_retriever_model(args.emb_model)
        if args.retrieve_pool == "all":
            logger.info("Loading embeddings for ALL tools pool")
            tool_pool, pool_embeddings = load_encoded_tools(
                all_tool, "ALL", args.emb_model
            )
        else:
            logger.info(f"Loading embeddings for domain: {domain}")
            tool_pool, pool_embeddings = load_encoded_tools(
                domain_tool, domain, args.emb_model
            )
        tool_list = tool_pool  # Will be overridden per query with retrieved subset

    # evaluate
    logger.info(
        f"Processing {len(test_query) - chk_index} queries starting from index {chk_index}"
    )

    for i in range(chk_index, len(test_query)):
        # Validate required fields exist
        if "query" not in test_query[i]:
            raise KeyError(f"Missing query field: query")
        if "tool list" not in test_query[i]:
            raise KeyError("Missing 'tool_list' field in test data")

        # robust settings
        retry_count = 0
        success = False

        # tool_list setup based on tool_select mode
        if args.tool_select == "fixed":
            gt_tool_list = test_query[i]["tool list"]
            gt_names = [d["tool name"] for d in gt_tool_list]
            matched = [d for d in domain_tool if d["tool name"] in gt_names]
            unmatched = [d for d in domain_tool if d["tool name"] not in gt_names]
            tool_list = matched + random.sample(unmatched, args.k - len(gt_names))
        elif args.tool_select == "retrieval":
            # Retrieve tools for this specific query from the selected pool
            query_text = test_query[i]["query"]
            retrieved_tools = retrieve_tools(
                retriever_model,
                query_text,
                tool_pool,
                pool_embeddings,
                top_k=args.k,
            )
            tool_list = retrieved_tools
            logger.info(
                f"Retrieved {len(tool_list)} tools for query: '{query_text[:50]}...' from pool '{args.retrieve_pool}'"
            )

        # Preprocess tool list to include only essential fields
        essential_fields = ["parent tool name", "required_parameters", "optional_parameters", "tool name", "tool description", "API name", "domain name"]
        
        processed_tool_list = []
        for tool in tool_list:
            processed_tool = {}
            for field in essential_fields:
                if field in tool:
                    processed_tool[field] = tool[field]
                else:
                    # Set default values for missing fields
                    if field == "parent tool name":
                        processed_tool[field] = "Unknown"
                    elif field in ["required_parameters", "optional_parameters"]:
                        processed_tool[field] = []
                    elif field == "tool description":
                        processed_tool[field] = "No description available"
                    elif field == "API name":
                        processed_tool[field] = "Unknown"
                    elif field == "domain name":
                        processed_tool[field] = "Unknown"
            processed_tool_list.append(processed_tool)

        while retry_count < args.max_retries and not success:
            try:
                logger.info(
                    f"Processing query {i+1}/{len(test_query)} (index {i}) - attempt {retry_count + 1}"
                )
                # Sanitize inputs before template replacement
                query_text = sanitize_input(test_query[i]["query"])
                tools_text = sanitize_input(str(processed_tool_list))
                prompt_query = (
                    prompts[args.method]
                    .replace("<query>", query_text)
                    .replace("<tools>", tools_text)
                )

                # Generate content with retry logic already built-in
                response = generate_content_with_retry(args.model, prompt_query)

                # Validate response
                if not response or not isinstance(response, str):
                    raise ValueError("Empty or invalid response from model")

                logger.debug(f"Generated response for query {i}: {response[:100]}...")

                # Extract and validate JSON with retry for parsing errors
                try:
                    extracted_json = extract_json_from_markdown_fence(
                        response
                    )
                    if not isinstance(extracted_json, dict):
                        raise ValueError("Extracted result is not a dictionary")
                    
                    if "tool list" not in extracted_json:
                        raise ValueError("No 'tool list' field found in response")
                    
                    extracted_tools = extracted_json["tool list"]
                    if not isinstance(extracted_tools, list):
                        raise ValueError("'tool list' field is not a list")
                except ValueError as parse_error:
                    logger.warning(
                        f"JSON parsing failed for query {i}, attempt {retry_count + 1}: {str(parse_error)}"
                    )
                    if retry_count == args.max_retries - 1:
                        # On final attempt, store empty list and continue
                        extracted_tools = []
                        logger.error(
                            f"Final parsing attempt failed for query {i}. Storing empty tool list."
                        )
                    else:
                        raise parse_error

                results = {
                    "query": test_query[i]["query"],
                    "pred tool list": extracted_tools,
                    "gt tool list": test_query[i]["tool list"],
                    "gt answer": test_query[i].get("final_answer", ""),
                    "trajectory_type": test_query[i].get("trajectory_type", "unknown"),
                    "task_name": test_query[i].get("task_name", "unknown"),
                    "task_description": test_query[i].get("task_description", ""),
                }

                # calculate metrics
                if args.traj_type == "parallel":
                    results["traj_exact_match"] = exact_match_tools(
                        results["gt tool list"], results["pred tool list"]
                    )
                elif args.traj_type == "sequential":
                    results["traj_exact_match"] = exact_match_tools(
                        results["gt tool list"], results["pred tool list"], order=True
                    )
                results["traj_inclusion"] = inclusion_tools(
                    results["gt tool list"], results["pred tool list"]
                )
                results["tool_traj_usage"] = tool_traj_usage(
                    results["gt tool list"], results["pred tool list"]
                )

                # Add retrieved tools if using retrieval mode
                if args.tool_select == "retrieval":
                    results["retrieved_tools"] = processed_tool_list
                    results["retrieval_rate"] = retrieval_rate(
                        results["gt tool list"], results["retrieved_tools"]
                    )

                records.append(results)
                success = True

                # save checkpoint and log file after successful processing
                global_checkpoint[domain] = i + 1
                save_checkpoint_json(global_checkpoint, global_chk_file)
                try:
                    with open(log_file, "w") as f:
                        json.dump(records, f, indent=2)
                    logger.info(
                        f"Successfully processed and saved query {i+1}/{len(test_query)}"
                    )

                    # Exit after first successful query for testing
                    # logger.info("Testing mode: Exiting after first successful query")
                    # exit(0)

                except Exception as save_error:
                    logger.error(
                        f"Failed to save log file after query {i+1}: {str(save_error)}"
                    )
                    # If we can't save the log file, we should also revert the checkpoint
                    global_checkpoint[domain] = i
                    save_checkpoint_json(global_checkpoint, global_chk_file)
                    raise save_error

            except (APIError, RateLimitError) as e:
                retry_count += 1
                logger.warning(
                    f"API error processing query {i}, attempt {retry_count}/{args.max_retries}: {str(e)}"
                )
                if retry_count < args.max_retries:
                    wait_time = 2**retry_count  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to process query {i} after {args.max_retries} attempts due to API error: {str(e)}"
                    )
                    # Add failed result to maintain index consistency
                    results = {
                        "query": test_query[i]["query"],
                        "pred tool list": [],
                        "gt tool list": test_query[i]["tool list"],
                        "gt answer": test_query[i].get("final_answer", ""),
                        "trajectory_type": test_query[i].get(
                            "trajectory_type", "unknown"
                        ),
                        "task_name": test_query[i].get("task_name", "unknown"),
                        "task_description": test_query[i].get("task_description", ""),
                        "error": f"API_ERROR: {str(e)}",
                        "traj_exact_match": False,
                        "traj_inclusion": 0.0,
                        "tool_traj_usage": [],
                    }

                    # Add retrieved tools if using retrieval mode
                    if args.tool_select == "retrieval":
                        results["retrieved_tools"] = processed_tool_list
                        results["retrieval_rate"] = None

                    records.append(results)
                    success = True  # Mark as "success" to continue processing
                    global_checkpoint[domain] = i + 1
                    save_checkpoint_json(global_checkpoint, global_chk_file)
                    # Save log file immediately after checkpoint
                    try:
                        with open(log_file, "w") as f:
                            json.dump(records, f, indent=2)
                    except Exception as save_error:
                        logger.error(
                            f"Failed to save log file after API error for query {i+1}: {str(save_error)}"
                        )
                        global_checkpoint[domain] = i
                        save_checkpoint_json(
                            global_chk_file, global_checkpoint
                        )  # Revert checkpoint
                        raise save_error

            except (ValueError, KeyError) as e:
                retry_count += 1
                logger.warning(
                    f"Parsing/validation error processing query {i}, attempt {retry_count}/{args.max_retries}: {str(e)}"
                )
                if retry_count < args.max_retries:
                    time.sleep(1)  # Short wait for parsing errors
                else:
                    logger.error(
                        f"Failed to process query {i} after {args.max_retries} attempts due to parsing error: {str(e)}"
                    )
                    # Add failed result to maintain index consistency
                    results = {
                        "query": test_query[i].get("query", "QUERY_ERROR"),
                        "pred tool list": [],
                        "gt tool list": test_query[i].get("tool list", []),
                        "gt answer": test_query[i].get("final_answer", ""),
                        "trajectory_type": test_query[i].get(
                            "trajectory_type", "unknown"
                        ),
                        "task_name": test_query[i].get("task_name", "unknown"),
                        "task_description": test_query[i].get("task_description", ""),
                        "error": f"PARSING_ERROR: {str(e)}",
                        "traj_exact_match": False,
                        "traj_inclusion": 0.0,
                        "tool_traj_usage": [],
                    }

                    # Add retrieved tools if using retrieval mode
                    if args.tool_select == "retrieval":
                        results["retrieved_tools"] = processed_tool_list
                        results["retrieval_rate"] = None
                    records.append(results)
                    success = True  # Mark as "success" to continue processing
                    global_checkpoint[domain] = i + 1
                    save_checkpoint_json(global_checkpoint, global_chk_file)
                    # Save log file immediately after checkpoint
                    try:
                        with open(log_file, "w") as f2:
                            json.dump(records, f2, indent=2)
                    except Exception as save_error:
                        logger.error(
                            f"Failed to save log file after parsing error for query {i+1}: {str(save_error)}"
                        )
                        global_checkpoint[domain] = i
                        save_checkpoint_json(
                            global_chk_file, global_checkpoint
                        )  # Revert checkpoint
                        raise save_error

            except Exception as e:
                logger.critical(f"Unexpected error processing query {i}: {str(e)}")
                # Save checkpoint and results before re-raising
                global_checkpoint[domain] = i
                save_checkpoint_json(global_checkpoint, global_chk_file)
                try:
                    with open(log_file, "w") as f:
                        json.dump(records, f, indent=2)
                except Exception as save_error:
                    logger.error(f"Failed to save log file before re-raising: {str(save_error)}")
                raise

        # Configurable sleep interval between queries
        time.sleep(args.sleep_interval)

        # Note: Log file is now saved immediately after each successful query processing
        # This periodic save is kept as a backup for additional safety
        if (i + 1) % 10 == 0 or i == len(test_query) - 1:
            try:
                with open(log_file, "w") as f:
                    json.dump(records, f, indent=2)
                logger.info(f"Backup save completed after processing query {i+1}")
            except Exception as save_error:
                logger.error(
                    f"Failed to backup save results after query {i+1}: {str(save_error)}"
                )

    # Final save
    try:
        with open(log_file, "w") as f:
            json.dump(records, f, indent=2)
        logger.info(f"Completed processing all queries for domain {domain}")
    except Exception as final_save_error:
        logger.error(
            f"Failed to save final results for domain {domain}: {str(final_save_error)}"
        )

logger.info("Tool evaluation completed successfully.")
