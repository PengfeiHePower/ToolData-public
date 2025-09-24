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
from pathlib import Path

# Import basic model functions from centralized providers
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_with_retry,
    extract_json_from_markdown_fence,
    sanitize_input,
    APIError,
    RateLimitError,
    ModelNotAvailableError,
)

# Import tool execution function
from utils.tool_exe import tool_exe

# Import retrieval functions
from utils.retriever import load_retriever_model, load_encoded_tools, retrieve_tools
from utils.metrics import (
    exact_match_tools,
    inclusion_tools,
    tool_traj_usage,
    retrieval_rate,
)
from utils.checkpoints import load_checkpoint_json, save_checkpoint_json

# API keys are already validated by utils.model_providers


# Custom exceptions (additional ones not in utils.model_providers)
class InvalidResponseError(APIError):
    """Raised when API returns invalid response"""

    pass


# All retry logic is now handled by the centralized generate_content_with_retry function

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
        "gpt-4.1-mini",
        "o4-mini",
        "gpt-oss:20b",
        "gpt-oss:120b",
    ],
)
parser.add_argument(
    "-tool_select",
    type=str,
    default="domain",
    help="tool selection mode",
    choices=["domain", "all", "fixed", "retrieval"],
)
parser.add_argument(
    "-method",
    type=str,
    default="react",
    help="problem solving method, agentic methods.",
    choices=["react", "speculative"],
)
parser.add_argument(
    "-k",
    type=int,
    default=20,
    help="tool pool size for fixed pool and top_k for retrieval",
)
# retrieval settings (only used when tool_select is 'retrieval')
parser.add_argument(
    "-emb_model",
    type=str,
    default="ToolBench_IR",
    help="embedding model for retrieval, necessary for retrieval",
    choices=["ToolBench_IR", "bge-large", "all-MiniLM"],
)
parser.add_argument(
    "-retrieve_mode",
    type=str,
    default="static",
    help="retrieve stage: 'static' for static tools, 'dynamic' for dynamic retrieval",
    choices=["static", "dynamic"],
)
parser.add_argument(
    "-retrieve_pool",
    type=str,
    default="domain",
    help="retrieve pool, all means retrieve all tools, subtool means retrieve subtools",
    choices=["all", "domain"],
)
# trajectory and file settings
parser.add_argument(
    "-traj_type",
    type=str,
    default="parallel",
    help="trajectory type",
    choices=["parallel", "sequential"],
)
parser.add_argument(
    "-traj_file", type=str, default="simple_ver", help="trajectory file"
)
# log setting
parser.add_argument(
    "-log_dir", type=str, default="./log/simple_query/model", help="log directory"
)
parser.add_argument(
    "-chk_dir",
    type=str,
    default="./chk/simple_query/model",
    help="checkpoint directory",
)
parser.add_argument(
    "-base_data_dir",
    type=str,
    default="../public_data",
    help="base data directory",
)
parser.add_argument(
    "-sleep_interval",
    type=float,
    default=0.5,
    help="sleep interval between API calls (seconds)",
)
parser.add_argument(
    "-max_retries",
    type=int,
    default=1,
    help="maximum number of retry attempts for failed queries",
)
parser.add_argument(
    "-reset", action="store_true", help="reset checkpoint and start from beginning"
)

args = parser.parse_args()


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

# Create directories if they don't exist
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.chk_dir, exist_ok=True)

#### data loading
# domain list
TARGET_DIRECTORY = Path(os.path.join(args.base_data_dir, args.traj_type))
domain_list = sorted([p.name for p in TARGET_DIRECTORY.iterdir() if p.is_dir()])
num_domain = len(domain_list)

# all tools
with open(os.path.join(args.base_data_dir, "tools/all_tools.json"), "r") as f:
    all_tool = json.load(f)


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
                    tool_spec["executed_output"] = result
                    return {
                        "type": "tool",
                        "tool_spec": tool_spec,
                        "observation": f"Tool '{tool_name}' executed successfully. Result: {result}",
                    }
                except Exception as exe_error:
                    tool_spec["executed_output"] = f"ERROR: {str(exe_error)}"
                    return {
                        "type": "tool",
                        "tool_spec": tool_spec,
                        "observation": f"Tool '{tool_name}' execution failed: {str(exe_error)}",
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


def run_react_conversation_dynamic(
    query, tool_list, model, max_turns=8, retriever_model=None, domain_embeddings=None
):
    """
    Run a complete ReAct conversation with dynamic tool retrieval.

    Workflow:
    1. Initial Query & Tool Retrieval: Retrieve relevant tools based on user query
    2. ReAct Cycle: Generate thought -> action -> observation
    3. Iterative Re-Retrieval: Use current thought context to retrieve new tools for next step
    4. Repeat until completion

    Args:
        query (str): The user query to solve
        tool_list (list): Full list of available tools (tool database)
        model (str): Model name for generation
        max_turns (int): Maximum number of ReAct turns (default 8)
        retriever_model: Retriever model for dynamic tool retrieval
        domain_embeddings: Domain embeddings for dynamic tool retrieval

    Returns:
        dict: Result containing conversation history, final answer, and call counts
    """
    # Validate required parameters
    if retriever_model is None or domain_embeddings is None:
        raise ValueError(
            "retriever_model and domain_embeddings are required for dynamic retrieval"
        )

    # Initialize prompt with ReAct instruction template
    query_text = sanitize_input(query)
    # Initial retrieval
    retrieved_history = []
    try:
        retrieved_tools = retrieve_tools(
            retriever_model, query_text, tool_list, domain_embeddings, top_k=args.k
        )
    except Exception as e:
        retrieved_tools = tool_list[: args.k]

    retrieved_history.append(retrieved_tools)
    tools_text = sanitize_input(str(retrieved_tools))
    conversation_history = f"Query: {query_text}\n"

    logger.info(f"Query: {query_text}")

    n_calls, n_badcalls = 0, 0
    final_answer = ""
    called_tools = []  # Track tools called during conversation
    done = False
    obs = ""

    # Main ReAct loop (exactly like original webthink)
    for i in range(1, max_turns + 1):

        n_calls += 1

        # Step 1: THINK - Generate the thought first
        # The model reflects on the conversation history to decide what to do next.
        thought_retrieval_prompt = (
            react_instruction
            + react_examples
            + f"Available tools:\n {tools_text}\n"
            + conversation_history
            + f"Thought {i}:"
        )

        try:
            thought_retrieval = generate_content_with_retry(
                model, thought_retrieval_prompt, stop=[f"\nAction {i}:"]
            ).strip()
            logger.info(f"thought_retrieval_{i}: {thought_retrieval}")
        except Exception as e:
            raise

        conversation_history += f"Thought {i}: {thought_retrieval}\n"
        # Step 2: RETRIEVE - Use the new thought to get relevant tools
        try:
            retrieved_tools = retrieve_tools(
                retriever_model,
                thought_retrieval,  # Use the latest thought for retrieval
                tool_list,
                domain_embeddings,
                top_k=args.k,
            )

        except Exception as retrieval_error:
            logger.warning(f"Tool retrieval failed: {str(retrieval_error)}")
            retrieved_tools = tool_list[: args.k]
        retrieved_history.append(retrieved_tools)
        conversation_history += f"Tool list is updated for subsequent thought!"

        # Step 3: ACT - Generate the action using the newly retrieved tools
        tools_text = sanitize_input(str(retrieved_tools))
        thought_action_prompt = (
            react_instruction
            + react_examples
            + f"Available tools:\n {tools_text}\n"
            + conversation_history
            + f"Thought {i}:"
        )

        try:
            # Validate stop sequence to prevent blank stop sequence error
            stop_sequence = f"\nObservation {i}:"
            if not stop_sequence.strip():
                stop_sequence = "\nObservation:"
            thought_action = generate_content_with_retry(
                model, thought_action_prompt, stop=[stop_sequence]
            ).strip()
        except Exception as e:
            raise

        # Check if this is a Finish action before trying to parse
        if "Finish[" in thought_action:
            # Extract the final answer from Finish[answer] format
            start_idx = thought_action.find("Finish[")
            end_idx = thought_action.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                final_answer = thought_action[
                    start_idx + 7 : end_idx
                ]  # Remove "Finish[" and "]"
                logger.info(f"final_answer_{i}: {final_answer}")
                # Build the final step string
                conversation_history += f"Thought {i}: {thought_action.strip()}\nAction {i}: Finish\nObservation {i}: Task completed. Final answer: {final_answer}\n"
                # Set done and break
                done = True
                break
            else:
                pass

        try:
            # Parse thought and action (like original)
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
            # print thought/action
            logger.info(f"thought_action_{i}: {thought}")
            logger.info(f"action_{i}: {action}")
        except:
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            # Use safer non-blank stop sequences to avoid API ValidationException
            # Avoid whitespace-only entries (e.g., "\n") which some providers reject as blank
            recovery_stop = ["\nObservation", "Observation", "\nAction", "\nThought"]
            action_prompt = (
            react_instruction
            + react_examples
            + f"Available tools:\n {tools_text}\n"
            + conversation_history
            + f"Thought {i}: "
            )
            action = generate_content_with_retry(
                model,
                action_prompt + f"{thought}\nAction {i}:",
                stop=recovery_stop
                ).strip()
            logger.info(f"thought_action_{i}: {thought}")
            logger.info(f"action_{i}: {action}")

        # Step 4: OBSERVE - Execute the action
        try:
            obs_result = execute_action(action)
        except Exception as e:
            raise

        # Handle observation result
        if isinstance(obs_result, dict) and obs_result.get("type") == "finish":
            final_answer = obs_result.get("answer", "")
            obs = f"Task completed. Final answer: {final_answer}"
            done = True
        elif isinstance(obs_result, dict) and obs_result.get("type") == "tool":
            # Tool was called - collect it and use observation
            tool_spec = obs_result.get("tool_spec", {})
            called_tools.append(tool_spec)
            obs = obs_result.get("observation", "").replace(
                "\\n", "")  # Clean obs like original
            done = False
        else:
            obs = str(obs_result).replace("\\n", "")  # Clean obs like original
            done = False

        logger.info(f"observation_{i}: {obs}")
        conversation_history += f"Observation {i}: {obs}\n"
        if done:
            break

    # Force completion if not done (like original)
    if not done:
        obs_result = execute_action("Finish[]")
        if isinstance(obs_result, dict) and obs_result.get("type") == "finish":
            final_answer = obs_result.get("answer", "")

    # Return info similar to original webthink
    return {
        "conversation_history": conversation_history,
        "final_answer": final_answer,
        "called_tools": called_tools,  # List of tool_spec dicts with exe_result
        "n_calls": n_calls,
        "n_badcalls": n_badcalls,
        "retrieve_mode": "dynamic",
        "retrieved_tools": retrieved_history,
    }


def run_react_conversation(query, tool_list, model, max_turns=8):
    """
    Run a complete ReAct conversation following the efficient original webthink pattern.

    Args:
        query (str): The user query to solve
        tool_list (list): List of available tools
        model (str): Model name for generation
        max_turns (int): Maximum number of ReAct turns (default 8 like original)

    Returns:
        dict: Result containing conversation history, final answer, and call counts
    """

    # Initialize prompt with ReAct instruction template (like original webthink_prompt)
    query_text = sanitize_input(query)
    tools_text = sanitize_input(str(tool_list))
    prompt = (
        react_instruction
        + react_examples
        + f"Available tools:\n {tools_text}\n"
        + f"Query: {query_text}\n"
    )

    print(f"Query: {query_text}")

    n_calls, n_badcalls = 0, 0
    final_answer = ""
    called_tools = []  # Track tools called during conversation
    done = False
    obs = ""
    # Main ReAct loop (exactly like original webthink)
    for i in range(1, max_turns + 1):

        n_calls += 1

        # Generate Thought and Action with stop token (like original)
        try:
            # Validate stop sequence to prevent blank stop sequence error
            stop_sequence = f"\nObservation {i}:"
            if not stop_sequence.strip():
                stop_sequence = "\nObservation:"
            thought_action = generate_content_with_retry(
                model, prompt + f"Thought {i}:", stop=[stop_sequence]
            )
        except Exception as e:
            raise

        # Check if this is a Finish action before trying to parse
        if "Finish[" in thought_action:

            # Extract the final answer from Finish[answer] format
            start_idx = thought_action.find("Finish[")
            end_idx = thought_action.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                final_answer = thought_action[
                    start_idx + 7 : end_idx
                ]  # Remove "Finish[" and "]"
                # Build the final step string
                step_str = f"Thought {i}: {thought_action.strip()}\nAction {i}: Finish\nObservation {i}: Task completed. Final answer: {final_answer}\n"
                prompt += step_str

                # Set done and break
                done = True
                break
            else:
                pass

        try:
            # Parse thought and action (like original)
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
            print(f"thought {i}:{thought}")
            print(f"action {i}:{action}")
        except:
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            # Use safer non-blank stop sequences to avoid API ValidationException
            # Avoid whitespace-only entries (e.g., "\n") which some providers reject as blank
            recovery_stop = ["\nObservation", "Observation", "\nAction", "\nThought"]
            action = generate_content_with_retry(
                model,
                prompt + f"Thought {i}: {thought}\nAction {i}:",
                stop=recovery_stop,
            ).strip()
        # Execute action and get observation (like original step function)
        try:
            obs_result = execute_action(action)
        except Exception as e:
            raise

        # Handle observation result
        if isinstance(obs_result, dict) and obs_result.get("type") == "finish":
            final_answer = obs_result.get("answer", "")
            obs = f"Task completed. Final answer: {final_answer}"
            done = True

        elif isinstance(obs_result, dict) and obs_result.get("type") == "tool":
            # Tool was called - collect it and use observation
            tool_spec = obs_result.get("tool_spec", {})
            called_tools.append(tool_spec)
            obs = obs_result.get("observation", "").replace(
                "\\n", ""
            )  # Clean obs like original
            done = False
        else:
            obs = str(obs_result).replace("\\n", "")  # Clean obs like original
            done = False

        # Build step string (exactly like original)
        step_str = (
            f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        )
        prompt += step_str
        if done:
            break

    # Force completion if not done (like original)
    if not done:

        obs_result = execute_action("Finish[]")
        if isinstance(obs_result, dict) and obs_result.get("type") == "finish":
            final_answer = obs_result.get("answer", "")

    # Return info similar to original webthink
    return {
        "conversation_history": prompt,
        "final_answer": final_answer,
        "called_tools": called_tools,  # List of tool_spec dicts with exe_result
        "n_calls": n_calls,
        "n_badcalls": n_badcalls,
    }


# Global checkpoint file
if args.tool_select == "fixed":
    global_chk_file = os.path.join(
        args.chk_dir,
        f"global_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{str(args.k)}.json",
    )
elif args.tool_select == "retrieval":
    global_chk_file = os.path.join(
        args.chk_dir,
        f"global_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{args.emb_model}_{args.retrieve_mode}_{args.retrieve_pool}_{str(args.k)}.json",
    )
else:
    global_chk_file = os.path.join(
        args.chk_dir,
        f"global_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}.json",
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
        if domain == "Finance":
            if args.tool_select not in ["fixed", "retrieval"]:
                logger.info(f"Skipping domain {domain} due to out-of-context issue!")
                continue
        # Remove log files
        if args.tool_select == "fixed":
            log_file = os.path.join(
                args.log_dir,
                f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{str(args.k)}.json",
            )
        elif args.tool_select == "retrieval":
            log_file = os.path.join(
                args.log_dir,
                f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{args.emb_model}_{args.retrieve_mode}_{args.retrieve_pool}_{str(args.k)}.json",
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
                f"{domain}_{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{args.emb_model}_{args.retrieve_mode}_{args.retrieve_pool}_{str(args.k)}.json",
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
react_instruction = prompts["react_instruction"]
react_examples = prompts["react_examples"]

for domain in domain_list:
    logger.info(f"Evaluating domain: {domain}")

    # Determine log file path based on tool_select mode
    if args.tool_select == "fixed":
        log_file = os.path.join(
            args.log_dir,
            f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{str(args.k)}.json",
        )
    elif args.tool_select == "retrieval":
        log_file = os.path.join(
            args.log_dir,
            f"{domain}/{args.traj_type}_{args.traj_file}_{args.method}_{args.model}_{args.tool_select}_{args.emb_model}_{args.retrieve_mode}_{args.retrieve_pool}_{str(args.k)}.json",
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

    # Initialize retriever variables
    retriever_model = None
    domain_embeddings = None
    # Initialize tool_list based on tool_select mode
    if args.tool_select == "domain":
        tool_list = domain_tool
    elif args.tool_select == "all":
        tool_list = all_tool
    elif args.tool_select == "retrieval":
        # Load retriever model and embeddings for this domain
        logger.info(f"Loading retriever model: {args.emb_model}")
        logger.info(f"Retrieve mode: {args.retrieve_mode}")
        logger.info(f"Retrieve pool: {args.retrieve_pool}")
        retriever_model = load_retriever_model(args.emb_model)
        logger.info(f"Loading embeddings for domain: {domain}")
        if args.retrieve_pool == "domain":
            domain_tool, domain_embeddings = load_encoded_tools(
                domain_tool, domain, args.emb_model
            )
        elif args.retrieve_pool == "all":
            # For "all" pool, we need to load all tools
            domain_tool, domain_embeddings = load_encoded_tools(
                domain_tool, "All", args.emb_model
            )

    # evaluate
    logger.info(
        f"Processing {len(test_query) - chk_index} queries starting from index {chk_index}"
    )

    for i in range(chk_index, len(test_query)):
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
            # Retrieve tools for this specific query
            if args.retrieve_mode == "static":
                query_text = test_query[i]["query"]
                retrieved_tools = retrieve_tools(
                    retriever_model,
                    query_text,
                    domain_tool,
                    domain_embeddings,
                    top_k=args.k,
                )
                tool_list = retrieved_tools
                logger.info(
                    f"Static retrieval: Retrieved {len(tool_list)} tools for query: '{query_text[:50]}...'"
                )
            elif args.retrieve_mode == "dynamic":
                tool_list = domain_tool

                # Preprocess tool list to include only essential fields
        essential_fields = [
            "parent tool name",
            "required_parameters",
            "optional_parameters",
            "tool name",
            "tool description",
            "API name",
            "domain name",
        ]

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

        # Helper function to extract partial results from conversation history
        def extract_partial_results(conversation_text, error_msg):
            """Extract partial results from conversation history when ReAct fails."""
            try:
                if conversation_text:
                    # Look for any tool calls in the conversation history
                    import re

                    tool_calls = re.findall(r"Tool\[.*?\]", conversation_text)
                    extracted_tools = []
                    for tool_call in tool_calls:
                        try:
                            tool_spec = extract_json_from_markdown_fence(tool_call)
                            if tool_spec:
                                extracted_tools.append(tool_spec)
                        except:
                            # If we can't parse the tool call, create a basic entry
                            extracted_tools.append({"raw_tool_call": tool_call})

                    # Try to extract any final answer attempt
                    if "Final answer:" in conversation_text:
                        extracted_answer = conversation_text.split("Final answer:")[
                            -1
                        ].strip()
                    else:
                        extracted_answer = f"FAILED: {error_msg}"

                    extracted_calls = len(tool_calls) if tool_calls else 0
                else:
                    # No conversation history available
                    extracted_answer = f"FAILED: {error_msg}"
                    extracted_tools = []
                    extracted_calls = 0

                return extracted_tools, extracted_answer, extracted_calls
            except Exception as extract_error:

                return [], f"FAILED: {error_msg}", 0

        # Initialize variables for tracking ReAct progress
        conversation = ""
        called_tools = []
        n_calls = 0
        final_answer = ""

        while retry_count < args.max_retries and not success:

            try:
                logger.info(
                    f"Processing query {i+1}/{len(test_query)} (index {i}) - attempt {retry_count + 1}"
                )

                # Validate required fields exist
                if "query" not in test_query[i]:
                    raise KeyError(f"Missing query field: query")
                if "tool list" not in test_query[i]:
                    raise KeyError("Missing 'tool list' field in test data")

                # Run ReAct conversation using the wrapped function
                query_text = test_query[i]["query"]

                try:
                    # Use the wrapped ReAct function
                    # Pass retrieval parameters if using "multiple" strategy
                    if (
                        args.retrieve_mode == "dynamic"
                        and (retriever_model is not None)
                        and (domain_embeddings is not None)
                    ):
                        react_result = run_react_conversation_dynamic(
                            query_text,
                            processed_tool_list,
                            args.model,
                            max_turns=8,
                            retriever_model=retriever_model,
                            domain_embeddings=domain_embeddings,
                        )
                    else:
                        react_result = run_react_conversation(
                            query_text, processed_tool_list, args.model, max_turns=8
                        )

                    # Extract results

                    conversation = react_result["conversation_history"]
                    n_calls = react_result["n_calls"]
                    final_answer = react_result["final_answer"]
                    called_tools = react_result["called_tools"]

                except Exception as parse_error:

                    logger.warning(
                        f"ReAct conversation failed for query {i}, attempt {retry_count + 1}: {str(parse_error)}"
                    )
                    if retry_count == args.max_retries - 1:
                        # Extract partial results using the helper function
                        called_tools, final_answer, n_calls = extract_partial_results(
                            conversation, str(parse_error)
                        )
                        logger.error(
                            f"Final ReAct conversation failed for query {i}, but partial results saved."
                        )

                    else:
                        raise ValueError(str(parse_error))

                results = {
                    "query": test_query[i]["query"],
                    "gt answer": test_query[i].get("final_answer", ""),
                    "final_answer": final_answer,
                    "pred tool list": called_tools,  # List of tool_spec dicts with exe_result
                    "gt tool list": test_query[i]["tool list"],
                    "trajectory_type": test_query[i].get("trajectory_type", "unknown"),
                    "task_name": test_query[i].get("task_name", "unknown"),
                    "task_description": test_query[i].get("task_description", ""),
                }

                # Add error information if this was a failed attempt
                if retry_count == args.max_retries - 1 and "FAILED:" in final_answer:
                    results["error"] = final_answer
                    results["partial_success"] = True
                    results["tools_executed"] = len(called_tools) if called_tools else 0

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
                    if args.retrieve_mode == "static":
                        results["retrieved_tools"] = tool_list
                        results["retrieval_rate"] = retrieval_rate(
                            results["gt tool list"], results["retrieved_tools"]
                        )
                    elif args.retrieve_mode == "dynamic":
                        _rr = locals().get("react_result")
                        if isinstance(_rr, dict) and "retrieved_tools" in _rr:
                            results["retrieved_tools"] = _rr["retrieved_tools"]
                        else:
                            # Fallback to the processed tool list used for prompting
                            results["retrieved_tools"] = processed_tool_list
                records.append(results)
                success = True

                # Update global checkpoint
                global_checkpoint[domain] = i + 1

                # Save both checkpoint and log file after each query for interrupt/resume safety
                save_checkpoint_json(global_checkpoint, global_chk_file)
                try:
                    with open(log_file, "w") as f:
                        json.dump(records, f, indent=4)
                except Exception as save_error:
                    logger.error(
                        f"Failed to save log file after query {i+1}: {str(save_error)}"
                    )

                logger.info(f"Successfully processed query {i+1}/{len(test_query)}")

                # Exit after first query for testing (disabled)
                # exit(0)

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
                    # Extract partial results if available
                    if conversation:
                        called_tools, final_answer, n_calls = extract_partial_results(
                            conversation, f"API_ERROR: {str(e)}"
                        )
                    else:
                        called_tools, final_answer, n_calls = (
                            [],
                            f"FAILED: API_ERROR: {str(e)}",
                            0,
                        )

                    # Add failed result to maintain index consistency
                    results = {
                        "query": test_query[i].get("query", "QUERY_ERROR"),
                        "gt answer": test_query[i].get("final_answer", ""),
                        "final_answer": final_answer,
                        "pred tool list": called_tools,
                        "gt tool list": test_query[i].get("tool list", []),
                        "error": f"API_ERROR: {str(e)}",
                        "trajectory_type": test_query[i].get(
                            "trajectory_type", "unknown"
                        ),
                        "task_name": test_query[i].get("task_name", "unknown"),
                        "task_description": test_query[i].get("task_description", ""),
                        "traj_exact_match": False,
                        "traj_inclusion": 0.0,
                        "tool_traj_usage": [],
                        "partial_success": len(called_tools) > 0,
                        "tools_executed": len(called_tools),
                    }
                    records.append(results)
                    success = True  # Mark as "success" to continue processing

                    # Update global checkpoint and save both files
                    global_checkpoint[domain] = i + 1
                    save_checkpoint_json(global_checkpoint, global_chk_file)
                    try:
                        with open(log_file, "w") as f:
                            json.dump(records, f, indent=4)
                    except Exception as save_error:
                        logger.error(
                            f"Failed to save log file after error in query {i+1}: {str(save_error)}"
                        )

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
                    # Extract partial results if available
                    if conversation:
                        called_tools, final_answer, n_calls = extract_partial_results(
                            conversation, f"PARSING_ERROR: {str(e)}"
                        )
                    else:
                        called_tools, final_answer, n_calls = (
                            [],
                            f"FAILED: PARSING_ERROR: {str(e)}",
                            0,
                        )

                    # Add failed result to maintain index consistency
                    results = {
                        "query": test_query[i].get("query", "QUERY_ERROR"),
                        "gt answer": test_query[i].get("final_answer", ""),
                        "final_answer": final_answer,
                        "pred tool list": called_tools,
                        "gt tool list": test_query[i].get("tool list", []),
                        "error": f"PARSING_ERROR: {str(e)}",
                        "trajectory_type": test_query[i].get(
                            "trajectory_type", "unknown"
                        ),
                        "task_name": test_query[i].get("task_name", "unknown"),
                        "task_description": test_query[i].get("task_description", ""),
                        "traj_exact_match": False,
                        "traj_inclusion": 0.0,
                        "tool_traj_usage": [],
                        "partial_success": len(called_tools) > 0,
                        "tools_executed": len(called_tools),
                    }
                    records.append(results)
                    success = True  # Mark as "success" to continue processing

                    # Update global checkpoint and save both files
                    global_checkpoint[domain] = i + 1
                    save_checkpoint_json(global_checkpoint, global_chk_file)
                    try:
                        with open(log_file, "w") as f:
                            json.dump(records, f, indent=4)
                    except Exception as save_error:
                        logger.error(
                            f"Failed to save log file after error in query {i+1}: {str(save_error)}"
                        )

            except Exception as e:
                logger.critical(f"Unexpected error processing query {i}: {str(e)}")

                # Extract partial results if available before re-raising
                if conversation:
                    called_tools, final_answer, n_calls = extract_partial_results(
                        conversation, f"UNEXPECTED_ERROR: {str(e)}"
                    )

                    # Create a result record with partial information
                    results = {
                        "query": test_query[i].get("query", "QUERY_ERROR"),
                        "gt answer": test_query[i].get("final_answer", ""),
                        "final_answer": final_answer,
                        "pred tool list": called_tools,
                        "gt tool list": test_query[i].get("tool list", []),
                        "error": f"UNEXPECTED_ERROR: {str(e)}",
                        "trajectory_type": test_query[i].get(
                            "trajectory_type", "unknown"
                        ),
                        "task_name": test_query[i].get("task_name", "unknown"),
                        "task_description": test_query[i].get("task_description", ""),
                        "traj_exact_match": False,
                        "traj_inclusion": 0.0,
                        "tool_traj_usage": [],
                        "partial_success": len(called_tools) > 0,
                        "tools_executed": len(called_tools),
                    }
                    records.append(results)

                # Save checkpoint and results before re-raising
                global_checkpoint[domain] = i
                save_checkpoint_json(global_checkpoint, global_chk_file)
                with open(log_file, "w") as f:
                    json.dump(records, f, indent=4)
                raise

        # Configurable sleep interval between queries
        time.sleep(args.sleep_interval)

    # Mark domain as completed and save final state
    try:
        logger.info(f"Completed processing all queries for domain {domain}")

        # Final checkpoint save (though it should already be up-to-date)
        global_checkpoint[domain] = len(test_query)
        save_checkpoint_json(global_checkpoint, global_chk_file)

        # Final log file save (though it should already be up-to-date)
        with open(log_file, "w") as f:
            json.dump(records, f, indent=4)

    except Exception as final_save_error:
        logger.error(
            f"Failed to save final state for domain {domain}: {str(final_save_error)}"
        )
