import os
import json
import time
import logging
import argparse
import random
import sys
import signal
import shutil
from typing import Dict, Any

# Import basic model functions from centralized providers
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_with_retry,
    extract_json_from_markdown_fence
)

parser = argparse.ArgumentParser('Fix non-executable trajectories across domains')
parser.add_argument('-base_dir', type=str, default='/home/ec2-user/mountS3/newToolData/simple_query', help='base directory containing domain subdirectories')
parser.add_argument('-traj_file', type=str, default='simple_traj_parallel_consis_gemini-2.5-pro_v2', help='trajectory filename without extension')
parser.add_argument('-model', type=str, default='claude_v37', help='model name', choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
parser.add_argument('-max_retries', type=int, default=3, help='maximum retry attempts per request')
parser.add_argument('-retry_delay', type=float, default=1.0, help='initial retry delay in seconds')
parser.add_argument('-dry_run', action='store_true', help='show what would be fixed without making changes')
parser.add_argument('-checkpoint_dir', type=str, default='./checkpoints', help='directory to store checkpoint files')
parser.add_argument('-reset', action='store_true', help='reset checkpoints and start from beginning')
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state for graceful shutdown
shutdown_requested = False

def signal_handler(signum, _):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Create checkpoint directory
os.makedirs(args.checkpoint_dir, exist_ok=True)

def get_checkpoint_file(domain):
    """Get checkpoint file path for a domain"""
    return os.path.join(args.checkpoint_dir, f"traj_query_check_{domain}_{args.traj_file}.json")

def load_checkpoint(domain):
    """Load checkpoint for a domain"""
    checkpoint_file = get_checkpoint_file(domain)
    
    # If reset flag is set, delete existing checkpoint
    if args.reset and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info(f"Reset: Deleted checkpoint for {domain}")
        return {"record_index": 0, "total_processed": 0}
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint for {domain}: record {checkpoint.get('record_index', 0)}")
        return checkpoint
    except FileNotFoundError:
        logger.info(f"No checkpoint found for {domain}, starting from beginning")
        return {"record_index": 0, "total_processed": 0}
    except Exception as e:
        logger.warning(f"Error loading checkpoint for {domain}: {e}, starting fresh")
        return {"record_index": 0, "total_processed": 0}

def validate_json_response(response_dict: Dict[str, Any], record_idx: int, query_type: str) -> bool:
    """Validate that JSON response has required fields"""
    required_fields = ["judgement", "reason", "refined query"]
    
    for field in required_fields:
        if field not in response_dict:
            logger.error(f"Record {record_idx}: Missing '{field}' in {query_type} response")
            return False
    
    # Validate judgement is boolean
    if not isinstance(response_dict["judgement"], bool):
        logger.error(f"Record {record_idx}: 'judgement' must be boolean in {query_type} response")
        return False
    
    # Validate reason and refined query are strings
    if not isinstance(response_dict["reason"], str) or not isinstance(response_dict["refined query"], str):
        logger.error(f"Record {record_idx}: 'reason' and 'refined query' must be strings in {query_type} response")
        return False
    
    return True

def save_checkpoint(domain, record_index, total_processed):
    """Save checkpoint for a domain"""
    checkpoint_file = get_checkpoint_file(domain)
    checkpoint = {
        "record_index": record_index,
        "total_processed": total_processed,
        "timestamp": time.time(),
        "domain": domain,
        "traj_file": args.traj_file,
        "model": args.model
    }
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.debug(f"Saved checkpoint for {domain}: record {record_index}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint for {domain}: {e}")



check_prompt = """
You will be given a tool-calling trajectory, query and requirements. You need to determine if the query satisfies the requirements, and refine it accordingly. 

Tool-calling trajectory:
<tool list>

Query:
<query>

Requirements:
<requirement>

Please respond in the following json format:
```json
{
"judgement": [True/False],
"reason": [Briefly explain your decision.],
"refined query": [If judgement is True just repeat the query; if judgement is False refine the query and put it here.]
}
```

NOTICE: the query must align with the trajectory; please avoid too much openness and vagueness in the query.
"""

# Validate and load configuration files
try:
    with open('/home/ec2-user/mountS3/newToolData/selected_category.json', 'r') as f:
        select_cate = json.load(f)
    domain_list = list(select_cate.keys())[1:]
    num_domain = len(domain_list)
    
    if not domain_list:
        logger.error("No domains found in selected_category.json")
        sys.exit(1)
    
    with open('/home/ec2-user/mountS3/newToolData/query_gen/query_gen_prompt.json', 'r') as f:
        gen_prompt = json.load(f)
    
    # Validate that required prompts exist
    if 'simple' not in gen_prompt:
        logger.error("Missing 'simple' prompt in query_gen_prompt.json")
        sys.exit(1)
    
    # Validate domain-specific hard prompts exist
    missing_prompts = []
    for domain in domain_list:
        if f'{domain}_hard' not in gen_prompt:
            missing_prompts.append(f'{domain}_hard')
    
    if missing_prompts:
        logger.error(f"Missing hard prompts for domains: {missing_prompts}")
        sys.exit(1)
    
    logger.info(f"Configuration validated: {num_domain} domains found")
    
except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {e}")
    sys.exit(1)
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in configuration file: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    sys.exit(1)

total_domains_processed = 0
total_records_processed = 0

# Main processing loop with shutdown checks
for domain_idx, domain in enumerate(domain_list):
    if shutdown_requested:
        logger.info("Shutdown requested, stopping domain processing")
        break
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing domain: {domain}")
    logger.info(f"{'='*60}")
    try:
        requirement_simple = gen_prompt['simple']
        requirement_hard = gen_prompt[f'{domain}_hard']
    except KeyError as e:
        logger.error(f"Missing requirement for domain {domain}: {e}")
        continue
    
    # Load checkpoint for this domain
    checkpoint = load_checkpoint(domain)
    
    # Construct trajectory file path
    trajectory_file = os.path.join(args.base_dir, domain, f"{args.traj_file}.json")
    
    
    # Check if trajectory file exists
    if not os.path.exists(trajectory_file):
        logger.warning(f"Trajectory file not found: {trajectory_file}")
        continue


    # Load trajectory data
    try:
        with open(trajectory_file, 'r') as f:
            trajectories = json.load(f)
    except Exception as e:
        logger.error(f"Error loading {trajectory_file}: {e}")
        continue
    
    if not isinstance(trajectories, list):
        logger.error(f"Expected list in {trajectory_file}, got {type(trajectories)}")
        continue

    logger.info(f"Total records in {domain}: {len(trajectories)}")
    logger.info(f"Resuming from record: {checkpoint['record_index']}")
    
    domain_processed_count = 0
    
    # Process records starting from checkpoint
    for i in range(checkpoint['record_index'], len(trajectories)):
        if shutdown_requested:
            logger.info(f"Shutdown requested, stopping at record {i+1} in domain {domain}")
            break
            
        record = trajectories[i]
        
        try:
            # Check if record has the required field
            if 'query_simple_claude_v37' not in record:
                logger.warning(f"Record {i+1} missing 'query_simple_claude_v37' field - skipping")
                continue
            
            if 'query_hard_claude_v37' not in record:
                logger.warning(f"Record {i+1} missing 'query_hard_claude_v37' field - skipping")
                continue
            
            if 'tool list' not in record:
                logger.warning(f"Record {i+1} missing 'tool list' field - skipping")
                continue
            
            simple_query = record['query_simple_claude_v37']
            hard_query = record['query_hard_claude_v37']
            tool_list = record['tool list']
            
            logger.info(f"Processing record {i+1}/{len(trajectories)} in {domain}")
            
            if args.dry_run:
                logger.info(f"DRY RUN: Would check query: {simple_query[:100]}...")
                domain_processed_count += 1
                continue
                
            # Perform the trajectory-query check
            check_input_simple = check_prompt.replace("<tool list>", str(tool_list)).replace("<query>", simple_query).replace("<requirement>", requirement_simple)
            check_input_hard = check_prompt.replace("<tool list>", str(tool_list)).replace("<query>", hard_query).replace("<requirement>", requirement_hard)
            try:
                response_simple = generate_content_with_retry(args.model, check_input_simple)
                response_simple_dict = extract_json_from_markdown_fence(response_simple, expect_dict=True)
                if not response_simple_dict:
                    logger.error(f"Failed to extract valid JSON from simple query response for record {i+1}")
                    continue
                
                # Validate simple response structure
                if not validate_json_response(response_simple_dict, i+1, "simple"):
                    continue
                    
                response_hard = generate_content_with_retry(args.model, check_input_hard)
                response_hard_dict = extract_json_from_markdown_fence(response_hard, expect_dict=True)
                if not response_hard_dict:
                    logger.error(f"Failed to extract valid JSON from hard query response for record {i+1}")
                    continue
                
                # Validate hard response structure
                if not validate_json_response(response_hard_dict, i+1, "hard"):
                    continue
            except Exception as model_error:
                logger.error(f"Error generating or parsing model response for record {i+1}: {model_error}")
                continue

            # update query file
            record["simple_refined"] = response_simple_dict["judgement"]
            record["refined_query_simple_claude_v37"] = response_simple_dict["refined query"]
            record["hard_refined"] = response_hard_dict["judgement"]
            record["refined_query_hard_claude_v37"] = response_hard_dict["refined query"]
            
            # Log refinement status
            simple_status = "needs refine" if not response_simple_dict["judgement"] else "does not need refine"
            hard_status = "needs refine" if not response_hard_dict["judgement"] else "does not need refine"
            logger.info(f"Record {i+1}: Simple query {simple_status}, Hard query {hard_status}")
                
            domain_processed_count += 1
            
            # Save checkpoint and trajectory file for each record
            try:
                with open(trajectory_file, 'w') as f:
                    json.dump(trajectories, f, indent=2)
                
                save_checkpoint(domain, i + 1, 
                              checkpoint['total_processed'] + domain_processed_count)
                
                logger.debug(f"Saved progress: {i+1}/{len(trajectories)} records")
            except Exception as save_error:
                logger.error(f"Error saving progress: {save_error}")
                    
        except Exception as e:
            logger.error(f"Error processing record {i+1} in {domain}: {e}")
            continue
    
    # Final save and cleanup for this domain
    if not args.dry_run and not shutdown_requested:
        try:
            with open(trajectory_file, 'w') as f:
                json.dump(trajectories, f, indent=2)
            
            # Update final checkpoint
            save_checkpoint(domain, len(trajectories),
                          checkpoint['total_processed'] + domain_processed_count)
                          
            logger.info(f"✓ Domain {domain} completed ({domain_processed_count} records processed)")
            
        except Exception as e:
            logger.error(f"Error in final save for {domain}: {e}")
    elif shutdown_requested:
        logger.info(f"Domain {domain} processing interrupted ({domain_processed_count} records processed)")
    
    # Update global totals
    total_domains_processed += 1
    total_records_processed += domain_processed_count
    
    logger.info(f"Domain {domain} summary: {domain_processed_count} processed")

# Final summary
logger.info(f"\n{'='*60}")
if shutdown_requested:
    logger.info("TRAJECTORY QUERY CHECK SUMMARY (INTERRUPTED)")
else:
    logger.info("TRAJECTORY QUERY CHECK SUMMARY")
logger.info(f"{'='*60}")
logger.info(f"Model: {args.model}")
logger.info(f"Trajectory file pattern: {args.traj_file}.json")
logger.info(f"Domains processed: {total_domains_processed}/{num_domain}")
logger.info(f"Total records processed: {total_records_processed}")
logger.info(f"Dry run: {args.dry_run}")
logger.info(f"Reset mode: {args.reset}")
if shutdown_requested:
    logger.info("Status: Interrupted by user/system signal")
else:
    logger.info("Status: Completed successfully")
logger.info(f"{'='*60}")