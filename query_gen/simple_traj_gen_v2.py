import os
import json
import time
import logging
import re
import argparse
import random
import sys

# Import basic model functions from centralized providers
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_providers import (
    generate_content_with_retry,
    ModelNotAvailableError,
    extract_json_from_markdown_fence
)

parser = argparse.ArgumentParser('evaluate tool usage')
parser.add_argument('-model', type=str, default='gemini-2.5-pro', help='model name', choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
parser.add_argument('-save_dir', type=str, default='/home/ec2-user/mountS3/newToolData/simple_query', help='file-saving directory')
parser.add_argument('-chk_dir', type=str, default='./chk/traj_gen', help='checkpoint directory')
parser.add_argument('-traj_type', type=str, default='parallel', help='trajectory type', choices=['parallel', 'sequential', 'mixed'])
parser.add_argument('-num_tools', type=int, default=None, help='number of tools to involve in generation (None for automatic)')
parser.add_argument('-min_tools', type=int, default=3, help='minimum number of tools required')
parser.add_argument('-max_tools', type=int, default=10, help='maximum number of tools allowed')
parser.add_argument('-num_query', type=int, default=5, help='maximum number of queries for one task type')
parser.add_argument('-queries_per_tool_count', type=int, default=None, help='number of queries to generate for each tool count (overrides num_query when used with min/max_tools)')
parser.add_argument('-max_retries', type=int, default=3, help='maximum retry attempts per request')
parser.add_argument('-retry_delay', type=float, default=1.0, help='initial retry delay in seconds')
parser.add_argument('-enable_checking', action='store_true', help='enable automatic checking and refinement')
parser.add_argument('-check_model', type=str, default=None, help='model to use for checking (defaults to main model)')
args = parser.parse_args()

# All generation functions and utilities are now centralized in model_providers.py

def validate_trajectory(trajectory_data, tool_list, traj_type, num_tools=None, min_tools=2, max_tools=5):
    """
    Validates a generated trajectory against requirements.
    
    Args:
        trajectory_data (dict): Generated trajectory with 'query' and 'tool list'
        tool_list (list): Available tools
        traj_type (str): Type of trajectory (parallel, sequential, mixed)
        num_tools (int): Exact number of tools required (None for range)
        min_tools (int): Minimum number of tools
        max_tools (int): Maximum number of tools
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    if 'query' not in trajectory_data or 'tool list' not in trajectory_data:
        errors.append("Missing required fields: 'query' or 'tool list'")
        return False, errors
    
    tools_used = trajectory_data['tool list']
    num_tools_used = len(tools_used)
    
    # Check tool count constraints
    if num_tools is not None and num_tools_used != num_tools:
        errors.append(f"Expected exactly {num_tools} tools, got {num_tools_used}")
    elif num_tools_used < min_tools:
        errors.append(f"Too few tools: {num_tools_used} < {min_tools}")
    elif num_tools_used > max_tools:
        errors.append(f"Too many tools: {num_tools_used} > {max_tools}")
    
    # Check if tools exist in tool list
    available_tool_names = {tool['tool name'] for tool in tool_list}
    for tool in tools_used:
        if 'tool name' not in tool:
            errors.append("Tool missing 'tool name' field")
            continue
        if tool['tool name'] not in available_tool_names:
            errors.append(f"Tool '{tool['tool name']}' not found in available tools")
    
    # Validate trajectory type constraints
    if traj_type == 'sequential' and num_tools_used > 1:
        # Check for dependency markers in sequential trajectories
        found_dependencies = False
        for tool in tools_used[1:]:  # Skip first tool
            for param in tool.get('required parameters', []) + tool.get('optional parameters', []):
                if '{{' in str(param.get('value', '')) and '}}' in str(param.get('value', '')):
                    found_dependencies = True
                    break
        if not found_dependencies:
            errors.append("Sequential trajectory should have dependency markers like {{output from step 1}}")
    
    return len(errors) == 0, errors

def refine_trajectory(trajectory_data, validation_errors, model, tool_list, domain, task_name, task_description, traj_type, requirements_dict, num_tools=None, min_tools=2, max_tools=5):
    """
    Attempts to refine a trajectory based on validation errors.
    
    Returns:
        dict: Refined trajectory data
    """
    error_summary = "; ".join(validation_errors)
    
    # Get trajectory type specific requirements
    trajectory_requirements = requirements_dict.get(f'traj_gen_{traj_type}', '')
    
    refinement_prompt = f"""
The following trajectory has validation errors that need to be fixed:

Original Query: {trajectory_data.get('query', 'N/A')}
Original Tool List: {json.dumps(trajectory_data.get('tool list', []), indent=2)}

Validation Errors: {error_summary}

IMPORTANT - Follow these trajectory type requirements:
{trajectory_requirements}

Context:
- Domain: {domain}
- Task: {task_name}
- Task Description: {task_description}
- Trajectory Type: {traj_type}
- Available Tools: {json.dumps(tool_list, indent=2)}
- Tool Count: {"Exactly " + str(num_tools) if num_tools else f"Between {min_tools} and {max_tools}"}

Please fix the validation errors while following the trajectory type requirements above.

Your response should be in the following JSON format:

```json
{
  "query": "<corrected natural language query>",
  "tool list": [
    {
      "tool name": "<tool name>",
      "tool description": "<tool description>",
      "required parameters": [
        {"name": "<parameter name>", "value": "<parameter value>"},
        ...
      ],
      "optional parameters": [
        {"name": "<parameter name>", "value": "<parameter value>"},
        ...
      ]
    },
    ...
  ]
}
```
"""
    
    try:
        refined_response = generate_content_with_retry(model, refinement_prompt)
        return extract_json_from_markdown_fence(refined_response)
    except Exception as e:
        raise Exception(f"Failed to refine trajectory: {e}")

def generate_trajectory_with_constraints(model, domain, task_name, task_description, tool_list, traj_type, prompt_template, num_tools=None, min_tools=3, max_tools=10):
    """
    Generates a trajectory with specific tool count constraints.
    """
    # Sample tools if num_tools is specified and we have more tools than needed
    if num_tools and len(tool_list) > num_tools:
        sampled_tools = random.sample(tool_list, min(num_tools * 2, len(tool_list)))  # Sample more for variety
    else:
        sampled_tools = tool_list
    
    # Modify prompt to include tool count constraints
    constraint_text = ""
    if num_tools:
        constraint_text = f"Use exactly {num_tools} tools."
    else:
        constraint_text = f"Use between {min_tools} and {max_tools} tools."
    
    # Add trajectory type specific instructions
    if traj_type == 'mixed':
        constraint_text += " Mix both parallel and sequential tool calls where some tools can run in parallel while others depend on previous outputs."
    
    enhanced_prompt = prompt_template.replace(
        "Your task is to:",
        f"Your task is to:\n{constraint_text}\n\n"
    )
    
    enhanced_prompt = enhanced_prompt.replace("<domain>", domain)
    enhanced_prompt = enhanced_prompt.replace("<task_type>", task_name)
    enhanced_prompt = enhanced_prompt.replace("<task_description>", task_description)
    enhanced_prompt = enhanced_prompt.replace("<tool_list>", str(sampled_tools))
    
    response = generate_content_with_retry(model, enhanced_prompt)
    return extract_json_from_markdown_fence(response)

def load_checkpoint(chk_file): # chk file is a json file
    try:
        with open(chk_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # no checkpoint file, start from beginning

def save_checkpoint(index, chk_file):
    with open(chk_file, "w") as f:
        json.dump(index, f, indent=4)

#### data loading
# domain list
with open('/home/ec2-user/mountS3/newToolData/selected_category.json', 'r') as f:
    select_cate = json.load(f)
domain_list = list(select_cate.keys())[9:]
num_domain = len(domain_list)

# Create checkpoint directory if it doesn't exist
os.makedirs(args.chk_dir, exist_ok=True)

# main
info_prompt = """
You are given the following structured information:

Domain: <domain>
Task Type: <task_type>
Task Description: <task_description>

Tool List:
<tool_list>
"""
format_prompt = """
Your response should be in the following JSON format:

```json
{
"query": "<natural language query here>",
"tool list": [
{"tool name": "<tool name>",
"tool description": "<tool description>",
"required parameters\": [{"name": "<parameter name>", "value": "<parameter value>"},...],
"optional parameters": [{"name": "<parameter name>", "value": "<parameter value>"},...]
},
...]
}
```
"""
with open('/home/ec2-user/mountS3/newToolData/query_gen/traj_gen_prompt.json', 'r') as f:
    requirements = json.load(f)
# iterate over domains
for domain in domain_list:
    print(f"Generating trajectory for domain: {domain}")
    with open(f'/home/ec2-user/mountS3/newToolData/tools/{domain}/true_filter_subtool.json', 'r') as f:
        tool_list = json.load(f)
    with open(f'/home/ec2-user/mountS3/newToolData/simple_query/{domain}/task_type.json', 'r') as f:
        task_type = json.load(f)
    keys_to_keep = ['parent tool name', 'API name', 'domain name', 'tool name', 'tool description', 'required_parameters', 'optional_parameters']
    # load chk
    chk_file = os.path.join(args.chk_dir, f"simple_traj_{args.traj_type}_{args.model}_{domain}_v2.json")
    chk_index = load_checkpoint(chk_file)
    # load save files - create directory if it doesn't exist
    save_dir = os.path.join(args.save_dir, domain)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"simple_traj_{args.traj_type}_{args.model}_v2.json")
    
    # Always try to load existing records to avoid overwriting
    try:
        with open(save_file, 'r') as f:
            records = json.load(f)
        print(f"Loaded {len(records)} existing records for domain {domain}")
    except FileNotFoundError:
        records = []
        print(f"Starting with empty records for domain {domain}")
    # iterate over task types
    for i in range(len(task_type)):
        target_task = task_type[i]
        # target_tool_type = target_task['tool classes']
        # target_tools = [item for item in tool_list if item['category'] in target_tool_type]
        target_tools = tool_list
        
        # Determine generation strategy
        if args.num_tools is not None:
            # Generate for specific tool count
            tool_counts_to_generate = [args.num_tools]
            queries_per_count = args.num_query
        elif args.queries_per_tool_count is not None:
            # Generate k queries for each tool count from min to max
            tool_counts_to_generate = list(range(args.min_tools, args.max_tools + 1))
            queries_per_count = args.queries_per_tool_count
        else:
            # Default: generate using range constraints
            tool_counts_to_generate = [None]  # Use range validation
            queries_per_count = args.num_query
        
        # Generate for each tool count
        for tool_count in tool_counts_to_generate:
            print(f"Generating trajectories for task '{target_task['task name']}' with {tool_count if tool_count else f'{args.min_tools}-{args.max_tools}'} tools")
            
            # Create checkpoint key that includes tool count
            if tool_count is not None:
                chk_key = f"{target_task['task name']}_tools{tool_count}"
            else:
                chk_key = f"{target_task['task name']}_range"
            
            # initialize checkpoint
            if chk_key in chk_index:
                start_idx = chk_index[chk_key]
            else:
                start_idx = 0
                chk_index[chk_key] = 0
            
            # iteratively generate trajectory
            for j in range(start_idx, queries_per_count):
                retry_count = 0
                success = False
                
                while retry_count < args.max_retries and not success:
                    try:
                        # Choose trajectory type (handle mixed type)
                        current_traj_type = args.traj_type
                        if args.traj_type == 'mixed':
                            current_traj_type = random.choice(['parallel', 'sequential'])
                        
                        # Get appropriate prompt template
                        prompt_template = info_prompt + requirements[f'traj_gen_{current_traj_type}']+format_prompt
                        
                        # Generate trajectory with constraints
                        response_traj = generate_trajectory_with_constraints(
                            args.model, domain, target_task['task name'], 
                            target_task['task description'], target_tools, 
                            current_traj_type, prompt_template,
                            tool_count, args.min_tools, args.max_tools  # Use specific tool_count
                        )
                        
                        # Validate trajectory if checking is enabled
                        if args.enable_checking:
                            check_model = args.check_model or args.model
                            is_valid, validation_errors = validate_trajectory(
                                response_traj, target_tools, current_traj_type,
                                tool_count, args.min_tools, args.max_tools  # Use specific tool_count
                            )
                            
                            if not is_valid:
                                print(f"Validation failed for record {j}: {'; '.join(validation_errors)}")
                                # Attempt refinement
                                try:
                                    response_traj = refine_trajectory(
                                        response_traj, validation_errors, check_model,
                                        target_tools, domain, target_task['task name'],
                                        target_task['task description'], current_traj_type,
                                        requirements, tool_count, args.min_tools, args.max_tools  # Use specific tool_count
                                    )
                                    print(f"Successfully refined trajectory for record {j}")
                                    
                                    # Re-validate after refinement
                                    is_valid, validation_errors = validate_trajectory(
                                        response_traj, target_tools, current_traj_type,
                                        tool_count, args.min_tools, args.max_tools  # Use specific tool_count
                                    )
                                    if not is_valid:
                                        print(f"Refinement still has issues: {'; '.join(validation_errors)}")
                                except Exception as refinement_error:
                                    print(f"Refinement failed: {refinement_error}")
                        
                        results = {
                            'query': response_traj["query"], 
                            'tool list': response_traj["tool list"],
                            'trajectory_type': current_traj_type,
                            'num_tools_used': len(response_traj["tool list"]),
                            'target_tool_count': tool_count  # Add target tool count for clarity
                        }
                        
                        # Add validation info if checking was enabled
                        if args.enable_checking:
                            results['validated'] = is_valid
                            if not is_valid:
                                results['validation_errors'] = validation_errors
                        
                        records.append(results)
                        
                        # Save checkpoint on success (use new checkpoint key)
                        chk_index[chk_key] += 1
                        save_checkpoint(chk_index, chk_file)
                        success = True
                        print(f"Successfully processed record {j} for task '{target_task['task name']}' (type: {current_traj_type}, target: {tool_count}, actual: {len(response_traj['tool list'])} tools)")
                        
                    except Exception as e:
                        retry_count += 1
                        delay = args.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        print(f"Attempt {retry_count} failed for record {j} (task: {target_task['task name']}, tool_count: {tool_count}): {e}")
                        
                        if retry_count < args.max_retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                        else:
                            print(f"Max retries ({args.max_retries}) reached for record {j}, skipping...")
                            # Still increment checkpoint to avoid getting stuck on this record
                            chk_index[chk_key] += 1
                            save_checkpoint(chk_index, chk_file)
                
                # Save results after each successful record
                with open(save_file, 'w') as f:
                    json.dump(records, f, indent=4)
    # exit(0)
                

print("\n" + "="*80)
print("TRAJECTORY GENERATION SUMMARY")
print("="*80)

# Aggregate statistics across all domains
# total_records = 0
# traj_type_counts = {}
# tool_count_stats = []
# validation_stats = {'total': 0, 'validated': 0, 'failed': 0}

# for domain in domain_list:
#     save_file = os.path.join(args.save_dir, domain, f"simple_traj_{args.traj_type}_{args.model}_v2.json")
#     try:
#         with open(save_file, 'r') as f:
#             domain_records = json.load(f)
#             total_records += len(domain_records)
            
#             for record in domain_records:
#                 # Count trajectory types
#                 traj_type = record.get('trajectory_type', args.traj_type)
#                 traj_type_counts[traj_type] = traj_type_counts.get(traj_type, 0) + 1
                
#                 # Tool count statistics
#                 num_tools = record.get('num_tools_used', len(record.get('tool list', [])))
#                 tool_count_stats.append(num_tools)
                
#                 # Validation statistics
#                 if args.enable_checking:
#                     validation_stats['total'] += 1
#                     if record.get('validated', True):
#                         validation_stats['validated'] += 1
#                     else:
#                         validation_stats['failed'] += 1
#     except FileNotFoundError:
#         continue

# print(f"Total trajectories generated: {total_records}")
# print(f"Trajectory types: {dict(traj_type_counts)}")
# if tool_count_stats:
#     print(f"Tools per trajectory - Min: {min(tool_count_stats)}, Max: {max(tool_count_stats)}, Avg: {sum(tool_count_stats)/len(tool_count_stats):.1f}")

# if args.enable_checking and validation_stats['total'] > 0:
#     success_rate = validation_stats['validated'] / validation_stats['total'] * 100
#     print(f"Validation success rate: {success_rate:.1f}% ({validation_stats['validated']}/{validation_stats['total']})")

# print("="*80)

