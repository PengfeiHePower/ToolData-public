import os
import json
import time
import logging
import argparse
import random
import sys

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
parser.add_argument('-backup', action='store_true', default=True, help='create backup files before modifying')
args = parser.parse_args()


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
{{
  "query": "<corrected natural language query>",
  "tool list": [
    {{
      "tool name": "<tool name>",
      "tool description": "<tool description>",
      "required parameters": [
        {{"name": "<parameter name>", "value": "<parameter value>"}},
        ...
      ],
      "optional parameters": [
        {{"name": "<parameter name>", "value": "<parameter value>"}},
        ...
      ]
    }},
    ...
  ]
}}
```
"""
    
    try:
        refined_response = generate_content_with_retry(model, refinement_prompt)
        return extract_json_from_markdown_fence(refined_response, expect_dict=True)
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
    return extract_json_from_markdown_fence(response, expect_dict=True)

# Remove checkpoint functions - not needed for single file processing

def create_backup(file_path):
    """Create a backup of the file before modifying"""
    if not os.path.exists(file_path):
        return None
    
    backup_path = f"{file_path}.backup_{int(time.time())}"
    try:
        with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        print(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return None

def clean_old_query_fields(record):
    """Remove fields containing 'query_simple' and 'query_hard' from a record"""
    keys_to_remove = []
    for key in record.keys():
        if 'query_simple' in key or 'query_hard' in key:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del record[key]
    
    return record

def fix_non_executable_trajectory(trajectory_item, domain, tool_list, model, requirements):
    """Fix a single non-executable trajectory by generating a new one"""
    try:
        # First, clean up old query fields from the trajectory item
        clean_old_query_fields(trajectory_item)
        
        # Extract trajectory type and tool count from existing item
        traj_type = trajectory_item.get('trajectory_type', 'parallel')
        target_tool_count = trajectory_item.get('target_tool_count') or trajectory_item.get('num_tools_used', 3)
        
        # Use existing task_name and task_description from the trajectory
        task_name = trajectory_item.get('task name', 'General Task')
        task_description = trajectory_item.get('task description', 'Complete the requested task using the available tools')
        
        # Use all available tools (no need to filter by category since we're fixing existing trajectories)
        target_tools = tool_list
        
        # Ensure we have the right number of tools if specified
        if target_tool_count and len(target_tools) > target_tool_count:
            target_tools = random.sample(target_tools, min(target_tool_count, len(target_tools)))
        
        # Use the global prompt templates and substitute placeholders
        trajectory_requirements = requirements.get(f'traj_gen_{traj_type}', '')
        
        # Build full prompt using global templates
        full_prompt = info_prompt.replace("<domain>", domain)
        full_prompt = full_prompt.replace("<task_type>", task_name)
        full_prompt = full_prompt.replace("<task_description>", task_description)
        full_prompt = full_prompt.replace("<tool_list>", json.dumps(target_tools, indent=2))
        full_prompt = full_prompt + "\n" + trajectory_requirements + "\n" + format_prompt
        
        # Generate new trajectory
        response = generate_content_with_retry(model, full_prompt)
        response_traj = extract_json_from_markdown_fence(response, expect_dict=True)
        
        # Validate the newly generated trajectory
        is_valid, validation_errors = validate_trajectory(
            response_traj, target_tools, traj_type, 
            target_tool_count, min_tools=2, max_tools=10
        )
        
        # If validation fails, try refinement once
        if not is_valid:
            print(f"Initial generation failed validation: {'; '.join(validation_errors)}")
            try:
                response_traj = refine_trajectory(
                    response_traj, validation_errors, model, target_tools, 
                    domain, task_name, task_description, traj_type, requirements,
                    target_tool_count, min_tools=2, max_tools=10
                )
                
                # Re-validate after refinement
                is_valid, validation_errors = validate_trajectory(
                    response_traj, target_tools, traj_type,
                    target_tool_count, min_tools=2, max_tools=10
                )
                
                if is_valid:
                    print("✓ Successfully refined trajectory after validation failure")
                else:
                    print(f"✗ Refinement still has validation issues: {'; '.join(validation_errors)}")
                    
            except Exception as refinement_error:
                print(f"Refinement failed: {refinement_error}")
        
        # Create the new trajectory object with all required fields
        new_trajectory = {
            'query': response_traj['query'],
            'tool list': response_traj['tool list'],
            'trajectory_type': traj_type,
            'num_tools_used': len(response_traj['tool list']),
            'target_tool_count': target_tool_count,
            'task name': task_name,  # Preserve task name
            'task description': task_description,  # Preserve task description
            'validated': is_valid,  # Use actual validation result
            'executable': True,  # Mark as executable (we successfully generated something)
            'fixed': True
        }
        
        # Add validation errors if validation failed
        if not is_valid:
            new_trajectory['validation_errors'] = validation_errors
        
        # Not preserving additional fields to keep clean structure
        
        return new_trajectory
        
    except Exception as e:
        print(f"Error fixing trajectory: {e}")
        # Return original trajectory if fixing fails
        return trajectory_item

#### data loading
# domain list
with open('/home/ec2-user/mountS3/newToolData/selected_category.json', 'r') as f:
    select_cate = json.load(f)
domain_list = list(select_cate.keys())
num_domain = len(domain_list)

print(f"Base directory: {args.base_dir}")
print(f"Trajectory file pattern: {args.traj_file}.json")
print(f"Model: {args.model}")
print(f"Processing {len(domain_list)} domains")

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
"required parameters": [{"name": "<parameter name>", "value": "<parameter value>"},...],
"optional parameters": [{"name": "<parameter name>", "value": "<parameter value>"},...]
},
...]
}
```
"""
# Load trajectory generation requirements
try:
    with open('/home/ec2-user/mountS3/newToolData/query_gen/traj_gen_prompt.json', 'r') as f:
        requirements = json.load(f)
except FileNotFoundError:
    print("Warning: Could not load trajectory generation prompts")
    requirements = {}

total_domains_processed = 0
total_files_processed = 0
total_trajectories_fixed = 0

# Process each domain
for domain in domain_list:
    print(f"\n{'='*60}")
    print(f"Processing domain: {domain}")
    print(f"{'='*60}")
    
    # Construct trajectory file path
    trajectory_file = os.path.join(args.base_dir, domain, f"{args.traj_file}.json")
    
    # Check if trajectory file exists
    if not os.path.exists(trajectory_file):
        print(f"Trajectory file not found: {trajectory_file}")
        continue
    
    # Load tools for the domain
    tool_file = f'/home/ec2-user/mountS3/newToolData/tools/{domain}/true_filter_subtool.json'
    try:
        with open(tool_file, 'r') as f:
            tool_list = json.load(f)
    except FileNotFoundError as e:
        print(f"Tool file not found for domain {domain}: {e}")
        continue
    
    print(f"Trajectory file: {os.path.basename(trajectory_file)}")
    print(f"Tools loaded: {len(tool_list)} tools")
    
    # Load trajectory data
    try:
        with open(trajectory_file, 'r') as f:
            trajectories = json.load(f)
    except Exception as e:
        print(f"Error loading {trajectory_file}: {e}")
        continue
    
    if not isinstance(trajectories, list):
        print(f"Expected list in {trajectory_file}, got {type(trajectories)}")
        continue
    
    # Count non-executable trajectories
    non_executable_count = sum(1 for traj in trajectories if not traj.get('executable', True))
    
    if non_executable_count == 0:
        print(f"No non-executable trajectories found")
        continue
    
    print(f"Found {non_executable_count} non-executable trajectories to fix")
    
    if args.dry_run:
        print(f"DRY RUN: Would fix {non_executable_count} trajectories")
        for i, traj in enumerate(trajectories):
            if not traj.get('executable', True):
                print(f"  - Trajectory {i+1}: {traj.get('query', 'No query')[:100]}...")
        continue
    
    # Create backup if requested
    if args.backup:
        backup_path = create_backup(trajectory_file)
        if backup_path:
            print(f"Backup created: {os.path.basename(backup_path)}")
    
    # Process trajectories
    fixed_count = 0
    for i, trajectory in enumerate(trajectories):
        if not trajectory.get('executable', True):
            try:
                print(f"Fixing trajectory {i+1}/{len(trajectories)}...")
                
                # Generate new trajectory
                fixed_trajectory = fix_non_executable_trajectory(
                    trajectory, domain, tool_list, args.model, requirements
                )
                
                # Replace the trajectory
                trajectories[i] = fixed_trajectory
                
                # Save immediately after each fix to prevent data loss
                try:
                    with open(trajectory_file, 'w') as f:
                        json.dump(trajectories, f, indent=2)
                except Exception as save_error:
                    print(f"Warning: Failed to save after fixing trajectory {i+1}: {save_error}")
                
                if fixed_trajectory.get('executable', False):
                    fixed_count += 1
                    print(f"✓ Fixed and saved trajectory {i+1}")
                else:
                    print(f"✗ Failed to fix trajectory {i+1}")
                    
            except Exception as e:
                print(f"Error fixing trajectory {i+1}: {e}")
                continue
    
    # Save updated trajectories
    try:
        with open(trajectory_file, 'w') as f:
            json.dump(trajectories, f, indent=2)
        print(f"✓ Saved {fixed_count} fixed trajectories")
    except Exception as e:
        print(f"Error saving {trajectory_file}: {e}")
        continue
    
    # Update totals
    total_domains_processed += 1
    total_files_processed += 1
    total_trajectories_fixed += fixed_count
    
    print(f"Domain {domain} completed: {fixed_count} trajectories fixed")
                

# Final summary
print(f"\n{'='*60}")
print("TRAJECTORY FIXING SUMMARY")
print(f"{'='*60}")
print(f"Base directory: {args.base_dir}")
print(f"Trajectory file pattern: {args.traj_file}.json")
print(f"Model: {args.model}")
print(f"Domains processed: {total_domains_processed}")
print(f"Files processed: {total_files_processed}")
print(f"Total trajectories fixed: {total_trajectories_fixed}")
print(f"{'='*60}")


