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

parser = argparse.ArgumentParser('Sequential Tool Trajectory Generation')
parser.add_argument('-model', type=str, default='claude_v37', 
                   help='model name for trajectory generation', 
                   choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 
                           'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 
                           'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
parser.add_argument('-save_dir', type=str, default='/home/ec2-user/mountS3/newToolData/simple_query', 
                   help='directory to save generated trajectories')
parser.add_argument('-chk_dir', type=str, default='./chk/traj_sequential', 
                   help='checkpoint directory for resuming generation')
parser.add_argument('-min_tools', type=int, default=3, 
                   help='minimum number of tools in trajectory')
parser.add_argument('-max_tools', type=int, default=10, 
                   help='maximum number of tools in trajectory')
parser.add_argument('-trajectories_per_tool_count', type=int, default=5, 
                   help='number of trajectories to generate for each tool count')
parser.add_argument('-max_retries', type=int, default=3, 
                   help='maximum retry attempts per trajectory generation')
parser.add_argument('-retry_delay', type=float, default=1.0, 
                   help='initial retry delay in seconds (exponential backoff)')
parser.add_argument('-reset', action='store_true', 
                   help='reset checkpoint and start from the beginning')
args = parser.parse_args()

def load_checkpoint(chk_file):
    """Load checkpoint from a single file containing progress for all domains/tasks"""
    try:
        with open(chk_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # no checkpoint file, start from beginning
    except json.JSONDecodeError:
        print(f"Warning: Corrupted checkpoint file {chk_file}, starting fresh")
        return {}

def save_checkpoint_and_records(checkpoint_data, chk_file, records, save_file):
    """Atomically save both checkpoint and records together"""
    try:
        # Save checkpoint
        with open(chk_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save records
        with open(save_file, 'w') as f:
            json.dump(records, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving checkpoint/records: {e}")
        return False

# =====================================================================
# PROMPT TEMPLATES
# =====================================================================

INITIAL_TOOL_SELECTION_PROMPT = """
You are an expert system designer. Your task is to START a sequential tool-calling trajectory for the following:

Domain: {domain}
Task Description: {task_description}

Available Tools:
{tool_set_json}

Your goal is to select the FIRST tool that should be used to begin this task. This tool will be the starting point of a sequential chain where each subsequent tool will depend on the output of previous tools.

Please respond with the following JSON format:

```json
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
    ],
    "API name": "<API name>",
    "domain name": "<domain name>",
    "parent tool name": "<parent tool name>"
}}
```

Choose the most logical starting tool that will provide useful output for subsequent tools.
"""

NEXT_TOOL_SELECTION_PROMPT = """
You are continuing a sequential tool-calling trajectory. Here's what has happened so far:

Domain: {domain}
Task Description: {task_description}

Previous Steps:
{previous_context}

Previous outputs:
{previous_outputs}

Available Tools:
{tool_set_json}

Current step: {current_step} of {step_limit}

Your task is to:
1. Choose the NEXT tool that should logically follow from the previous steps;
2. Use outputs from previous steps in your tool parameters


The tool you select must:
- Use output from at least one previous step as input
- Move the task closer to completion
- Be realistic and logical given the context

Please respond with one of these JSON formats:

```json
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
    ],
    "API name": "<API name>",
    "domain name": "<domain name>",
    "parent tool name": "<parent tool name>"
}}
```
"""

QUERY_GENERATION_PROMPT = """
Generate a natural language query that would require the following sequential tool-calling trajectory:

Domain: {domain}
Task: {task_name}
Task Description: {task_description}

Tool Sequence:
{tool_sequence}

The query should be realistic, complex, and clearly require this specific sequence of tools to complete.
Respond with just the query text, no additional formatting.
"""

# =====================================================================
# CORE FUNCTIONS
# =====================================================================

def prompt_model_for_next_step(model, task_description, domain, tool_set, previous_steps=None, step_limit=5):
    """
    Prompts the language model to get the next tool and its parameters for sequential trajectory.
    
    Args:
        model: Model name/identifier to use
        task_description: Description of the task to accomplish
        domain: Domain/category of the task
        tool_set: List of available tools (domain tools for first step, connected tools for subsequent steps)
        previous_steps: List of previous steps in the trajectory (None for first step)
        step_limit: Maximum number of steps allowed
    
    Returns:
        Dict with complete tool information in the expected format
    """
    current_step = len(previous_steps) if previous_steps else 0
    
    if previous_steps is None or len(previous_steps) == 0:
        # Step 1: Initialize the trajectory
        prompt = INITIAL_TOOL_SELECTION_PROMPT.format(
            domain=domain,
            task_description=task_description,
            tool_set_json=json.dumps(tool_set, indent=2)
        )
    else:
        # Step 2+: Expand the trajectory based on previous steps
        previous_context = "\n".join([
            f"Step {i+1}: Tool '{step.get('tool name', 'unknown')}' with parameters {step.get('required parameters', [])}"
            for i, step in enumerate(previous_steps)
        ])
        
        previous_outputs = "\n".join([
            f"Step {i+1} output: {step.get('output', 'No output')[:200]}..."
            for i, step in enumerate(previous_steps)
        ])
        
        prompt = NEXT_TOOL_SELECTION_PROMPT.format(
            domain=domain,
            task_description=task_description,
            previous_context=previous_context,
            previous_outputs=previous_outputs,
            tool_set_json=json.dumps(tool_set, indent=2),
            current_step=current_step + 1,
            step_limit=step_limit
        )
    
    try:
        response = generate_content_with_retry(model, prompt)
        result = extract_json_from_markdown_fence(response, expect_dict=True)
        
        # Check if extraction failed
        if result is None:
            try:
                result = json.loads(response)
            except json.JSONDecodeError as je:
                # Try to extract JSON-like content more aggressively
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                else:
                    raise ValueError(f"Failed to parse JSON from response: {je}")
        
        # Validate response
        if not isinstance(result, dict):
            raise ValueError(f"Response is not a dictionary: {type(result)}")
            
        if "tool name" not in result:
            raise ValueError("Response missing 'tool name' field")
        
        if result.get("tool name") == "finish":
            return {"tool name": "finish", "reasoning": result.get("reasoning", "Task complete")}
        
        # Check if tool exists in tool_set
        available_tools = {tool.get('tool name') for tool in tool_set}
        if result["tool name"] not in available_tools:
            raise ValueError(f"Tool '{result['tool name']}' not found in available tools")
        
        return result
        
    except Exception as e:
        print(f"Error in model prompt: {e}")
        # Fallback to finish if there's an error
        return {"tool name": "finish", "reasoning": f"Error occurred: {e}"}

def substitute_placeholders(parameters, previous_tools):
    """
    Replace placeholder references like {{output from step 1}} with actual values from previous steps.
    
    Args:
        parameters: Dict of parameters that may contain placeholders
        previous_tools: List of previous tool dicts with outputs
    
    Returns:
        Dict with placeholders replaced by actual values
    """
    if not previous_tools:
        return parameters
    
    import re
    substituted = {}
    
    for key, value in parameters.items():
        if isinstance(value, str):
            # Look for patterns like {{output from step 1}} or {{step 2 output}}
            placeholder_pattern = r'\{\{([^}]+)\}\}'
            matches = re.findall(placeholder_pattern, value)
            
            new_value = value
            for match in matches:
                # Try to extract step number
                step_match = re.search(r'step\s+(\d+)', match.lower())
                if step_match:
                    step_num = int(step_match.group(1)) - 1  # Convert to 0-based index
                    if 0 <= step_num < len(previous_tools):
                        # Replace with the actual output from that step
                        step_output = str(previous_tools[step_num].get('output', ''))
                        new_value = new_value.replace('{{' + match + '}}', step_output)
            
            substituted[key] = new_value
        else:
            substituted[key] = value
    
    return substituted

def execute_tool_step(tool_name, parameters, tool_set):
    """
    Execute a single tool step by finding the tool in the tool_set and calling tool_exe.
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Parameters for the tool
        tool_set: List of available tools with their metadata
    
    Returns:
        String output from tool execution
    """
    # Find the tool in the tool_set
    tool_info = None
    for tool in tool_set:
        if tool.get('tool name') == tool_name:
            tool_info = tool
            break
    
    if not tool_info:
        return f"ERROR: Tool '{tool_name}' not found in tool set"
    
    # Get parameter definitions from tool info
    tool_required_params = tool_info.get('required_parameters', [])
    tool_optional_params = tool_info.get('optional_parameters', [])
    
    # Extract parameter names for comparison
    required_param_names = {param.get('name') for param in tool_required_params if isinstance(param, dict)}
    optional_param_names = {param.get('name') for param in tool_optional_params if isinstance(param, dict)}
    
    # Convert parameters to the format expected by tool_exe
    formatted_required_params = []
    formatted_optional_params = []
    
    for param_name, param_value in parameters.items():
        param_entry = {
            'name': param_name,
            'value': param_value
        }
        
        if param_name in required_param_names:
            formatted_required_params.append(param_entry)
        elif param_name in optional_param_names:
            formatted_optional_params.append(param_entry)
        # If parameter doesn't match any defined parameter, treat as required by default
        else:
            formatted_required_params.append(param_entry)
    
    # Create tool dict for tool_exe
    tool_dict = {
        'tool name': tool_info.get('tool name'),
        'domain name': tool_info.get('domain name'),
        'parent tool name': tool_info.get('parent tool name'),
        'API name': tool_info.get('API name'),
        'required parameters': formatted_required_params,
        'optional parameters': formatted_optional_params
    }
    
    # Import and execute the tool
    from utils.tool_exe import tool_exe
    return tool_exe(tool_dict)

def generate_sequential_trajectory(model, domain, task_name, task_description, tool_list, step_limit=5):
    """
    Generates a sequential tool-calling trajectory where each tool depends on previous outputs.
    
    Args:
        model: Model name to use for generation
        domain: Domain/category of the task
        task_name: Name of the task type
        task_description: Detailed description of the task
        tool_list: List of available tools for the domain
        step_limit: Maximum number of steps in the trajectory
    
    Returns:
        Dict containing the complete trajectory with query and tool list
    """
    trajectory_tools = []  # This will contain the final tool list for the trajectory
    
    print(f"\n=== Generating Sequential Trajectory ===")
    print(f"Domain: {domain}")
    print(f"Task: {task_name}")
    print(f"Description: {task_description}")
    print(f"Step Limit: {step_limit}")
    
    for step_num in range(step_limit):
        print(f"\n--- Step {step_num + 1} ---")
        
        # Determine available tools for this step
        if step_num == 0:
            # First step: use all domain tools
            available_tools = tool_list
        else:
            # Subsequent steps: use connected tools from the previous tool
            previous_tool = trajectory_tools[-1]
            previous_tool_name = previous_tool.get('tool name')
            
            # Find the original tool definition to get connected tools
            original_tool = next((tool for tool in tool_list if tool.get('tool name') == previous_tool_name), {})
            connected_tools_list = original_tool.get('connected tools', [])
            
            # Find the actual tool objects for connected tools
            available_tools = []
            tool_name_map = {tool.get('tool name'): tool for tool in tool_list}
            
            for connected_tool in connected_tools_list:
                connected_tool_name = connected_tool.get('tool name')
                if connected_tool_name and connected_tool_name in tool_name_map:
                    available_tools.append(tool_name_map[connected_tool_name])
            
            # Fallback: if no connected tools found, allow any domain tool to continue
            if not available_tools:
                print(f"No connected tools found from previous step. Using all domain tools as fallback.")
                available_tools = tool_list
        
        print(f"Available tools: {[tool.get('tool name') for tool in available_tools]}")
        
        # Get the next step from the model
        next_tool_dict = prompt_model_for_next_step(
            model, task_description, domain, available_tools, 
            trajectory_tools, step_limit
        )
        
        tool_name = next_tool_dict.get("tool name")

        if tool_name == "finish":
            reasoning = next_tool_dict.get("reasoning", "Task complete")
            print(f"✅ Model decided to finish: {reasoning}")
            break
        
        print(f"Selected tool: {tool_name}")
        print(f"Tool dict: {next_tool_dict}")
        
        # Validate tool dict completeness
        required_fields = ['tool name', 'tool description', 'required parameters', 'optional parameters', 'API name', 'domain name', 'parent tool name']
        missing_fields = [field for field in required_fields if field not in next_tool_dict]
        if missing_fields:
            print(f"Warning: Tool dict missing fields: {missing_fields}")
            # Fill in missing fields from the original tool definition
            original_tool = next((tool for tool in available_tools if tool.get('tool name') == tool_name), {})
            for field in missing_fields:
                if field in original_tool:
                    next_tool_dict[field] = original_tool[field]
                else:
                    next_tool_dict[field] = "" if field != 'required parameters' and field != 'optional parameters' else []
        
        # Ensure parameters are in list format
        if 'required parameters' in next_tool_dict and not isinstance(next_tool_dict['required parameters'], list):
            next_tool_dict['required parameters'] = []
        if 'optional parameters' in next_tool_dict and not isinstance(next_tool_dict['optional parameters'], list):
            next_tool_dict['optional parameters'] = []
        
        # Extract parameters for execution
        required_params = next_tool_dict.get('required parameters', [])
        optional_params = next_tool_dict.get('optional parameters', [])
        
        # Convert to dict format for substitution
        parameters = {}
        for param in required_params + optional_params:
            if isinstance(param, dict) and 'name' in param and 'value' in param:
                parameters[param['name']] = param['value']
        
        # Substitute placeholders with actual values from previous steps
        actual_parameters = substitute_placeholders(parameters, trajectory_tools)
        
        # Execute the tool
        try:
            output = execute_tool_step(tool_name, actual_parameters, available_tools)
            output_str = str(output)
            
            # Check for execution errors
            if output_str.startswith("ERROR:"):
                next_tool_dict['execution_status'] = 'failed'
            else:
                next_tool_dict['execution_status'] = 'success'
                
        except Exception as e:
            output = f"ERROR: Tool execution failed - {str(e)}"
            next_tool_dict['execution_status'] = 'failed'
        
        # Add output to the tool dict
        next_tool_dict['output'] = output
        
        # Append the complete tool dict to trajectory
        trajectory_tools.append(next_tool_dict)
        
        # Stop if we've reached step limit or if multiple consecutive failures
        if step_num >= step_limit - 1:
            print(f"Reached step limit ({step_limit})")
            break
            
        # Count consecutive failures
        consecutive_failures = 0
        for tool in reversed(trajectory_tools):
            if tool.get('execution_status') == 'failed':
                consecutive_failures += 1
            else:
                break
        
        if consecutive_failures >= 2:
            print(f"Stopping trajectory due to {consecutive_failures} consecutive tool execution failures")
            break
    
    # Validate trajectory has at least one tool
    if not trajectory_tools:
        print("Warning: No tools were successfully added to trajectory")
        return {
            'query': f"Failed to generate sequential trajectory for {task_description}",
            'tool list': [],
            'trajectory_type': 'sequential',
            'num_tools_used': 0,
            'task_name': task_name,
            'task_description': task_description,
            'executable': False,
            'error': 'No tools in trajectory'
        }
    
    # Generate a natural language query that describes the full trajectory
    tool_sequence = chr(10).join([f"{i+1}. {tool.get('tool name', 'unknown')}: {tool.get('tool description', '')}" for i, tool in enumerate(trajectory_tools)])
    query_prompt = QUERY_GENERATION_PROMPT.format(
        domain=domain,
        task_name=task_name,
        task_description=task_description,
        tool_sequence=tool_sequence
    )
    
    try:
        query_response = generate_content_with_retry(model, query_prompt)
        query = query_response.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Error generating query: {e}")
        query = f"Complete {task_description} using sequential tool calls"
    
    # Check if trajectory has any successful executions
    successful_tools = [tool for tool in trajectory_tools if tool.get('execution_status') == 'success']
    
    result = {
        'query': query,
        'tool list': trajectory_tools,  # Use the trajectory_tools directly
        'trajectory_type': 'sequential',
        'num_tools_used': len(trajectory_tools),
        'num_successful_tools': len(successful_tools),
        'task_name': task_name,
        'task_description': task_description,
        'executable': len(successful_tools) > 0  # Only executable if at least one tool succeeded
    }
    
    print(f"\n=== Generated Trajectory ===")
    print(f"Query: {query}")
    print(f"Number of tools: {len(trajectory_tools)}")
    print(f"Successful tools: {len(successful_tools)}")
    print(f"Executable: {result['executable']}")
    
    return result


### data loading
# domain list
with open('/home/ec2-user/mountS3/newToolData/selected_category.json', 'r') as f:
    select_cate = json.load(f)
domain_list = list(select_cate.keys())
num_domain = len(domain_list)

# Create checkpoint directory if it doesn't exist
os.makedirs(args.chk_dir, exist_ok=True)

# Use single checkpoint file for all domains
chk_file = os.path.join(args.chk_dir, f"sequential_traj_{args.model}_checkpoint.json")

# Handle reset flag
if args.reset:
    print("Reset flag detected. Clearing checkpoint and starting from the beginning.")
    if os.path.exists(chk_file):
        os.remove(chk_file)
        print(f"Removed checkpoint file: {chk_file}")
    global_checkpoint = {}
else:
    global_checkpoint = load_checkpoint(chk_file)

# main loop
for domain in domain_list:
    print(f"Generating trajectory for domain: {domain}")
    with open(f'/home/ec2-user/mountS3/newToolData/tools/{domain}/true_filter_subtool.json', 'r') as f:
        tool_list = json.load(f)
    with open(f'/home/ec2-user/mountS3/newToolData/simple_query/{domain}/task_type.json', 'r') as f:
        task_type = json.load(f)
    keys_to_keep = ['parent tool name', 'API name', 'domain name', 'tool name', 'tool description', 'required_parameters', 'optional_parameters']
    # Setup save directory and file
    save_dir = os.path.join(args.save_dir, domain)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"simple_traj_sequential_{args.model}.json")

    # Always try to load existing records to avoid overwriting
    try:
        with open(save_file, 'r') as f:
            records = json.load(f)
        print(f"Loaded {len(records)} existing records for domain {domain}")
    except FileNotFoundError:
        records = []
        print(f"Starting with empty records for domain {domain}")
    
    # iterate over task types
    for i, target_task in enumerate(task_type):
        target_tools = tool_list
        
        # Generate trajectories for different tool counts
        for tool_count in range(args.min_tools, args.max_tools + 1):
            print(f"Generating trajectories for task '{target_task['task name']}' with {tool_count} tools")
            
            # Create unique checkpoint key for this domain/task/tool_count combination
            chk_key = f"{domain}_{target_task['task name']}_{tool_count}"
            
            # Get starting index from global checkpoint
            start_idx = global_checkpoint.get(chk_key, 0)
            
            # Generate multiple trajectories for this tool count
            for j in range(start_idx, args.trajectories_per_tool_count):
                retry_count = 0
                success = False
                
                while retry_count < args.max_retries and not success:
                    try:
                        # Generate sequential trajectory
                        response_traj = generate_sequential_trajectory(
                            args.model, domain, target_task['task name'],
                            target_task['task description'], target_tools,
                            tool_count  # Use the specific tool count for this iteration
                        )
                        
                        records.append(response_traj)
                        
                        # Update global checkpoint
                        global_checkpoint[chk_key] = j + 1
                        
                        # Save both checkpoint and records atomically
                        if save_checkpoint_and_records(global_checkpoint, chk_file, records, save_file):
                            success = True
                            print(f"Successfully processed record {j} for task '{target_task['task name']}' with {tool_count} tools")
                        else:
                            raise Exception("Failed to save checkpoint/records")
                        
                    except Exception as e:
                        retry_count += 1
                        delay = args.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        print(f"Attempt {retry_count} failed for record {j}: {e}")
                        
                        if retry_count < args.max_retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                        else:
                            print(f"Max retries ({args.max_retries}) reached for record {j}, skipping...")
                            # Still increment checkpoint to avoid getting stuck on this record
                            global_checkpoint[chk_key] = j + 1
                            save_checkpoint_and_records(global_checkpoint, chk_file, records, save_file)
            exit(0)
