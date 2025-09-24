"""
This file is used to execute tol-calling trajectory and obtain final results
"""
import os
import requests
import time
import json
from typing import Dict, List, Any, Optional
from requests.exceptions import Timeout, RequestException


def tool_exe(tool_dict: Dict[str, Any], service_url: str=os.getenv("API_URL"), 
             toolbench_key: str=os.getenv("TOOLBENCH_KEY"), 
             max_retries: int = 3, base_timeout: int = 15) -> str:
    """
    Execute a single tool from a trajectory and return its result.
    
    Args:
        tool_dict: Dict containing tool info with keys like 'tool name', 'required parameters', etc.
        service_url: API endpoint URL
        toolbench_key: Authentication key
        max_retries: Maximum number of retry attempts
        base_timeout: Base timeout in seconds
    
    Returns:
        String result from tool execution
    """
    headers = {"toolbench_key": toolbench_key}
    
    # Extract tool information
    tool_name = tool_dict.get('tool name', '')
    required_params = tool_dict.get('required parameters', [])
    optional_params = tool_dict.get('optional parameters', [])
    
    # Convert required and optional parameters to dict format
    parameters = {}
    for param in required_params:
        if isinstance(param, dict) and 'name' in param and 'value' in param:
            parameters[param['name']] = param['value']
    
    for param in optional_params:
        if isinstance(param, dict) and 'name' in param and 'value' in param:
            parameters[param['name']] = param['value']
    
    # Prepare payload for API call
    payload = {
        "category": tool_dict.get('domain name', ''),
        "tool_name": tool_dict.get('parent tool name', tool_name),
        "api_name": tool_dict.get('API name', tool_name),
        "tool_input": parameters,
        "strip": "truncate",
        "toolbench_key": toolbench_key
    }
    
    # Execute with retry logic
    timeouts = [base_timeout, base_timeout * 2, base_timeout * 4]
    
    for attempt in range(max_retries):
        try:
            timeout = timeouts[attempt] if attempt < len(timeouts) else timeouts[-1]
            response = requests.post(service_url, json=payload, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if 'response' in response_data:
                        return response_data['response']
                    else:
                        return f"ERROR: No response field in API response. Available fields: {list(response_data.keys())}"
                except Exception as json_error:
                    return f"ERROR: Failed to parse JSON response - {str(json_error)}. Raw response: {response.text[:500]}"
            else:
                return f"ERROR: HTTP {response.status_code} - {response.text}"
                
        except Timeout:
            if attempt == max_retries - 1:
                return f"ERROR: Timeout after {max_retries} attempts"
            time.sleep(2 ** attempt)
        except RequestException as e:
            if attempt == max_retries - 1:
                return f"ERROR: Request failed - {str(e)}"
            time.sleep(2 ** attempt)
        except Exception as e:
            return f"ERROR: Unexpected error - {str(e)}"
    
    return "ERROR: Tool execution failed"


def execute_trajectory(trajectory_data: List[Dict[str, Any]], service_url: str = "http://8.130.32.149:8080/rapidapi",
                      toolbench_key: str = "8QoWASH6KnRNaEc0C1ZJq32dLeTMsbPzgYvuOmkGDlFX9B5wt4") -> List[Dict[str, Any]]:
    """
    Execute a complete tool-calling trajectory and return results.
    
    Args:
        trajectory_data: List of trajectory items, each containing 'query' and 'tool list'
        service_url: API endpoint URL
        toolbench_key: Authentication key
    
    Returns:
        List of results for each trajectory item
    """
    results = []
    
    for traj_item in trajectory_data:
        query = traj_item.get('query', '')
        tool_list = traj_item.get('tool list', [])
        
        tool_results = []
        for tool in tool_list:
            result = tool_exe(tool, service_url, toolbench_key)
            tool_results.append({
                'tool_name': tool.get('tool name', ''),
                'tool_description': tool.get('tool description', ''),
                'result': result
            })
        
        results.append({
            'query': query,
            'tool_results': tool_results
        })
    
    return results


def load_tools_database(tools_file_path: str = "/home/ubuntu/newToolData/public_data/v4/tools/all_tools.json") -> List[Dict[str, Any]]:
    """
    Load the tools database from JSON file.
    
    Args:
        tools_file_path: Path to the all_tools.json file
        
    Returns:
        List of all tools from the database
    """
    try:
        with open(tools_file_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        return tools if isinstance(tools, list) else []
    except Exception as e:
        print(f"Error loading tools database: {e}")
        return []


def find_tool_by_name(tool_name: str, tools_database: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find a tool in the database by its name.
    
    Args:
        tool_name: Name of the tool to find
        tools_database: List of all tools from the database
        
    Returns:
        Tool dictionary if found, None otherwise
    """
    for tool in tools_database:
        if tool.get('tool name', '') == tool_name:
            return tool
    return None


def execute_tool_by_name(tool_name: str, parameters: Dict[str, Any], 
                        tools_file_path: str = "/home/ubuntu/newToolData/public_data/v4/tools/all_tools.json",
                        service_url: str = os.getenv("API_URL"), 
                        toolbench_key: str = os.getenv("TOOLBENCH_KEY"),
                        max_retries: int = 3, base_timeout: int = 15) -> str:
    """
    Execute a tool by its name and parameters.
    
    This function looks up the tool in the all_tools.json database, extracts the necessary
    information (category, API name, etc.), and calls the existing tool_exe function.
    
    Args:
        tool_name: Name of the tool to execute (must match 'tool name' field in database)
        parameters: Dictionary of parameter names and values {"param_name": "param_value", ...}
        tools_file_path: Path to the all_tools.json file
        service_url: API endpoint URL
        toolbench_key: Authentication key
        max_retries: Maximum number of retry attempts
        base_timeout: Base timeout in seconds
    
    Returns:
        String result from tool execution
        
    Example:
        result = execute_tool_by_name(
            "AI Trip Planner: Get Trip Plan",
            {"days": "5", "destination": "Paris, France"}
        )
    """
    # Load tools database
    tools_database = load_tools_database(tools_file_path)
    if not tools_database:
        return "ERROR: Could not load tools database"
    
    # Find the tool by name
    tool = find_tool_by_name(tool_name, tools_database)
    if not tool:
        return f"ERROR: Tool '{tool_name}' not found in database"
    
    # Validate required parameters
    required_params = tool.get('required_parameters', [])
    required_param_names = [param.get('name', '') for param in required_params if param.get('name')]
    
    missing_params = []
    for param_name in required_param_names:
        if param_name not in parameters:
            missing_params.append(param_name)
    
    if missing_params:
        return f"ERROR: Missing required parameters: {missing_params}"
    
    # Prepare tool_dict in the format expected by tool_exe
    tool_dict = {
        'tool name': tool.get('tool name', ''),
        'parent tool name': tool.get('parent tool name', ''),
        'API name': tool.get('API name', ''),
        'domain name': tool.get('domain name', ''),
        'tool description': tool.get('tool description', ''),
        'required parameters': [],
        'optional parameters': []
    }
    
    # Convert parameters to the format expected by tool_exe
    for param_name, param_value in parameters.items():
        param_entry = {
            'name': param_name,
            'value': param_value
        }
        
        # Check if it's a required parameter
        if param_name in required_param_names:
            tool_dict['required parameters'].append(param_entry)
        else:
            tool_dict['optional parameters'].append(param_entry)
    
    # Execute the tool using the existing tool_exe function
    return tool_exe(tool_dict, service_url, toolbench_key, max_retries, base_timeout)


def list_available_tools(tools_file_path: str = "/home/ubuntu/newToolData/public_data/v4/tools/all_tools.json") -> List[Dict[str, str]]:
    """
    List all available tools with their names and descriptions.
    
    Args:
        tools_file_path: Path to the all_tools.json file
        
    Returns:
        List of dictionaries containing tool names and descriptions
    """
    tools_database = load_tools_database(tools_file_path)
    if not tools_database:
        return []
    
    available_tools = []
    for tool in tools_database:
        available_tools.append({
            'name': tool.get('tool name', ''),
            'description': tool.get('tool description', ''),
            'category': tool.get('domain name', ''),
            'parent_tool': tool.get('parent tool name', '')
        })
    
    return available_tools


def get_tool_info(tool_name: str, tools_file_path: str = "/home/ubuntu/newToolData/public_data/v4/tools/all_tools.json") -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific tool.
    
    Args:
        tool_name: Name of the tool to get info for
        tools_file_path: Path to the all_tools.json file
        
    Returns:
        Dictionary containing tool information including parameters, or None if not found
    """
    tools_database = load_tools_database(tools_file_path)
    if not tools_database:
        return None
    
    tool = find_tool_by_name(tool_name, tools_database)
    if not tool:
        return None
    
    # Extract parameter information
    required_params = tool.get('required_parameters', [])
    optional_params = tool.get('optional_parameters', [])
    
    return {
        'name': tool.get('tool name', ''),
        'description': tool.get('tool description', ''),
        'category': tool.get('domain name', ''),
        'parent_tool': tool.get('parent tool name', ''),
        'api_name': tool.get('API name', ''),
        'required_parameters': [
            {
                'name': param.get('name', ''),
                'type': param.get('type', ''),
                'description': param.get('description', ''),
                'default': param.get('default', '')
            }
            for param in required_params
        ],
        'optional_parameters': [
            {
                'name': param.get('name', ''),
                'type': param.get('type', ''),
                'description': param.get('description', ''),
                'default': param.get('default', '')
            }
            for param in optional_params
        ]
    }