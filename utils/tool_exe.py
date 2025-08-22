"""
This file is used to execute tol-calling trajectory and obtain final results
"""
import requests
import time
from typing import Dict, List, Any
from requests.exceptions import Timeout, RequestException


def tool_exe(tool_dict: Dict[str, Any], service_url: str = "http://8.130.32.149:8080/rapidapi", 
             toolbench_key: str = "8QoWASH6KnRNaEc0C1ZJq32dLeTMsbPzgYvuOmkGDlFX9B5wt4", 
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
                response_data = response.json()
                if 'response' in response_data:
                    return response_data['response']
                else:
                    return f"ERROR: No response field in API response"
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