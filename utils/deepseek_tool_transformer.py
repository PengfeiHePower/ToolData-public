#!/usr/bin/env python3
"""
Tool Format Transformer

This script transforms tools from the current format to the model-specified format
for agentic tool usage capabilities.

Current format structure:
- parent tool name
- parent tool description  
- required_parameters (array of {name, type, description, default})
- optional_parameters (array of {name, type, description, default})
- tool name
- tool description
- API name
- domain name
- output_info
- connected tools
- code

Target format structure:
- type: "function"
- function: {name, description, parameters: {type: "object", properties: {...}, required: [...], optional: [...]}}
"""

import json
import os
from typing import List, Dict, Any, Optional


def map_parameter_type(param_type: str) -> str:
    """
    Map parameter types from current format to JSON Schema types.
    
    Args:
        param_type: The parameter type from current format
        
    Returns:
        JSON Schema compatible type
    """
    type_mapping = {
        "STRING": "string",
        "NUMBER": "number", 
        "INTEGER": "integer",
        "BOOLEAN": "boolean",
        "ARRAY": "array",
        "OBJECT": "object",
        "DATE (YYYY-MM-DD)": "string",
        "DATE": "string",
        "TIME": "string",
        "DATETIME": "string",
        "EMAIL": "string",
        "URL": "string",
        "PHONE": "string",
        "JSON": "object",
        "FILE": "string"
    }
    
    # Handle complex types by extracting the base type
    if "(" in param_type:
        base_type = param_type.split("(")[0].strip()
    else:
        base_type = param_type.strip()
    
    return type_mapping.get(base_type.upper(), "string")


def transform_parameter(param: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single parameter from current format to JSON Schema format.
    
    Args:
        param: Parameter in current format
        
    Returns:
        Parameter in JSON Schema format
    """
    param_name = param.get("name", "")
    param_type = param.get("type", "STRING")
    param_description = param.get("description", "")
    
    # Map the type to JSON Schema format
    json_type = map_parameter_type(param_type)
    
    # Create the parameter schema
    param_schema = {
        "type": json_type,
        "description": param_description
    }
    
    # Add default value if present and not empty
    if "default" in param and param["default"] != "":
        param_schema["default"] = param["default"]
    
    return param_schema


def transform_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single tool from current format to model-specified format.
    
    Args:
        tool: Tool in current format
        
    Returns:
        Tool in model-specified format
    """
    # Extract basic information
    tool_name = tool.get("tool name", tool.get("API name", "unnamed_tool"))
    tool_description = tool.get("tool description", tool.get("parent tool description", ""))
    
    # Process parameters
    required_params = tool.get("required_parameters", [])
    optional_params = tool.get("optional_parameters", [])
    
    # Create properties dictionary
    properties = {}
    required_param_names = []
    
    # Add required parameters
    for param in required_params:
        param_name = param.get("name", "")
        if param_name:
            properties[param_name] = transform_parameter(param)
            required_param_names.append(param_name)
    
    # Add optional parameters
    for param in optional_params:
        param_name = param.get("name", "")
        if param_name:
            properties[param_name] = transform_parameter(param)
    
    # Create optional parameter names list
    optional_param_names = []
    for param in optional_params:
        param_name = param.get("name", "")
        if param_name:
            optional_param_names.append(param_name)
    
    # Create the transformed tool
    transformed_tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_param_names,
                "optional": optional_param_names
            }
        }
    }
    
    return transformed_tool


def load_tools_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load tools from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing tools
        
    Returns:
        List of tools
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        return tools if isinstance(tools, list) else [tools]
    except Exception as e:
        print(f"Error loading tools from {file_path}: {e}")
        return []


def transform_tools_from_file(input_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Transform tools from a file and optionally save to output file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save transformed tools
        
    Returns:
        List of transformed tools
    """
    # Load tools from file
    tools = load_tools_from_file(input_file)
    
    if not tools:
        print(f"No tools found in {input_file}")
        return []
    
    # Transform each tool
    transformed_tools = []
    for tool in tools:
        try:
            transformed_tool = transform_tool(tool)
            transformed_tools.append(transformed_tool)
        except Exception as e:
            print(f"Error transforming tool {tool.get('tool name', 'unknown')}: {e}")
            continue
    
    # Save to output file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transformed_tools, f, indent=2, ensure_ascii=False)
            print(f"Transformed {len(transformed_tools)} tools saved to {output_file}")
        except Exception as e:
            print(f"Error saving transformed tools to {output_file}: {e}")
    
    return transformed_tools


def transform_all_tools_in_directory(input_dir: str, output_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transform all tool files in a directory.
    
    Args:
        input_dir: Directory containing tool JSON files
        output_dir: Optional directory to save transformed tools
        
    Returns:
        Dictionary mapping filenames to transformed tools
    """
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return {}
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    
    # Process each JSON file in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            output_file = None
            
            if output_dir:
                output_filename = filename.replace('.json', '_transformed.json')
                output_file = os.path.join(output_dir, output_filename)
            
            print(f"Processing {filename}...")
            transformed_tools = transform_tools_from_file(input_file, output_file)
            results[filename] = transformed_tools
    
    return results


def main():
    """
    Main function to demonstrate usage.
    """
    # Example usage
    input_directory = "/home/ubuntu/newToolData/public_data/v4/tools"
    output_directory = "/home/ubuntu/newToolData/transformed_tools"
    
    print("Starting tool transformation...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # Transform all tools
    results = transform_all_tools_in_directory(input_directory, output_directory)
    
    # Print summary
    total_tools = sum(len(tools) for tools in results.values())
    print(f"\nTransformation complete!")
    print(f"Total tools transformed: {total_tools}")
    
    for filename, tools in results.items():
        print(f"  {filename}: {len(tools)} tools")
    
    # Create a combined file with all transformed tools
    all_transformed_tools = []
    for tools in results.values():
        all_transformed_tools.extend(tools)
    
    if all_transformed_tools:
        combined_output = os.path.join(output_directory, "all_transformed_tools.json")
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_transformed_tools, f, indent=2, ensure_ascii=False)
        print(f"\nCombined file created: {combined_output}")


if __name__ == "__main__":
    main()
