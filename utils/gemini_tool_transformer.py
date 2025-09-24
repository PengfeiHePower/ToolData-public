#!/usr/bin/env python3
"""
Gemini Tool Format Transformer

This script transforms tools from the current format to the Gemini model-specified format.

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

Gemini Target format structure:
- name: tool name
- description: tool description
- parameters: {type: "object", properties: {...}, required: [...]}
"""

import json
import os
import re
from typing import List, Dict, Any, Optional


def sanitize_tool_name(tool_name: str) -> str:
    """
    Preserve the original tool name without sanitization.
    
    Args:
        tool_name: Original tool name
        
    Returns:
        Original tool name (preserved)
    """
    # Return the original tool name without any modification
    return tool_name


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
    Transform a single parameter from current format to Gemini JSON Schema format.
    
    Args:
        param: Parameter in current format
        
    Returns:
        Parameter in Gemini JSON Schema format
    """
    param_name = param.get("name", "")
    param_type = param.get("type", "STRING")
    param_description = param.get("description", "")
    
    # Map the type to JSON Schema format
    json_type = map_parameter_type(param_type)
    
    # Create the parameter schema (Gemini format includes description)
    param_schema = {
        "type": json_type,
        "description": param_description
    }
    
    return param_schema


def transform_tool_to_gemini(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single tool from current format to Gemini format.
    
    Args:
        tool: Tool in current format
        
    Returns:
        Tool in Gemini format
    """
    # Extract basic information
    tool_name = tool.get("tool name", tool.get("API name", "unnamed_tool"))
    tool_description = tool.get("tool description", tool.get("parent tool description", ""))
    
    # Sanitize tool name for Gemini
    sanitized_name = sanitize_tool_name(tool_name)
    
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
    
    # Create the parameters schema
    parameters = {
        "type": "object",
        "properties": properties,
        "required": required_param_names
    }
    
    # Create the transformed tool in Gemini format
    transformed_tool = {
        "name": sanitized_name,
        "description": tool_description,
        "parameters": parameters
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


def transform_tools_to_gemini_from_file(input_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Transform tools from a file to Gemini format and optionally save to output file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save transformed tools
        
    Returns:
        List of transformed tools in Gemini format
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
            transformed_tool = transform_tool_to_gemini(tool)
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


def transform_all_tools_to_gemini_in_directory(input_dir: str, output_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transform all tool files in a directory to Gemini format.
    
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
                output_filename = filename.replace('.json', '_gemini_transformed.json')
                output_file = os.path.join(output_dir, output_filename)
            
            print(f"Processing {filename}...")
            transformed_tools = transform_tools_to_gemini_from_file(input_file, output_file)
            results[filename] = transformed_tools
    
    return results


def main():
    """
    Main function to demonstrate usage.
    """
    # Example usage
    input_directory = "/home/ubuntu/newToolData/public_data/v4/tools"
    output_directory = "/home/ubuntu/newToolData/gemini_transformed_tools"
    
    print("Starting Gemini tool transformation...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # Transform all tools
    results = transform_all_tools_to_gemini_in_directory(input_directory, output_directory)
    
    # Print summary
    total_tools = sum(len(tools) for tools in results.values())
    print(f"\nGemini transformation complete!")
    print(f"Total tools transformed: {total_tools}")
    
    for filename, tools in results.items():
        print(f"  {filename}: {len(tools)} tools")
    
    # Create a combined file with all transformed tools
    all_transformed_tools = []
    for tools in results.values():
        all_transformed_tools.extend(tools)
    
    if all_transformed_tools:
        combined_output = os.path.join(output_directory, "all_gemini_transformed_tools.json")
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_transformed_tools, f, indent=2, ensure_ascii=False)
        print(f"\nCombined file created: {combined_output}")


if __name__ == "__main__":
    main()
