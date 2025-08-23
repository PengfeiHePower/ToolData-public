"""Formatting utilities for LLM prompts and tool outputs."""
from typing import List, Dict, Any

def format_tool_results(tool_list: List[Dict[str, Any]]) -> str:
    """Format tool results into a readable block suitable for LLM prompts.

    Mirrors the behavior used in simple_query/gt_tool_exe.py so that other
    modules can reuse the same formatting without duplication.
    """
    formatted_results: List[str] = []
    for i, tool in enumerate(tool_list, 1):
        tool_name = tool.get('tool name', 'Unknown Tool')
        tool_desc = tool.get('tool description', 'No description')
        tool_required = tool.get('required parameters', [])
        tool_optional = tool.get('optional parameters', [])
        exe_output = tool.get('executed_output', 'No output')
        exe_output_str = str(exe_output)
        # Keep it reasonably bounded for prompts
        if len(exe_output_str) > 4000:
            exe_output_str = exe_output_str[:4000] + '...'

        formatted_results.append(f"{i}. Tool: {tool_name}")
        formatted_results.append(f"   Description: {tool_desc}")
        formatted_results.append(f"   Required parameters: {str(tool_required)}")
        formatted_results.append(f"   Optional parameters: {str(tool_optional)}")
        formatted_results.append(f"   Output: {exe_output_str}")
        formatted_results.append("")  # spacer line

    return "\n".join(formatted_results)
