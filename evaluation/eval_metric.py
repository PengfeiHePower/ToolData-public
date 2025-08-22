"""
Compare predicted tool-calling trajectory and gt trajectories to get metrics
"""

import ast
import re
from datetime import datetime
import json
import os

def safe_eval(value):
    try:
        result = ast.literal_eval(str(value))
        # Handle numeric string comparisons
        if isinstance(result, str) and result.isdigit():
            return int(result)
        return result
    except Exception:
        return str(value).strip()

def normalize_numeric(value):
    """Convert numeric strings to appropriate numeric types"""
    if isinstance(value, (int, float)):
        return value
    
    if isinstance(value, str):
        value = value.strip()
        # Try integer conversion
        if value.isdigit():
            return int(value)
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
    
    return value

def normalize_whitespace(value):
    """Normalize whitespace in strings"""
    if not isinstance(value, str):
        return value
    
    # Remove extra spaces around commas: "Rome, Italy" -> "Rome,Italy"
    result = re.sub(r'\s*,\s*', ',', value)
    # Normalize multiple spaces to single space
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def normalize_location(location_str):
    """Normalize location strings to handle country code variations"""
    if not isinstance(location_str, str) or ',' not in location_str:
        return location_str
    
    # Common country mappings
    country_mappings = {
        'FR': 'France',
        'US': 'United States', 'USA': 'United States',
        'UK': 'United Kingdom', 'GB': 'United Kingdom',
        'DE': 'Germany', 'Deutschland': 'Germany',
        'IT': 'Italy', 'Italia': 'Italy',
        'ES': 'Spain', 'España': 'Spain',
        'JP': 'Japan',
        'CN': 'China',
        'AU': 'Australia',
        'CA': 'Canada',
        'IN': 'India',
        'BR': 'Brazil',
        'MX': 'Mexico',
        'RU': 'Russia',
        'KR': 'South Korea',
        'NL': 'Netherlands',
        'SE': 'Sweden',
        'NO': 'Norway',
        'DK': 'Denmark',
        'FI': 'Finland',
        'CH': 'Switzerland',
        'AT': 'Austria',
        'BE': 'Belgium',
        'PT': 'Portugal',
        'GR': 'Greece',
        'TR': 'Turkey',
        'PL': 'Poland',
        'CZ': 'Czech Republic',
        'HU': 'Hungary',
        'RO': 'Romania',
        'BG': 'Bulgaria',
        'HR': 'Croatia',
        'SI': 'Slovenia',
        'SK': 'Slovakia',
        'LT': 'Lithuania',
        'LV': 'Latvia',
        'EE': 'Estonia'
    }
    
    # Split by comma and process parts
    parts = [part.strip() for part in location_str.split(',')]
    normalized_parts = []
    
    for part in parts:
        # Check if this part is a country code that should be expanded
        if part.upper() in country_mappings:
            normalized_parts.append(country_mappings[part.upper()])
        else:
            # Check if this part is a full country name that has a common code
            part_lower = part.lower()
            for code, full_name in country_mappings.items():
                if full_name.lower() == part_lower:
                    normalized_parts.append(full_name)
                    break
            else:
                normalized_parts.append(part)
    
    return ','.join(normalized_parts)

def normalize_date(value):
    """Normalize date strings to YYYY-MM-DD format"""
    if not isinstance(value, str):
        return value
    
    # Check if it matches date pattern
    if re.match(r'\d{4}-\d{1,2}-\d{1,2}', value):
        try:
            # Parse and reformat to ensure consistency
            dt = datetime.strptime(value, '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except:
            pass
    
    return value

def normalize_boolean(value):
    """Normalize boolean-like strings"""
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ['true', '1', 'yes', 'on']:
            return True
        elif value_lower in ['false', '0', 'no', 'off']:
            return False
    
    return value

def normalize_value(value):
    """Enhanced value normalization using specialized normalization functions"""
    if value is None:
        return None
    
    # Convert to string first for initial processing
    str_val = str(value).strip()
    
    # Try ast.literal_eval first for safe evaluation
    try:
        result = ast.literal_eval(str_val)
    except:
        result = str_val
    
    # Apply normalization functions in sequence
    result = normalize_whitespace(result)  # Clean up whitespace first
    result = normalize_location(result)    # Handle location codes
    result = normalize_date(result)        # Handle date formats
    result = normalize_boolean(result)     # Handle boolean values
    result = normalize_numeric(result)     # Convert numeric strings last
    
    return result

def is_empty(value):
    return value is None or (isinstance(value, str) and value.strip() == '')

def normalize_param_list(param_list):
    """Normalize a list of {'name': ..., 'value': ...} dicts for comparison."""
    normalized = []
    for p in param_list:
        if is_empty(p.get('value')):  # Skip parameters with empty values
            continue
        name = p['name'].strip()
        value = normalize_value(p['value'])
        normalized.append((name, value))
    # Sort by name so order doesn't affect comparison
    return sorted(normalized, key=lambda x: x[0])

def compare_tool_parameters(pred, gt, debug=False):
    # Add default empty lists to handle missing keys gracefully
    pred_req = pred.get("required_parameters") or pred.get("required parameters") or []
    gt_req = gt.get("required_parameters") or gt.get("required parameters") or []
    
    pred_opt = pred.get("optional_parameters") or pred.get("optional parameters") or []
    gt_opt = gt.get("optional_parameters") or gt.get("optional parameters") or []
    
    pred_required = normalize_param_list(pred_req)
    gt_required = normalize_param_list(gt_req)
    pred_optional = normalize_param_list(pred_opt)
    gt_optional = normalize_param_list(gt_opt)
    
    required_match = pred_required == gt_required
    optional_match = pred_optional == gt_optional
    
    if debug and not (required_match and optional_match):
        print(f"Required mismatch: {pred_required} vs {gt_required}")
        print(f"Optional mismatch: {pred_optional} vs {gt_optional}")
    
    return required_match and optional_match

def load_checkpoint(chk_file: str) -> int:
    """Load checkpoint index from file."""
    try:
        with open(chk_file, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0  # no checkpoint file or invalid content, start from beginning

def save_checkpoint(index: int, chk_file: str) -> None:
    """Save checkpoint index to file."""
    os.makedirs(os.path.dirname(chk_file), exist_ok=True)
    with open(chk_file, "w") as f:
        f.write(str(index))


model = 'gemini-2.5-pro'
select = 'domain'
gen_model = 'claude_v37'
type = 'simple'
gen_model_traj = 'gemini-2.5-pro'
gen_model_query = 'claude_v37'
query_name = 'simple_traj_parallel'

# main
with open('/home/ec2-user/agent-tool/newToolData/selected_category.json', 'r') as f:
    select_cate = json.load(f)
domain_list = list(select_cate.keys())

# for domain in domain_list:
domain = domain_list[2]
# load original query
with open(f'/home/ec2-user/agent-tool/newToolData/simple_query/{domain}/{query_name}_{gen_model_traj}.json', 'r') as f:
# with open(f'/home/ec2-user/agent-tool/newToolData/evaluation/log/simple_query/formal/{query_name}_{gen_model_traj}_{gen_model_query}_{type}_{model}_{select}_{domain}.json', 'r') as f:
    all_query = json.load(f)
# load predictions
with open(f'/home/ec2-user/agent-tool/newToolData/evaluation/log/simple_query/formal/{query_name}_{gen_model_traj}_{gen_model_query}_{type}_{model}_{select}_{domain}.json', 'r') as f:
    pred = json.load(f)
correct_tool = []
inclusion_tool = []
tool_usage = []
print(f"Evaluation setups\n model: {model}, select: {select}, domain: {domain}, gen_model_traj: {gen_model_traj}, gen_model_query: {gen_model_query}, type: {type}")
# print(f"total query: {len(all_query)}")
for idx in range(len(all_query)):
    # print(idx)
    gt_tool_list = all_query[idx]['tool list']
    # gt_tool_list = all_query[idx]['gt tool list']
    pred_tool_list = pred[idx]['tool list']
    gt_tool_name = [item['tool name'] for item in gt_tool_list]
    pred_tool_name = [item['tool name'] for item in pred_tool_list]
    correct_tool.append(set(gt_tool_name) == set(pred_tool_name))
    inclusion_tool.append(len(set(gt_tool_name) & set(pred_tool_name))/max(1,len(gt_tool_name)))
    included = list(set(gt_tool_name) & set(pred_tool_name))
    for item in included:
        gt_target = next((p for p in gt_tool_list if p["tool name"] == item), None)
        pred_target = next((p for p in pred_tool_list if p["tool name"] == item), None)
        tool_usage.append(compare_tool_parameters(pred_target, gt_target))
print(f"correct tool: {sum(correct_tool)/len(correct_tool):.2f}")
print(f"inclusion tool: {sum(inclusion_tool)/max(1,len(inclusion_tool)):.2f}")
print(f'tool usage:{sum(tool_usage)/max(1,len(tool_usage)):.2f}')

