import os
import json
import requests
import time
import logging

from google import genai
from google.genai import types

import boto3
from botocore.config import Config

import re

import argparse
from copy import deepcopy

parser = argparse.ArgumentParser('evaluate tool usage')
parser.add_argument('-model', type=str, default='gemini-2.5-pro', help='model name', choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
parser.add_argument('-save_dir', type=str, default='/home/ec2-user/mountS3/newToolData/simple_query', help='file-saving directory')
parser.add_argument('-chk_dir', type=str, default='./chk/traj_gen', help='checkpoint directory')
parser.add_argument('-traj_type', type=str, default='parallel', help='trajectory type', choices=['parallel', 'sequential'])
parser.add_argument('-num_query', type=int, default=10, help='maximum number of queries for one task type')
parser.add_argument('-max_retries', type=int, default=3, help='maximum retry attempts per request')
parser.add_argument('-retry_delay', type=float, default=1.0, help='initial retry delay in seconds')
args = parser.parse_args()

# load vllm models (local)
vllm_api_meta = {
    "qwen-8b":{
        "model_name": "qwen-8b-8080",
        "api_url": "http://0.0.0.0:8080"
    },
    "qwen-32b":{
        "model_name": "qwen-32b-8081",
        "api_url": "http://0.0.0.0:8081"
    },
    "qwen-30b-A3B":{
        "model_name": "qwen-30b-A3B-8080",
        "api_url": "http://0.0.0.0:8080"
    }
}

def generate_content_vllm(model_meta, prompt, max_new_tokens=1000, temperature=0.7):
    """
    Generate text using a vLLM API server.

    Args:
        model_meta (str): Model key to lookup API configuration.
        prompt (str): User input for text generation.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.

    Returns:
        str: Generated text from the model.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "model": vllm_api_meta[model_meta]['model_name'],
        "messages": [
            {"role": "system", "content": 'You are a helpful assistant.'},
            {"role": "user", "content": prompt}
        ],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }
    # data["chat_template_kwargs"]={"enable_thinking": False}

    response = requests.post(
        f"{vllm_api_meta[model_meta]['api_url']}/v1/chat/completions", 
        headers=headers, 
        json=data,
        timeout=30  # Add timeout
    )
    
    if response.status_code == 200:
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected response format: {e}")
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")

# load gemini series
def generate_content_gemini(model, prompt):
    # WARNING: API key should be moved to environment variable for security
    api_key = os.getenv('GEMINI_API_KEY', "AIzaSyAvB63pIoUW8bVjVDCCHC-476vDS9Qz00Q")
    client = genai.Client(api_key=api_key)
    if 'gemini-2.5' in model:
        response = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1)
            )
        )
        return response.text
    elif 'gemini-2.0' in model:
        response = client.models.generate_content(
            model=model, contents=prompt
            )
        return response.text

# load bedrock models
config = Config(read_timeout=3600)
bedrock_meta = {
    "claude_v4":"us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude_v37":"us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "nova_pro":"amazon.nova-pro-v1:0",
    "nova_lite":"amazon.nova-lite-v1:0"
}
region = 'us-west-2'
def generate_content_bedrock(model, prompt, max_tokens=8000, temperature=0.3):
    client = boto3.client("bedrock-runtime", region_name=region, config=config)
    model_id = bedrock_meta[model]
    conversation = [
        {
            "role": "user",
            "content": [{"text": prompt}],
            }
            ]
    response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": 0.9},
            )
    response_text = response["output"]["message"]["content"][0]["text"]
    return response_text

# generate content function
def generate_content(model,prompt):
    if model in vllm_api_meta:
        return generate_content_vllm(model, prompt, max_new_tokens=8000, temperature=0.7)
    elif 'gemini' in model:
        return generate_content_gemini(model, prompt)
    elif model in bedrock_meta:
        return generate_content_bedrock(model, prompt, max_tokens=8000, temperature=0.3)

# extract json from markdown fence
def extract_json_from_markdown_fence(text: str):
    """
    Extracts and parses a JSON list of dictionaries from a markdown fenced code block.
    
    Args:
        text (str): The input string containing a markdown block with JSON content.
        
    Returns:
        list[dict]: Parsed JSON data as a list of dictionaries.
        
    Raises:
        ValueError: If no valid JSON block is found or parsing fails.
    """
    # Match fenced code block with json
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON fenced code block found.")
    
    json_str = match.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")

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
domain_list = list(select_cate.keys())
num_domain = len(domain_list)

# Create checkpoint directory if it doesn't exist
os.makedirs(args.chk_dir, exist_ok=True)

# main
with open('/home/ec2-user/mountS3/newToolData/query_gen/simple_query_prompt.json', 'r') as f:
    simple_query_prompt = json.load(f)
# iterate over domains
for domain in domain_list:
    print(f"Generating trajectory for domain: {domain}")
    with open(f'/home/ec2-user/mountS3/newToolData/tools/{domain}/filter_subtool.json', 'r') as f:
        tool_list = json.load(f)
    with open(f'/home/ec2-user/mountS3/newToolData/simple_query/{domain}/task_type.json', 'r') as f:
        task_type = json.load(f)
    keys_to_keep = ['fused name', 'fused description', 'required_parameters', 'optional_parameters']
    # load chk
    chk_file = os.path.join(args.chk_dir, f"simple_traj_{args.traj_type}_{args.model}_{domain}.json")
    chk_index = load_checkpoint(chk_file)
    # load save files - create directory if it doesn't exist
    save_dir = os.path.join(args.save_dir, domain)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"simple_traj_{args.traj_type}_{args.model}.json")
    if chk_index:
        with open(save_file, 'r') as f:
            records = json.load(f)
    else:
        records = []
    # iterate over task types
    for i in range(len(task_type)):
        target_task = task_type[i]
        target_tool_type = target_task['tool classes']
        target_tools = [item for item in tool_list if item['category'] in target_tool_type]
        # initialize checkpoint
        if target_task['task name'] in chk_index:
            start_idx = chk_index[target_task['task name']]
        else:
            start_idx = 0
            chk_index[target_task['task name']] = 0
        # iteratively generate trajectory
        for j in range(start_idx, args.num_query):
            retry_count = 0
            success = False
            
            while retry_count < args.max_retries and not success:
                try:
                    prompt_traj = deepcopy(simple_query_prompt[f'traj_gen_{args.traj_type}'])
                    prompt_traj = prompt_traj.replace("<domain>", domain)
                    prompt_traj = prompt_traj.replace("<task_type>", target_task['task name'])
                    prompt_traj = prompt_traj.replace("<task_description>", target_task['task description'])
                    prompt_traj = prompt_traj.replace("<tool_list>", str(target_tools))
                    
                    response_traj = generate_content(args.model, prompt_traj)
                    response_traj = extract_json_from_markdown_fence(response_traj)
                    results = {'query': response_traj["query"], 'tool list': response_traj["tool list"]}
                    records.append(results)
                    
                    # Save checkpoint on success
                    chk_index[target_task['task name']] += 1
                    save_checkpoint(chk_index, chk_file)
                    success = True
                    print(f"Successfully processed record {j} for task '{target_task['task name']}'")
                    
                except Exception as e:
                    retry_count += 1
                    delay = args.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    print(f"Attempt {retry_count} failed for record {j} (task: {target_task['task name']}): {e}")
                    
                    if retry_count < args.max_retries:
                        print(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"Max retries ({args.max_retries}) reached for record {j}, skipping...")
                        # Still increment checkpoint to avoid getting stuck on this record
                        chk_index[target_task['task name']] += 1
                        save_checkpoint(chk_index, chk_file)
            
            # Save results after each successful record
            with open(save_file, 'w') as f:
                json.dump(records, f, indent=4)

