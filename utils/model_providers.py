"""
Centralized model providers for all generation tasks.
Contains all model generation functions and routing logic.
"""

import os
import requests
import json
import re
import time
import logging
from typing import Optional
from functools import wraps

# Load environment variables
try:
    from dotenv import load_dotenv
    # .env is in the parent directory (newToolData/)
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
    print(f"Loading .env from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

# Validate API keys
gemini_key = os.getenv('GEMINI_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')
deepseek_key = os.getenv('DEEPSEEK_API')
moonshot_key = os.getenv('MOONSHOT_API_KEY')

if not gemini_key or not openai_key or not deepseek_key or not moonshot_key:
    print("❌ ERROR: API keys missing from .env file!")
    print("Required: GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, MOONSHOT_API_KEY")
    import sys
    sys.exit(1)

print("✅ Model providers initialized successfully")

# Utility functions for text processing and JSON extraction
def sanitize_input(text: str) -> str:
    """Sanitize input text to prevent injection attacks."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    # Remove or escape potentially dangerous characters
    text = text.replace('\x00', '')  # Remove null bytes
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Remove control chars
    return text

def extract_json_from_markdown_fence(text: str):
    """
    Extract JSON object/array from a string that may include markdown fences
    or noisy prefixes like "Response:" and timestamps. Uses a bracket-balancing
    scan to avoid partial matches.

    Returns the parsed JSON (dict or list) or raises ValueError if none found.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")

    text = sanitize_input(text)

    # Quick paths: try fenced code blocks first
    fence_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]
    for pattern in fence_patterns:
        m = re.search(pattern, text)
        if m:
            candidate = m.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # fall through to robust scanner
                pass

    # Remove common noisy prefixes on lines (e.g., "Response:")
    cleaned = re.sub(r"^\s*(Response:|Output:|Answer:)\s*", "", text.strip(), flags=re.IGNORECASE)

    # Bracket-balanced scan from first '{' or '['
    start_idx = None
    for idx, ch in enumerate(cleaned):
        if ch in '{[':
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError("No valid JSON found in text")

    stack = []
    in_string = False
    escape = False
    for j in range(start_idx, len(cleaned)):
        c = cleaned[j]
        if in_string:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_string = False
            continue
        else:
            if c == '"':
                in_string = True
                continue
            if c in '{[':
                stack.append(c)
            elif c in '}]':
                if not stack:
                    # unmatched closing, ignore
                    continue
                opening = stack.pop()
                if (opening == '{' and c != '}') or (opening == '[' and c != ']'):
                    # mismatched, keep scanning
                    continue
                if not stack:
                    # Found a balanced JSON substring
                    candidate = cleaned[start_idx:j+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try to continue in case there is another JSON later
                        # Seek next '{' or '[' after start_idx+1
                        next_start = None
                        for k in range(start_idx + 1, len(cleaned)):
                            if cleaned[k] in '{[':
                                next_start = k
                                break
                        if next_start is None:
                            break
                        start_idx = next_start
                        stack = []
                        in_string = False
                        escape = False
                        # restart outer loop from new start
                        j = start_idx
                        continue

    raise ValueError("No valid JSON found in text")

# Custom exceptions for robust error handling
class APIError(Exception):
    """Base exception for API-related errors"""
    pass

class ModelNotAvailableError(APIError):
    """Raised when a model is not available"""
    pass

class InvalidResponseError(APIError):
    """Raised when API returns invalid response"""
    pass

class RateLimitError(APIError):
    """Raised when API rate limit is exceeded"""
    pass

# Centralized retry decorator
def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logging.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                except (APIError, requests.RequestException) as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logging.warning(f"API error, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

# Model configurations
vllm_api_meta = {
    "qwen-8b": {
        "model_name": "qwen-8b-8080",
        "api_url": "http://0.0.0.0:8080"
    },
    "qwen-32b": {
        "model_name": "qwen-32b-8081",
        "api_url": "http://0.0.0.0:8081"
    },
    "qwen-30b-A3B": {
        "model_name": "qwen-30b-A3B-8080",
        "api_url": "http://0.0.0.0:8080"
    },
    "qwen-235b-A22B": {
        "model_name": "qwen-235b-a22b-fp8",
        "api_url": "http://0.0.0.0:2507"
    }
}

bedrock_meta = {
    "claude_v4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude_v37": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "nova_pro": "amazon.nova-pro-v1:0",
    "nova_lite": "amazon.nova-lite-v1:0",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0"
}

gemini_models = {
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", 
    "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash", 
    "gemini-1.5-flash-8b", "gemini-1.0-pro"
}

openai_models = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-4-32k",
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct",
    "o1-preview", "o1-mini", "o4-mini", "gpt-5-mini"
}

deepseek_models = {
    "deepseek-chat", "deepseek-reasoner"
}

ollama_models = {
    "gpt-oss:20b", "gpt-oss:120b"
}

moonshot_models = {
    "kimi-k2": "kimi-k2-0905-preview"
}

# Basic generation functions (without retry)
def generate_content_vllm(model_meta: str, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.7, stop: Optional[list] = None) -> str:
    """Generate content using vLLM API."""
    if model_meta not in vllm_api_meta:
        raise ValueError(f"Model {model_meta} not found in vllm_api_meta")
    
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
    
    # Add stop tokens if provided and not empty
    if stop is not None and len(stop) > 0:
        data["stop"] = stop

    try:
        response = requests.post(
            f"{vllm_api_meta[model_meta]['api_url']}/v1/chat/completions", 
            headers=headers, 
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"vLLM API Error {response.status_code}: {response.text}")
    except Exception as e:
        # Convert to appropriate exception types for retry logic
        if "rate limit" in str(e).lower() or "429" in str(e):
            raise RateLimitError(str(e))
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise APIError(str(e))
        else:
            raise e

def generate_content_gemini(model: str, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.7, stop: Optional[list] = None) -> str:
    """Generate content using Gemini API."""
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=gemini_key)
        
        if 'gemini-2.5' in model:
            config_params = {
                "thinking_config": types.ThinkingConfig(thinking_budget=-1),
                "temperature": temperature,
                "maxOutputTokens": max_new_tokens
            }
            if stop is not None and len(stop) > 0:
                config_params["stopSequences"] = stop
            
            response = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(**config_params)
            )
            return response.text
        elif 'gemini-2.0' in model or 'gemini-1.' in model:
            config_params = {
                "temperature": temperature,
                "maxOutputTokens": max_new_tokens
            }
            if stop is not None and len(stop) > 0:
                config_params["stopSequences"] = stop
            
            response = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(**config_params)
            )
            return response.text
        else:
            raise ValueError(f"Unsupported Gemini model: {model}")
    except Exception as e:
        # Convert to appropriate exception types for retry logic
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            raise RateLimitError(str(e))
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise APIError(str(e))
        else:
            raise e

def generate_content_bedrock(model: str, prompt: str, max_tokens: int = 8000, temperature: float = 0.3, stop: Optional[list] = None) -> str:
    """Generate content using AWS Bedrock."""
    try:
        import boto3
        from botocore.config import Config
        
        config = Config(read_timeout=3600)
        region = 'us-west-2'
        
        client = boto3.client("bedrock-runtime", region_name=region, config=config)
        model_id = bedrock_meta[model]
        
        conversation = [{
            "role": "user",
            "content": [{"text": prompt}]
        }]
        
        inference_config = {
            "maxTokens": max_tokens, 
            "temperature": temperature, 
            "topP": 0.9
        }
        
        # Only add stopSequences if stop is not None and not empty
        if stop is not None and len(stop) > 0:
            inference_config['stopSequences'] = stop
        
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig=inference_config,
        )

        if "gpt" in model:
            return response["output"]["message"]["content"][1]['text']
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        # Convert to appropriate exception types for retry logic
        if "throttling" in str(e).lower() or "rate" in str(e).lower():
            raise RateLimitError(str(e))
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise APIError(str(e))
        else:
            raise e

def generate_content_openai(model: str, prompt: str, max_tokens: int = 8000, temperature: float = 1.0, stop: Optional[list] = None) -> str:
    """Generate content using OpenAI API."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=openai_key)
        
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Only add stop if provided and not empty
        if stop is not None and len(stop) > 0:
            params["stop"] = stop
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        # Convert to appropriate exception types for retry logic
        if "rate_limit" in str(e).lower() or "429" in str(e):
            raise RateLimitError(str(e))
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise APIError(str(e))
        else:
            raise e

def generate_content_ollama(model: str, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, stop: Optional[list] = None):
    """Generate content using Ollama API."""
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:11434/v1",  # Ollama-compatible OpenAI API
            api_key="ollama"  # Dummy key
        )

        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Only add stop if provided and not empty
        if stop is not None and len(stop) > 0:
            params["stop"] = stop
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        # Convert to appropriate exception types for retry logic
        if "rate limit" in str(e).lower() or "429" in str(e):
            raise RateLimitError(str(e))
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise APIError(str(e))
        else:
            raise e
        
def generate_content_deepseek(model: str, prompt: str, max_tokens: int = 8000, temperature: float = 0.7, stop: Optional[list] = None):
    """generate content with DeepSeek API"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
        
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        # Only add stop if provided and not empty
        if stop is not None and len(stop) > 0:
            params["stop"] = stop
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        # Convert to appropriate exception types for retry logic
        if "rate_limit" in str(e).lower() or "429" in str(e):
            raise RateLimitError(str(e))
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise APIError(str(e))
        else:
            raise e

def generate_content_moonshot(model: str, prompt: str, max_tokens: int = 8000, temperature: float = 0.6, stop: Optional[list] = None) -> str:
    """Generate content using Moonshot API."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=moonshot_key, base_url="https://api.moonshot.ai/v1")
        
        params = {
            "model": moonshot_models[model],
            "messages": [{"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
                {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        # Only add stop if provided and not empty
        if stop is not None and len(stop) > 0:
            params["stop"] = stop
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            raise RateLimitError(str(e))
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise APIError(str(e))
        else:
            raise e


# Retry-enabled wrapper functions
@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_content_deepseek_with_retry(model, prompt, max_tokens=8000, temperature=0.7, stop=None):
    """Wrapper with retry logic for DeepSeek generation."""
    return generate_content_deepseek(model, prompt, max_tokens, temperature, stop)

@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_content_vllm_with_retry(model_meta, prompt, max_new_tokens=1000, temperature=0.7, stop=None):
    """Wrapper with retry logic for vLLM generation."""
    return generate_content_vllm(model_meta, prompt, max_new_tokens, temperature, stop)

@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_content_gemini_with_retry(model, prompt, max_new_tokens=1000, temperature=0.7, stop=None):
    """Wrapper with retry logic for Gemini generation."""
    return generate_content_gemini(model, prompt, max_new_tokens, temperature, stop)

@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_content_bedrock_with_retry(model, prompt, max_tokens=8000, temperature=0.3, stop=None):
    """Wrapper with retry logic for Bedrock generation."""
    return generate_content_bedrock(model, prompt, max_tokens, temperature, stop)

@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_content_openai_with_retry(model, prompt, max_tokens=8000, temperature=0.7, stop=None):
    """Wrapper with retry logic for OpenAI generation."""
    return generate_content_openai(model, prompt, max_tokens, temperature, stop)

@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_content_ollama_with_retry(model, prompt, max_new_tokens=1000, temperature=0.7, stop=None):
    """Wrapper with retry logic for Ollama generation."""
    return generate_content_ollama(model, prompt, max_new_tokens, temperature, stop)

@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_content_moonshot_with_retry(model, prompt, max_tokens=8000, temperature=0.6, stop=None):
    """Wrapper with retry logic for Moonshot generation."""
    return generate_content_moonshot(model, prompt, max_tokens, temperature, stop)

def generate_content(model: str, prompt: str, temperature: Optional[float] = None) -> str:
    """
    Main function to generate content using any supported model.
    
    Args:
        model: Model name
        prompt: Input prompt
        temperature: Optional temperature override
        
    Returns:
        Generated content string
        
    Raises:
        ValueError: If model is not supported
    """
    if model in vllm_api_meta:
        temp = temperature if temperature is not None else 0.7
        return generate_content_vllm(model, prompt, max_new_tokens=8000, temperature=temp)
    elif model in gemini_models:
        return generate_content_gemini(model, prompt)
    elif model in bedrock_meta:
        temp = temperature if temperature is not None else 0.3
        return generate_content_bedrock(model, prompt, max_tokens=8000, temperature=temp)
    elif model in openai_models:
        return generate_content_openai(model, prompt)
    elif model in deepseek_models:
        temp = temperature if temperature is not None else 0.7
        return generate_content_deepseek(model, prompt, max_tokens=8000, temperature=temp)
    elif model in ollama_models:
        temp = temperature if temperature is not None else 0.7
        return generate_content_ollama(model, prompt, max_new_tokens=8000, temperature=temp)
    elif model in moonshot_models:
        temp = temperature if temperature is not None else 0.6
        return generate_content_moonshot(model, prompt, max_tokens=8000, temperature=temp)
    else:
        raise ModelNotAvailableError(f"Model {model} is not supported")

def generate_content_with_retry(model: str, prompt: str, temperature: Optional[float] = None, stop: Optional[list] = None) -> str:
    """
    Generate content with automatic retry logic for any supported model.
    
    Args:
        model: Model name
        prompt: Input prompt
        temperature: Optional temperature override
        stop: Optional list of stop tokens to halt generation
        
    Returns:
        Generated content string
        
    Raises:
        ModelNotAvailableError: If model is not supported
        APIError: If API calls fail after retries
    """
    if model in vllm_api_meta:
        temp = temperature if temperature is not None else 0.7
        return generate_content_vllm_with_retry(model, prompt, max_new_tokens=8000, temperature=temp, stop=stop)
    elif model in gemini_models:
        temp = temperature if temperature is not None else 0.7
        return generate_content_gemini_with_retry(model, prompt, max_new_tokens=8000, temperature=temp, stop=stop)
    elif model in bedrock_meta:
        temp = temperature if temperature is not None else 0.3
        return generate_content_bedrock_with_retry(model, prompt, max_tokens=8000, temperature=temp, stop=stop)
    elif model in openai_models:
        temp = temperature if temperature is not None else 0.7
        return generate_content_openai_with_retry(model, prompt, max_tokens=8000, temperature=1.0, stop=stop)
    elif model in deepseek_models:
        temp = temperature if temperature is not None else 0.7
        return generate_content_deepseek_with_retry(model, prompt, max_tokens=8000, temperature=temp, stop=stop)
    elif model in ollama_models:
        temp = temperature if temperature is not None else 0.7
        return generate_content_ollama_with_retry(model, prompt, max_new_tokens=8000, temperature=temp, stop=stop)
    elif model in moonshot_models:
        temp = temperature if temperature is not None else 0.6
        return generate_content_moonshot_with_retry(model, prompt, max_tokens=8000, temperature=temp, stop=stop)
    else:
        raise ModelNotAvailableError(f"Model {model} is not supported")

def get_supported_models():
    """Return all supported models grouped by provider."""
    return {
        "vllm": list(vllm_api_meta.keys()),
        "gemini": list(gemini_models),
        "bedrock": list(bedrock_meta.keys()),
        "openai": list(openai_models),
        "deepseek": list(deepseek_models),
        "ollama": list(ollama_models),
        "moonshot": list(moonshot_models)
    }

# Export model configurations for files that need them
__all__ = [
    # Core generation functions
    'generate_content',
    'generate_content_with_retry',
    
    # Individual provider functions (basic)
    'generate_content_vllm', 
    'generate_content_gemini',
    'generate_content_bedrock', 
    'generate_content_openai',
    'generate_content_ollama',
    'generate_content_deepseek',
    'generate_content_moonshot',
    
    # Individual provider functions (with retry)
    'generate_content_vllm_with_retry',
    'generate_content_gemini_with_retry',
    'generate_content_bedrock_with_retry',
    'generate_content_openai_with_retry',
    'generate_content_ollama_with_retry',
    'generate_content_deepseek_with_retry',
    'generate_content_moonshot_with_retry',
    
    # Utility functions
    'get_supported_models',
    'sanitize_input',
    'extract_json_from_markdown_fence',
    'retry_with_backoff',
    
    # Exception classes
    'APIError',
    'ModelNotAvailableError', 
    'InvalidResponseError',
    'RateLimitError',
    
    # Model metadata
    'vllm_api_meta',
    'bedrock_meta', 
    'gemini_models',
    'openai_models',
    'deepseek_models',
    'ollama_models',
    'moonshot_models'
]
