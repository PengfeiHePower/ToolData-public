import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from load_env import load_api_keys

# Test API key loading
gemini_key, openai_key = load_api_keys()

# Print keys for validation (first 10 and last 4 chars)
print(f"GEMINI_API_KEY: {gemini_key[:10]}...{gemini_key[-4:]}")
print(f"OPENAI_API_KEY: {openai_key[:10]}...{openai_key[-4:]}")
print("🎉 Ready to use APIs!")
