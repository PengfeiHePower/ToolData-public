import os
import sys

def load_api_keys():
    """Load and validate API keys from .env file. Exit if keys are missing."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
    except ImportError:
        print("ERROR: python-dotenv not installed. Run: pip install python-dotenv")
        sys.exit(1)
    
    # Check required API keys
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not gemini_key or not openai_key:
        print("ERROR: API keys missing from .env file!")
        print("Required: GEMINI_API_KEY, OPENAI_API_KEY")
        sys.exit(1)
    
    print("✅ API keys loaded successfully")
    return gemini_key, openai_key