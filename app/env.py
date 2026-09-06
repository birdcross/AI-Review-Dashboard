import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_environment():
    """Load local secrets from .env without overriding existing OS env vars."""
    load_dotenv(PROJECT_ROOT / '.env', override=False)


def get_api_key(config):
    load_environment()
    env_name = config.get('ai', {}).get('api_key_env', 'GEMINI_API_KEY')
    return os.getenv(env_name, '').strip()
