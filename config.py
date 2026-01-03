import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_secret(key, section="general"):
    """Get secret from environment variable or streamlit secrets."""
    # Priority 1: Environment Variable
    if key in os.environ:
        return os.environ[key]
    
    # Priority 2: Streamlit Secrets
    try:
        if section in st.secrets:
            return st.secrets[section].get(key)
        # Check root level
        return st.secrets.get(key)
    except (FileNotFoundError, KeyError):
        return None

# --- Configuration ---
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "general")

NEO4J_URI = get_secret("NEO4J_URI", "neo4j")
NEO4J_USERNAME = get_secret("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = get_secret("NEO4J_DATABASE", "neo4j") or "neo4j"

# Constants
LLM_MODEL = "gpt-5.2-2025-12-11"  # Or gpt-3.5-turbo if cost is concern, but 4o is better for extraction
