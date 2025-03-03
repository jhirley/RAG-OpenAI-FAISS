import os
import streamlit as st
from dotenv import load_dotenv
from functools import lru_cache

# @lru_cache()
# get the api key from the environment variables or from the streamlit secrets file
def get_api_key(key_name):
    load_dotenv()
    if isinstance(key_name, list):
        keys = {}
        for key in key_name:
            api_key = os.getenv(key)
            if api_key:
                keys[key] = api_key
            else:
                keys[key] = st.secrets[key]
        return keys
    elif isinstance(key_name, str):
        api_key = os.getenv(key_name)
        if api_key:
            return api_key
        else:
            return st.secrets[key_name]

# Load environment variables from .env file
