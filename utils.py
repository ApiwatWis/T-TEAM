# utils.py
import pandas as pd
import streamlit as st
import numpy as np
import os
import requests
import io

DATA_FOLDER = "data"

def get_github_session():
    """Returns a requests session with GitHub authentication."""
    if "github" not in st.secrets:
        return None
    
    token = st.secrets["github"]["token"]
    session = requests.Session()
    session.headers.update({"Authorization": f"token {token}"})
    return session

def get_github_repo_info():
    if "github" not in st.secrets:
        return None, None, None
    s = st.secrets["github"]
    return s["username"], s["repo_name"], s.get("branch", "main")

def list_github_dirs(path=""):
    """
    Lists directories in the GitHub repo at the given path.
    Returns a list of directory names.
    """
    username, repo, _ = get_github_repo_info()
    if not username:
        # Fallback to local if secrets missing
        local_path = os.path.join(DATA_FOLDER, path)
        if os.path.exists(local_path):
             return [d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))]
        return []

    # API: https://api.github.com/repos/{owner}/{repo}/contents/{path}
    url = f"https://api.github.com/repos/{username}/{repo}/contents/{path}"
    session = get_github_session()
    try:
        response = session.get(url)
        if response.status_code == 200:
            items = response.json()
            # Filter for directories
            return [item['name'] for item in items if item['type'] == 'dir']
    except Exception as e:
        print(f"Error listing github dirs: {e}")
    return []

def list_github_files(path):
    """Lists files in a GitHub repo directory."""
    username, repo, _ = get_github_repo_info()
    if not username:
        # Fallback local
        full_path = os.path.join(DATA_FOLDER, path)
        if os.path.exists(full_path):
             return os.listdir(full_path)
        return []

    url = f"https://api.github.com/repos/{username}/{repo}/contents/{path}"
    session = get_github_session()
    try:
        response = session.get(url)
        if response.status_code == 200:
            items = response.json()
            return [item['name'] for item in items if item['type'] == 'file']
    except:
        pass
    return []

def read_github_file(path):
    """Reads a file from GitHub and returns content as string."""
    username, repo, branch = get_github_repo_info()
    if not username:
        # Fallback local
        full_path = os.path.join(DATA_FOLDER, path)
        if os.path.exists(full_path):
             with open(full_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read()
        return None

    # Raw URL
    # Note: path should be relative to repo root
    url = f"https://raw.githubusercontent.com/{username}/{repo}/{branch}/{path}"
    session = get_github_session()
    try:
        response = session.get(url)
        if response.status_code == 200:
            return response.text
    except:
        pass
    return None

def read_github_file_binary(path):
    """Reads a binary file from GitHub and returns bytes."""
    username, repo, branch = get_github_repo_info()
    if not username:
        full_path = os.path.join(DATA_FOLDER, path)
        if os.path.exists(full_path):
             with open(full_path, 'rb') as f: return f.read()
        return None

    url = f"https://raw.githubusercontent.com/{username}/{repo}/{branch}/{path}"
    session = get_github_session()
    try:
        response = session.get(url)
        if response.status_code == 200:
            return response.content
    except:
        pass
    return None

def github_file_exists(path):
    username, repo, _ = get_github_repo_info()
    if not username:
         return os.path.exists(os.path.join(DATA_FOLDER, path))

    # Use contents API to check existence
    url = f"https://api.github.com/repos/{username}/{repo}/contents/{path}"
    session = get_github_session()
    try:
        response = session.get(url)
        return response.status_code == 200
    except:
        return False

@st.cache_data
def get_data_root():
    """Determines if shots are in root or 'data' folder of the repo."""
    dirs_root = list_github_dirs("")
    if any(d.isdigit() for d in dirs_root):
        return ""
    
    dirs_data = list_github_dirs("data")
    if any(d.isdigit() for d in dirs_data):
        return "data"
        
    return "" # Default to root


@st.cache_data
def load_timeseries(shot_id):
    """Loads standard time-series data (Ip, Bt, Loop Voltage)"""
    path = os.path.join(DATA_FOLDER, f"shot_{shot_id}.txt")
    if os.path.exists(path):
        return pd.read_csv(path, sep='\s+', comment='#')
    return None

@st.cache_data
def load_hxr_data(shot_id):
    """
    Loads HXR data. 
    Assumed format: rows = events, columns = [Time, Energy_keV]
    """
    path = os.path.join(DATA_FOLDER, f"shot_{shot_id}_hxr.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def get_video_path(shot_id):
    """Returns path to video file if it exists"""
    path = os.path.join(DATA_FOLDER, f"shot_{shot_id}_cam.mp4")
    if os.path.exists(path):
        return path
    return None