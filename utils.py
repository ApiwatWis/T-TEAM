import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==========================================
# Authentication
# ==========================================

def check_auth():
    """
    Checks if the user is authenticated.
    Returns True if authenticated, False otherwise.
    Displays a password input if not authenticated.
    """
    if st.session_state.get('password_correct', False):
        return True

    # Show input
    if os.path.exists("assets/T-TEAM_Banner_03B.png"):
        st.image("assets/T-TEAM_Banner_03B.png", width="stretch")
    st.header("🔒 Login Required")
    
    pwd = st.text_input("Enter Access Password:", type="password")
    
    if pwd:
        correct_password = None
        # Check root level
        if "school_password" in st.secrets:
            correct_password = st.secrets["school_password"]
        elif "password" in st.secrets:
            correct_password = st.secrets["password"]  
        # Check inside [github] block (legacy)
        elif "github" in st.secrets:
             if "school_password" in st.secrets["github"]:
                correct_password = st.secrets["github"]["school_password"]
             elif "password" in st.secrets["github"]:
                 correct_password = st.secrets["github"]["password"]

        if correct_password is None:
             st.error("Password not configured in secrets.")
             return False

        if pwd == correct_password:
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("❌ Incorrect Password")
            
    return False

# ==========================================
# Google Drive Integration
# ==========================================

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
ROOT_FOLDER_NAME = "2601_T-TEAM-Data"

@st.cache_resource
def get_drive_service():
    if "gcp_service_account" not in st.secrets:
        # Fallback to local check? Or just fail silently until used
        return None
    try:
        service_account_info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=SCOPES
        )
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Failed to create Drive service: {e}")
        return None

@st.cache_data(ttl=3600)
def get_root_id():
    service = get_drive_service()
    if not service: return None
    q = f"name = '{ROOT_FOLDER_NAME}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get('files', [])
    if files:
        return files[0]['id']
    return None

@st.cache_data(ttl=300)
def get_id_from_path(path):
    """
    Resolves a path string (e.g. 'data/1001/IP1.txt') to a Drive File ID.
    Access starts at 2601_T-TEAM-Data.
    """
    root_id = get_root_id()
    if not root_id: return None
    
    if not path or path == "." or path == "/":
        return root_id

    # Normalize path
    parts = path.replace('\\', '/').strip('/').split('/')
    current_id = root_id
    
    service = get_drive_service()
    for part in parts:
        if not part: continue
        q = f"'{current_id}' in parents and name = '{part}' and trashed = false"
        res = service.files().list(q=q, fields="files(id)").execute()
        files = res.get('files', [])
        if not files:
            return None
        current_id = files[0]['id']
        
    return current_id

@st.cache_data(ttl=300)
def list_dirs(path=""):
    """Lists directories in the Drive folder at path (relative to root)."""
    folder_id = get_id_from_path(path)
    if not folder_id: return []

    service = get_drive_service()
    q = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    res = service.files().list(q=q, fields="files(name)").execute()
    return [f['name'] for f in res.get('files', [])]

@st.cache_data(ttl=300)
def list_files(path=""):
    """Lists files (not folders) in the Drive folder at path."""
    folder_id = get_id_from_path(path)
    if not folder_id: return []

    service = get_drive_service()
    q = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    res = service.files().list(q=q, fields="files(name)").execute()
    return [f['name'] for f in res.get('files', [])]

@st.cache_data(ttl=600)
def read_file(path):
    """Reads a file from Drive and returns content as string."""
    file_id = get_id_from_path(path)
    if not file_id: return None

    service = get_drive_service()
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

@st.cache_data(ttl=600)
def read_file_binary(path):
    """Reads a binary file from Drive."""
    file_id = get_id_from_path(path)
    if not file_id: return None

    service = get_drive_service()
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return fh.getvalue()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def file_exists(path):
    return get_id_from_path(path) is not None

# ==========================================
# Domain Specific Helpers
# ==========================================

@st.cache_data
def get_data_root():
    """
    Determines if shots are in root or 'data' folder.
    Prioritizes 'data' folder if it exists.
    """
    try:
        # Check if 'data' folder exists
        if file_exists("data"):
            # If it has numeric folders OR shot files, use it.
            # Also just default to 'data' if it exists to keep workspace clean
            return "data"
        
        # Fallback to root if data folder doesn't exist
        return ""
    except:
        pass
    
    return "data" # Optimistic default

@st.cache_data
def get_shot_list():
    """Returns a sorted list of available shot numbers."""
    root_loc = get_data_root()
    dirs = list_dirs(root_loc)
    shots = [d for d in dirs if d.isdigit()]
    if not shots:
        return ["1001"] # Placeholder
    return sorted(shots, key=lambda x: int(x), reverse=True)

@st.cache_data
def load_timeseries(shot_id):
    """Loads standard time-series data"""
    root = get_data_root()
    path = f"{root}/shot_{shot_id}.txt" if root else f"shot_{shot_id}.txt"
    
    content = read_file(path)
    if content:
         return pd.read_csv(io.StringIO(content), sep=r'\s+', comment='#')
    
    # Fallback: maybe inside the shot folder?
    path_inner = f"{root}/{shot_id}/shot_{shot_id}.txt" if root else f"{shot_id}/shot_{shot_id}.txt"
    content_inner = read_file(path_inner)
    if content_inner:
        return pd.read_csv(io.StringIO(content_inner), sep=r'\s+', comment='#')
        
    return None

@st.cache_data
def load_hxr_data(shot_id):
    root = get_data_root()
    path = f"{root}/shot_{shot_id}_hxr.csv"
    if not root: path = f"shot_{shot_id}_hxr.csv"
    
    content = read_file(path)
    if content:
        return pd.read_csv(io.StringIO(content))
    return None

def get_video_bytes(shot_id):
    root = get_data_root()
    path = f"{root}/shot_{shot_id}_cam.mp4"
    if not root: path = f"shot_{shot_id}_cam.mp4"
    return read_file_binary(path)

# Aliases for compatibility
read_github_file = read_file
read_github_file_binary = read_file_binary
list_github_dirs = list_dirs
list_github_files = list_files
get_github_shot_list = get_shot_list
github_file_exists = file_exists
