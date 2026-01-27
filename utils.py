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
    """Check if file exists in either local filesystem or Google Drive"""
    if "database_type" in st.session_state and st.session_state["database_type"] == "Local Files":
        import os
        # For local files, check actual filesystem
        if not os.path.isabs(path):
            # Relative path - need to check if it exists relative to workspace or data root
            # First try relative to current directory
            if os.path.exists(path):
                return True
            # Then try relative to data root if we can get it
            try:
                root = st.session_state.get("local_data_path", "").rstrip('/')
                if root:
                    full_path = os.path.join(root, path)
                    return os.path.exists(full_path)
            except:
                pass
            return False
        else:
            # Absolute path
            return os.path.exists(path)
    else:
        # Google Drive
        return get_id_from_path(path) is not None

# ==========================================
# Local File System Operations
# ==========================================

def list_dirs_local(path):
    """Lists directories in local filesystem"""
    import os
    full_path = path if os.path.isabs(path) else os.path.join(get_data_root(), path)
    if not os.path.exists(full_path):
        return []
    try:
        return [d for d in os.listdir(full_path) 
                if os.path.isdir(os.path.join(full_path, d))]
    except:
        return []

def list_files_local(path):
    """Lists files in local filesystem"""
    import os
    full_path = path if os.path.isabs(path) else os.path.join(get_data_root(), path)
    if not os.path.exists(full_path):
        return []
    try:
        return [f for f in os.listdir(full_path) 
                if os.path.isfile(os.path.join(full_path, f))]
    except:
        return []

def read_file_local(path):
    """Reads a file from local filesystem"""
    import os
    full_path = path if os.path.isabs(path) else os.path.join(get_data_root(), path)
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading local file {full_path}: {e}")
        return None

def read_file_binary_local(path):
    """Reads a binary file from local filesystem"""
    import os
    full_path = path if os.path.isabs(path) else os.path.join(get_data_root(), path)
    try:
        with open(full_path, 'rb') as f:
            return f.read()
    except:
        return None

# ==========================================
# Unified File Operations (Auto-detect source)
# ==========================================

def list_dirs_unified(path=""):
    """Lists directories from either local or Google Drive"""
    if "database_type" in st.session_state and st.session_state["database_type"] == "Local Files":
        return list_dirs_local(path)
    else:
        return list_dirs(path)

def list_files_unified(path=""):
    """Lists files from either local or Google Drive"""
    if "database_type" in st.session_state and st.session_state["database_type"] == "Local Files":
        return list_files_local(path)
    else:
        return list_files(path)

def read_file_unified(path):
    """Reads a file from either local or Google Drive"""
    if "database_type" in st.session_state and st.session_state["database_type"] == "Local Files":
        return read_file_local(path)
    else:
        return read_file(path)

def read_file_binary_unified(path):
    """Reads a binary file from either local or Google Drive"""
    if "database_type" in st.session_state and st.session_state["database_type"] == "Local Files":
        return read_file_binary_local(path)
    else:
        return read_file_binary(path)

# ==========================================
# Domain Specific Helpers
# ==========================================

def get_data_root():
    """
    Determines the data root path based on database configuration.
    Checks session state for database type and local path settings.
    Note: This function is NOT cached to allow dynamic switching between databases.
    """
    try:
        # Check if using local files from session state
        if "database_type" in st.session_state and st.session_state["database_type"] == "Local Files":
            if "local_data_path" in st.session_state:
                local_path = st.session_state["local_data_path"]
                # Return the path directly (already absolute)
                return local_path.rstrip('/')
        
        # Default behavior for Google Drive or when not configured
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

def get_shot_list():
    """Returns a sorted list of available shot numbers."""
    root_loc = get_data_root()
    dirs = list_dirs_unified(root_loc if root_loc else "")
    shots = [d for d in dirs if d.isdigit()]
    if not shots:
        return ["1001"] # Placeholder
    return sorted(shots, key=lambda x: int(x), reverse=True)

@st.cache_data
def load_timeseries(shot_id):
    """Loads standard time-series data"""
    root = get_data_root()
    path = f"{root}/shot_{shot_id}.txt" if root else f"shot_{shot_id}.txt"
    
    content = read_file_unified(path)
    if content:
         return pd.read_csv(io.StringIO(content), sep=r'\s+', comment='#')
    
    # Fallback: maybe inside the shot folder?
    path_inner = f"{root}/{shot_id}/shot_{shot_id}.txt" if root else f"{shot_id}/shot_{shot_id}.txt"
    content_inner = read_file_unified(path_inner)
    if content_inner:
        return pd.read_csv(io.StringIO(content_inner), sep=r'\s+', comment='#')
        
    return None

@st.cache_data
def load_hxr_data(shot_id):
    root = get_data_root()
    path = f"{root}/shot_{shot_id}_hxr.csv"
    if not root: path = f"shot_{shot_id}_hxr.csv"
    
    content = read_file_unified(path)
    if content:
        return pd.read_csv(io.StringIO(content))
    return None

def get_video_bytes(shot_id):
    root = get_data_root()
    path = f"{root}/shot_{shot_id}_cam.mp4"
    if not root: path = f"shot_{shot_id}_cam.mp4"
    return read_file_binary_unified(path)

# Aliases for compatibility (use unified versions by default)
read_github_file = read_file_unified
read_github_file_binary = read_file_binary_unified
list_github_dirs = list_dirs_unified
list_github_files = list_files_unified
get_github_shot_list = get_shot_list
github_file_exists = file_exists
