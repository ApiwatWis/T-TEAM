# utils.py
import pandas as pd
import streamlit as st
import numpy as np
import os

DATA_FOLDER = "data"

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