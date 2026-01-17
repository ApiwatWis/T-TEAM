# Home.py
import streamlit as st
import os

st.set_page_config(
    page_title="T-TEAM Home",
    page_icon="🇹🇭",
    layout="wide"
)

st.title("🇹🇭 Thailand Tokamak-1 Experiment Analysis Module")

st.markdown("""
Welcome to the **TT-1 Experimental Data Platform**.
Select a module from the sidebar to begin analysis.
""")

# --- GLOBAL SESSION STATE ---
# We store the 'Current Shot' in memory so all pages can access it.
if "current_shot" not in st.session_state:
    st.session_state["current_shot"] = "1001"

import utils

# --- SHOT SELECTION ---
st.sidebar.header("Global Settings")
# listing files to find available shots
# Try to finding shot folders in repo (root or data/)
available_shots = []
# 1. Try root
dirs_root = utils.list_github_dirs("")
# Filter for numeric folders (shots)
shots_root = [d for d in dirs_root if d.isdigit()]

if shots_root:
    available_shots = shots_root
else:
    # 2. Try 'data' folder
    dirs_data = utils.list_github_dirs("data")
    available_shots = [d for d in dirs_data if d.isdigit()]

# Sort shots
available_shots = sorted(available_shots, key=lambda x: int(x), reverse=True)

if not available_shots:
    available_shots = ["1001"] # fallback

selected_shot = st.selectbox(
    "Select Discharge (Shot #):", 
    options=available_shots
)

# Update the session state
st.session_state["current_shot"] = selected_shot

st.info(f"Current Active Shot: **{st.session_state['current_shot']}**")
st.write("Navigate to the **Sidebar** 👈 to choose an analysis tool.")