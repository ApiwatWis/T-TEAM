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

# --- SHOT SELECTION ---
st.sidebar.header("Global Settings")
# listing files to find available shots
available_files = [f for f in os.listdir("data") if f.endswith('.txt')]
available_shots = [f.split('_')[1].split('.')[0] for f in available_files]

selected_shot = st.selectbox(
    "Select Discharge (Shot #):", 
    options=available_shots if available_shots else ["1001"]
)

# Update the session state
st.session_state["current_shot"] = selected_shot

st.info(f"Current Active Shot: **{st.session_state['current_shot']}**")
st.write("Navigate to the **Sidebar** 👈 to choose an analysis tool.")