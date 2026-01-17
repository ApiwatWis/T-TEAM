import streamlit as st

st.set_page_config(page_title="Developer Notes", layout="wide")

st.title("📓 Developer Notes")
st.write("This page documents implementation details, environmental setup, and the current project roadmap.")

st.header("Project Architecture")
st.markdown("""
- **Backend**: Python 3.12 (venv)
- **Frontend**: Streamlit
- **Visualization**: Plotly (Dynamic Subplots) & Matplotlib
- **Configuration**: Metadata-driven using `signals.yaml`
""")

st.header("Key Features Implemented")
with st.expander("Recent Updates (Jan 2026)", expanded=True):
    st.markdown("""
    - **Metadata Integration**: Plots now use `signals.yaml` for units, short names (titles), and long names (descriptions).
    - **Physical Scaling**: Current signals (IP, IV, etc.) automatically scale from Amps to kA.
    - **HXR Spectrum**: Added whole-discharge energy histogram with dot-plot markers and Log-Log scaling.
    - **Digitizer Monitoring**: Added a 300 kHz threshold line to LaBr3 time traces to monitor signal saturation.
    - **Interactive Dashboard**: Linked X-axes for synchronized zooming/panning across all subplots.
    """)

st.header("Environmental Requirements")
st.code("""
# Install via terminal:
pip install streamlit pandas plotly numpy PyYAML opencv-python
""", language="bash")

st.header("Roadmap & To-Do")
st.checkbox("Implement cross-correlation analysis between magnetic probes.", value=False)
st.checkbox("Add equilibrium reconstruction (EFIT) visualization module.", value=False)
st.checkbox("Optimize data loading for high-rate HXR files (>1M events).", value=True)
st.checkbox("Integrate CCD video overlay with Time Trace synchronization.", value=False)
st.checkbox("Automated shot summary report generation (PDF).", value=False)

st.markdown("---")
st.caption("THAI TOKAMAK-1 EXPERIMENT ANALYSIS MODULE (T-TEAM) v1.2")
