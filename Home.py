import streamlit as st
import utils
import datetime

# Page Configuration
st.set_page_config(
    page_title="T-TEAM Home",
    page_icon="🇹🇭",
    layout="wide"
)

# Authentication check
if not utils.check_auth():
    st.stop()

# Banner
st.image("assets/T-TEAM_Banner_03B.png", width="stretch")

# Initialize Session State
if "current_shot" not in st.session_state:
    st.session_state["current_shot"] = "1001"

# --- Sidebar: Database Logic ---
st.sidebar.header("Database Status")

# Check query parameters or secrets for Google Drive Service Account
if "gcp_service_account" in st.secrets:
    st.sidebar.success("Connected to T-TEAM Data Drive")
else:
    st.sidebar.warning("Drive Credentials Not Found")

# Get shot list (using the function added to utils)
available_shots = utils.get_shot_list()

# Determine default index based on current session state
try:
    current_index = available_shots.index(st.session_state["current_shot"])
except ValueError:
    current_index = 0

def update_session_shots():
    # Sync the selection with the global current_shot and Time Trace page
    new_shot = st.session_state["shot_selector_home"]
    st.session_state["current_shot"] = new_shot
    # Force Time Trace selection to match this new choice
    st.session_state["selected_shots_ms"] = [new_shot]

# Selectbox
st.sidebar.selectbox(
    "Select Discharge (Shot #):",
    options=available_shots,
    index=current_index,
    key="shot_selector_home",
    on_change=update_session_shots
)

# Ensure session state is synced on first load or refresh if keys exist
if "shot_selector_home" in st.session_state:
    st.session_state["current_shot"] = st.session_state["shot_selector_home"]

# Initialize 'selected_shots_ms' for Time Trace if it doesn't exist
if "selected_shots_ms" not in st.session_state:
    st.session_state["selected_shots_ms"] = [st.session_state["current_shot"]]

# Update variable for local use
selected_shot = st.session_state["current_shot"]

# --- Main Layout ---

# Custom CSS for Header
st.markdown("""
    <style>
    .header-text {
        font_size: 70px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header-text">🇹🇭 T-TEAM: Thailand Tokamak-1 Experiment Analysis Module</div>', unsafe_allow_html=True)

# Split page layout
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Sawasdee & Warm Welcome! 🙏")
    st.markdown("""
    We are delighted to welcome participants of the **11th ASEAN School on Plasma and Nuclear Fusion (ASPNF2026)** and **SOKENDAI Asian School**.
    
    🌐 **[Visit School Website](https://sites.google.com/site/fusionschoolth/aspnf2026)**
    """)
    
    st.divider()
    
    st.subheader("About the Platform")
    st.markdown("""
    The **TT-1 Analysis Suite** provides a centralized interface for analyzing experimental data from the Thailand Tokamak-1 (TT-1).
    
    ### Available Modules:
    - **Time Traces**: Visualize key plasma parameters such as Plasma Current ($I_p$), Loop Voltage ($V_{loop}$), and Toroidal Magnetic Field ($B_t$).
    - **HXR Analysis**: Analyze Hard X-Ray emissions events and energy histograms.
    - **CCD Analysis**: View high-speed imaging of the plasma discharge.
    """)
    
    st.warning("""
    **Educational Use Only:**
    This platform is currently **under development** and is intended mainly for educational and training purposes during the school. 
    """)

with col_right:
    st.subheader("Shot Info")
    latest_shot = available_shots[0] if available_shots else "N/A"
    current_date = datetime.date.today().strftime("%B %d, %Y")
    
    st.info(f"Current Selected Shot: **{st.session_state['current_shot']}**")
    st.success(f"Latest Available Shot: **{latest_shot}**")
    st.write(f"📅 **Date:** {current_date}")
    
    st.markdown("Use the sidebar **Database Status** section to change the active discharge number for analysis.")
    
    st.divider()
    
    st.subheader("Mantained By")
    st.markdown("""
    **Thailand Tokamak-1 Team**and **Apiwat Wisitsorasak**  
    """)

# Contact & Suggestions
st.divider()
st.subheader("Contact & Suggestions")
st.markdown("""
If you have any feedback or inquiries, please contact: Dr. Apiwat Wisitsorasak 
📧 **apiwat.wis@kmutt.ac.th**
""")

st.caption("Developed by Apiwat Wisitsorasak | Thailand Tokamak-1 Team © 2026")