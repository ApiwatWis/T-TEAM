import streamlit as st
import utils
import datetime

# Page Configuration
st.set_page_config(
    page_title="T-TEAM Home",
    page_icon="🇹🇭",
    layout="wide"
)

# Banner
st.image("assets/T-TEAM_Banner_03B.png", width="stretch")

# Initialize Session State
if "current_shot" not in st.session_state:
    st.session_state["current_shot"] = "1001"

# Initialize database settings in session state
if "database_type" not in st.session_state:
    st.session_state["database_type"] = "Local Files"

if "local_data_path" not in st.session_state:
    st.session_state["local_data_path"] = "/Users/apiwat/git/T-TEAM-Data/data/"

# --- Sidebar: Database Configuration ---
st.sidebar.header("Database Configuration")

# Database type selection
database_type = st.sidebar.radio(
    "Select Database:",
    options=["Local Files", "Google Drive"],
    index=0 if st.session_state["database_type"] == "Local Files" else 1,
    key="db_type_selector"
)

# Update session state
st.session_state["database_type"] = database_type

# Local files configuration
if database_type == "Local Files":
    st.sidebar.subheader("Local Directory")
    
    local_path = st.sidebar.text_input(
        "Data Directory Path:",
        value=st.session_state["local_data_path"],
        key="local_path_input"
    )
    
    if local_path != st.session_state["local_data_path"]:
        st.session_state["local_data_path"] = local_path
        st.rerun()
    
    # Check if path exists
    import os
    if os.path.exists(local_path):
        st.sidebar.success("✓ Directory found")
        # Count available shots
        try:
            shot_folders = [d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d)) and d.isdigit()]
            num_shots = len(shot_folders)
            st.sidebar.info(f"📊 {num_shots} discharge(s) available")
        except Exception as e:
            st.sidebar.warning(f"Could not read directory: {e}")
    else:
        st.sidebar.error("✗ Directory not found")

# Google Drive configuration
elif database_type == "Google Drive":
    st.sidebar.subheader("Google Drive Status")
    
    if "gcp_service_account" in st.secrets:
        st.sidebar.success("✓ Connected to T-TEAM Data Drive")
        # Get shot list from Google Drive
        try:
            available_shots = utils.get_shot_list()
            num_shots = len([s for s in available_shots if s != "1001"])  # Exclude placeholder
            st.sidebar.info(f"📊 {num_shots} discharge(s) available")
        except Exception as e:
            st.sidebar.warning(f"Could not fetch data: {e}")
    else:
        st.sidebar.error("✗ Drive credentials not configured")
        st.sidebar.caption("Configure credentials in Streamlit secrets")

st.sidebar.divider()

# Database Status Summary
st.sidebar.header("Database Status")
if database_type == "Local Files":
    import os
    if os.path.exists(st.session_state["local_data_path"]):
        st.sidebar.success(f"🗂️ Using: Local Files")
        st.sidebar.caption(f"Path: {st.session_state['local_data_path']}")
    else:
        st.sidebar.error("❌ Local path not accessible")
else:
    if "gcp_service_account" in st.secrets:
        st.sidebar.success("☁️ Using: Google Drive")
    else:
        st.sidebar.error("❌ Google Drive not configured")

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
    st.subheader("Database Overview")
    
    # Get shot list based on database type
    try:
        if st.session_state["database_type"] == "Local Files":
            import os
            local_path = st.session_state["local_data_path"]
            if os.path.exists(local_path):
                available_shots = [d for d in sorted(os.listdir(local_path), reverse=True) 
                                 if os.path.isdir(os.path.join(local_path, d)) and d.isdigit()]
            else:
                available_shots = []
        else:
            available_shots = utils.get_shot_list()
            available_shots = [s for s in available_shots if s != "1001"]  # Remove placeholder
    except:
        available_shots = []
    
    latest_shot = available_shots[0] if available_shots else "N/A"
    current_date = datetime.date.today().strftime("%B %d, %Y")
    
    st.info(f"Database: **{st.session_state['database_type']}**")
    st.success(f"Latest Available Shot: **{latest_shot}**")
    st.metric("Total Discharges", len(available_shots))
    st.write(f"📅 **Date:** {current_date}")
    
    st.markdown("Use the sidebar **Database Configuration** to select your data source.")
    
    st.divider()
    
    st.subheader("Maintained By")
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