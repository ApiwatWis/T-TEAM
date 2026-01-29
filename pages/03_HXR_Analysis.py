import streamlit as st
import os
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import yaml
import utils
import io

# Set page title
st.title("HXR Analysis")

# Display database source
if "database_type" in st.session_state:
    db_type = st.session_state["database_type"]
    if db_type == "Local Files":
        db_path = st.session_state.get("local_data_path", "Not configured")
        st.info(f"📂 Database: **{db_type}** | Path: `{db_path}`")
    else:
        st.info(f"☁️ Database: **{db_type}**")
else:
    st.info("📂 Database: **Default** | Path: `data/`")

st.sidebar.title("HXR Analysis Setup")

# Main path for data files
BASE_PATH = utils.get_data_root()

# Load signals metadata
@st.cache_data
def load_signals_metadata():
    metadata_path = os.path.join(os.getcwd(), "signals.yaml")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

signals_metadata = load_signals_metadata()

# -----------------------------
# Data Loading Functions
# -----------------------------

def load_signal(path, filename):
    """Loads signal data from a .txt file."""
    signal_name = filename.replace(".txt", "") 
    full_path = os.path.join(path, filename)
    try:
        content = utils.read_github_file(full_path)
        if not content:
            return pd.DataFrame()
            
        data = pd.read_csv(
            io.StringIO(content),
            skiprows=8,
            delimiter=r'\s+',
            header=None,
            names=["Time", signal_name.upper()]
        )
        return data
    except Exception as e:
        return pd.DataFrame()

def load_labr3_signal(base_path, shot_id, detector_name):
    """
    Loads LaBr3 data and returns a time-series of Count Rate.
    detector_name: 'LaBr3_1' or 'LaBr3_2'
    """
    try:
        # Try both .CSV and .csv extensions
        content = None
        for ext in ['.CSV', '.csv']:
            filename = f"{detector_name}{ext}"
            file_path = os.path.join(base_path, shot_id, filename)
            content = utils.read_github_file(file_path)
            if content:
                break
        
        if not content:
            return pd.DataFrame()

        if len(content) < 100: 
             return pd.DataFrame()

        df = pd.read_csv(io.StringIO(content), delimiter=';', header=0)
        if df.empty or 'TIMETAG' not in df.columns:
            return pd.DataFrame()

        timetag_0 = df['TIMETAG'].values
        
        # Time conversion
        time_sec = timetag_0 / 250e6 / 1000.0 / 4.0
        time_ms = time_sec * 1000.0

        if len(time_ms) == 0:
            return pd.DataFrame()
        
        # Binning for Count Rate (1ms bin)
        time_min = np.floor(np.min(time_ms))
        time_max = np.ceil(np.max(time_ms))
        
        # Limit to reasonable time range
        max_duration = 2000.0  # ms
        
        if time_max - time_min > max_duration:
            hist, edges = np.histogram(time_ms, bins=100)
            max_bin_idx = np.argmax(hist)
            center_time = (edges[max_bin_idx] + edges[max_bin_idx + 1]) / 2
            
            start_t = max(0, center_time - max_duration/2)
            end_t = start_t + max_duration
        else:
            start_t = np.floor(time_min)
            end_t = np.ceil(time_max)
        
        if end_t <= start_t:
             start_t = time_min
             end_t = time_max + 1.0

        bin_size = 1.0 # ms
        bins = np.arange(start_t, end_t + bin_size, bin_size)
        
        counts, bin_edges = np.histogram(time_ms, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        rate = counts / (bin_size / 1000.0) # Hz
        
        return pd.DataFrame({
            "Time": bin_centers,
            detector_name.upper(): rate
        })

    except Exception as e:
        return pd.DataFrame()

def get_available_signals(base_path, sample_shot):
    """Retrieves a list of available signal files for a given shot."""
    if not sample_shot:
        return []
    shot_path = os.path.join(base_path, sample_shot)
    
    files = utils.list_github_files(shot_path)
    if not files:
        return []

    signals = set()
    valid_exts = {".txt", ".csv", ".CSV", ".TXT"}
    
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() in {".txt", ".csv"}:
             signals.add(name)
             
    signals = list(signals)
    return sorted(signals)

def get_all_available_shots(base_path):
    """Retrieves a list of all available shot directories."""
    shots = utils.list_github_dirs(base_path)
    
    try:
        return sorted(shots, key=lambda x: int(x) if x.isdigit() else x)
    except:
        return sorted(shots)

def cal_duration(data, time, start_threshold_ratio=0.05, end_threshold_ratio=0.05):
    """
    Calculates the duration based on start and end thresholds relative to max value.
    """
    if len(data) == 0 or np.all(data == 0): 
        return 0, -1, -1, None, None

    max_val = np.max(data)
    if max_val == 0: 
        return 0, -1, -1, None, None
        
    start_threshold = start_threshold_ratio * max_val
    end_threshold = end_threshold_ratio * max_val
    
    peak_idx = np.argmax(data)
    
    # Find start index
    pre_peak_indices = np.where(data[:peak_idx+1] > start_threshold)[0]
    if len(pre_peak_indices) > 0:
        start_idx = pre_peak_indices[0]
    else:
        start_idx = 0

    # Find end index
    post_peak_indices = np.where(data[peak_idx:] < end_threshold)[0]
    if len(post_peak_indices) > 0:
        end_idx = peak_idx + post_peak_indices[0]
    else:
        end_idx = len(data) - 1

    start_time = time[start_idx]
    end_time = time[end_idx]
    duration = end_time - start_time
    
    return duration, start_idx, end_idx, start_time, end_time

# -----------------------------
# Session State Initialization
# -----------------------------

if "hxr_load_and_plot_clicked" not in st.session_state:
    st.session_state["hxr_load_and_plot_clicked"] = False

if "hxr_plot_t0" not in st.session_state:
    st.session_state["hxr_plot_t0"] = 250

if "hxr_plot_t1" not in st.session_state:
    st.session_state["hxr_plot_t1"] = 500

# -----------------------------
# Get Available Shots
# -----------------------------

all_available_shots = get_all_available_shots(BASE_PATH)

if not all_available_shots:
    st.error(f"No discharge folders found in '{BASE_PATH}'. Please ensure there are folders containing data.")
else:
    # --- Discharge Selection ---
    st.sidebar.markdown("---") 
    st.sidebar.header("Discharge Selection") 

    if st.sidebar.button("Clear selections", key="hxr_clear"):
        st.session_state["hxr_selected_shots"] = []
        st.session_state["hxr_selected_ref_signals"] = []
        st.rerun()

    # Default selection
    default_selection = []
    if "current_shot" in st.session_state and st.session_state["current_shot"] in all_available_shots:
        default_selection = [st.session_state["current_shot"]]
    elif all_available_shots:
        default_selection = [all_available_shots[-1]]

    selected_shots = st.sidebar.multiselect(
        "Select Discharges",
        options=all_available_shots,
        default=default_selection,
        key="hxr_selected_shots"
    )

    # --- Reference Signals Selection ---
    ref_signal_list = []
    if selected_shots:
        sample_shot = selected_shots[0]
        signal_choices = get_available_signals(BASE_PATH, sample_shot)
        
        if not signal_choices:
            st.sidebar.warning(f"No signal files found for shot '{sample_shot}'.")
        else:
            # Default reference signals for HXR analysis
            default_ref_signals = ["VP0", "IP2"]
            
            # Filter to only available signals
            if "hxr_selected_ref_signals" not in st.session_state:
                default_refs = [sig for sig in default_ref_signals if sig in signal_choices]
                if not default_refs and len(signal_choices) >= 2:
                    default_refs = signal_choices[:2]
            else:
                default_refs = [sig for sig in st.session_state["hxr_selected_ref_signals"] if sig in signal_choices]
                if not default_refs:
                    default_refs = [sig for sig in default_ref_signals if sig in signal_choices]
                    if not default_refs and len(signal_choices) >= 2:
                        default_refs = signal_choices[:2]
            
            ref_signal_list = st.sidebar.multiselect(
                "Reference Signals (for correlation)", 
                signal_choices, 
                default=default_refs,
                key="hxr_selected_ref_signals",
                help="Select plasma diagnostics to correlate with HXR emissions"
            )
    else:
        st.sidebar.info("Please select at least one discharge.")

    # --- Plotting Options ---
    st.sidebar.markdown("---")
    st.sidebar.header("Plotting Options")
    
    use_long_names = st.sidebar.toggle("Use Descriptive Signal Names", value=False, key="hxr_long_names")
    highlight_interval = st.sidebar.toggle("Highlight Discharge Interval", value=True, key="hxr_highlight")
    
    highlight_ref_signal = "IP2"
    if highlight_interval:
        # Determine available IP signals for highlighting
        available_ip_signals = []
        if selected_shots and signal_choices:
            for ip_sig in ["IP2", "IP1"]:
                if ip_sig in signal_choices:
                    available_ip_signals.append(ip_sig)
        
        if available_ip_signals:
            highlight_ref_signal = st.sidebar.radio(
                "Reference Signal for Interval", 
                available_ip_signals,
                key="hxr_highlight_ref"
            )
    
    # Time Range Inputs
    st.sidebar.markdown("---")
    st.sidebar.header("Time Range")
    
    t0 = st.sidebar.number_input(
        "Plot Start time (ms)", 
        value=st.session_state["hxr_plot_t0"], 
        key="hxr_t0_input"
    )
    t1 = st.sidebar.number_input(
        "Plot End time (ms)", 
        value=st.session_state["hxr_plot_t1"], 
        key="hxr_t1_input"
    )
    
    # Update session state
    st.session_state["hxr_plot_t0"] = t0
    st.session_state["hxr_plot_t1"] = t1

    # --- Y-axis Labels and Subplot Titles ---
    # Combine reference signals with LaBr3 detectors
    all_signals = ref_signal_list + ["LaBr3_1", "LaBr3_2"]
    
    ylabels = {}
    st_titles = {}
    for sig in all_signals:
        meta = signals_metadata.get(sig, {})
        unit = meta.get('unit', '')
        short = meta.get('short', sig)
        long = meta.get('long', sig)
        
        # Determine display unit
        display_unit = unit
        if "I" in sig.upper() and unit == "A":
            display_unit = "kA"
        elif sig.upper().startswith("LABR3"):
            display_unit = "cps"  # counts per second
            
        ylabels[sig] = f"{sig} ({display_unit})" if display_unit else sig
        st_titles[sig] = long if use_long_names else short

    # --- Color Map for Shots ---
    unique_shots = list(set(selected_shots))
    base_colors = px.colors.qualitative.Plotly
    color_map = {shot: base_colors[i % len(base_colors)] for i, shot in enumerate(unique_shots)}

    # --- Load and Plot Button ---
    st.sidebar.markdown("---")
    
    # Custom CSS for orange button
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ff9f1c;
        color: white; 
        border: none;
    }
    div.stButton > button:hover {
        background-color: #e08e1a;
        color: white;
    }
    </style>""", unsafe_allow_html=True)

    if st.sidebar.button("Load and Plot", type="primary", key="hxr_load_plot"):
        st.session_state["hxr_load_and_plot_clicked"] = True

    # -----------------------------
    # Plotting Logic
    # -----------------------------

    if not st.session_state["hxr_load_and_plot_clicked"]:
        st.markdown("## 🔬 Welcome to HXR Analysis")
        st.markdown("""
        This page analyzes **Hard X-Ray (HXR) emissions** from LaBr3 detectors, 
        correlated with plasma diagnostics.
        
        ### Getting Started:
        1. **Select Discharges** in the sidebar
        2. **Select Reference Signals** (default: VP0, IP2) for correlation
        3. **Adjust Time Range** if needed
        4. Click **Load and Plot**
        
        ### Features:
        - 📊 Multi-signal dashboard with synchronized time axes
        - 🔍 **Box selection zoom**: Click and drag on any plot to zoom in
        - 🎯 Discharge interval highlighting (based on Plasma Current > 2kA)
        - 📈 LaBr3 count rates automatically included
        """)
        
        st.divider()
        st.subheader("📋 LaBr3 Detector Information")
        
        labr3_info = []
        for sig in ["LaBr3_1", "LaBr3_2"]:
            meta = signals_metadata.get(sig, {})
            if meta:
                labr3_info.append({
                    "Detector": f"`{sig}`",
                    "Type": "Lanthanum Bromide Scintillation",
                    "Unit": meta.get("unit", ""),
                    "Description": meta.get("long", "")
                })
        
        if labr3_info:
            st.markdown(pd.DataFrame(labr3_info).to_markdown(index=False))

        st.markdown("---")
        st.info("ℹ️ HXR emissions indicate fast electron populations and runaway electron generation during plasma discharges.")

    elif st.session_state["hxr_load_and_plot_clicked"] and selected_shots:
        st.write("### HXR Analysis Dashboard")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Calculate discharge durations if highlighting is enabled
        shot_durations = {}
        if highlight_interval:
            status_text.text("Calculating discharge durations...")
            low_ip_shots = []
            
            for shot in selected_shots:
                ip_cols = [highlight_ref_signal] if highlight_ref_signal in ["IP1", "IP2"] else ["IP2", "IP1", "IP"]
                     
                loaded = False
                for ip_name in ip_cols:
                    filename = f"{ip_name}.txt"
                    file_path = os.path.join(BASE_PATH, shot)
                    
                    try:
                        df_ip = load_signal(file_path, filename)
                        if df_ip.empty:
                            continue
                            
                        df_ip = df_ip[(df_ip["Time"] >= t0) & (df_ip["Time"] <= t1)]
                        if not df_ip.empty:
                            time_ip = df_ip["Time"].to_numpy()
                            data_ip = df_ip[ip_name.upper() if ip_name.upper() in df_ip.columns else df_ip.columns[1]].to_numpy()
                            
                            max_val = np.max(data_ip)
                            threshold_amp = 2000
                            
                            if max_val >= threshold_amp:
                                dur, s_idx, e_idx, s_time, e_time = cal_duration(
                                    data_ip, time_ip, 
                                    start_threshold_ratio=0.05, 
                                    end_threshold_ratio=0.05
                                )
                                shot_durations[shot] = {
                                    "duration": dur,
                                    "start_time": s_time,
                                    "end_time": e_time
                                }
                            else:
                                low_ip_shots.append(shot)
                            loaded = True
                            break
                    except Exception:
                        pass
            
            if low_ip_shots:
                st.warning(f"Plasma current (IP) is below 2 kA for shots: {', '.join(low_ip_shots)}. Discharge interval highlighting skipped for these shots.")
        
        # Calculate subplot layout (vertical stacking)
        num_plots = len(all_signals)
        subplot_rows = num_plots
        subplot_cols = 1
        
        # Dynamic height calculation
        height_per_row = 300
        total_height = height_per_row * subplot_rows
        
        vertical_spacing_val = 0.08
        if subplot_rows > 1:
            vertical_spacing_val = 60.0 / total_height
            max_spacing = 1.0 / (subplot_rows - 1)
            if vertical_spacing_val >= max_spacing:
                vertical_spacing_val = max_spacing * 0.9

        # Create subplots
        fig = make_subplots(
            rows=subplot_rows,
            cols=subplot_cols,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing_val,
            subplot_titles=[st_titles.get(sig, sig) for sig in all_signals]
        )

        signals_not_found = []
        legend_shots_added = set()
        
        total_signals = len(all_signals)

        # Plot each signal
        for idx, sig in enumerate(all_signals):
            progress = (idx) / total_signals
            progress_bar.progress(progress)
            status_text.text(f"Loading signal: {sig} ({idx+1}/{total_signals})...")
            
            row = idx + 1
            col = 1
            
            data_found_for_sig = False
            missing_shots = []

            for shot in selected_shots:
                df = pd.DataFrame()
                
                # Load LaBr3 or standard signal
                if sig.upper().startswith("LABR3"):
                    df = load_labr3_signal(BASE_PATH, shot, sig)
                    if df.empty:
                        missing_shots.append(shot)
                else:
                    filename = f"{sig}.txt"
                    file_path = os.path.join(BASE_PATH, shot)
                    
                    try:
                        df = load_signal(file_path, filename)
                        if df.empty:
                            missing_shots.append(shot)
                    except Exception:
                        df = pd.DataFrame()
                        missing_shots.append(shot)

                try:
                    if df.empty:
                        continue
                        
                    # Filter by time range
                    df = df[(df["Time"] >= t0) & (df["Time"] <= t1)]
                    if df.empty:
                        continue
                    
                    data_found_for_sig = True

                    # Scale current signals to kA
                    y_data = df[sig.upper()]
                    if "I" in sig.upper():
                        y_data = y_data / 1000

                    # Legend optimization
                    show_legend = False
                    if shot not in legend_shots_added:
                        show_legend = True
                        legend_shots_added.add(shot)

                    # Add trace
                    fig.add_trace(go.Scatter(
                        x=df["Time"],
                        y=y_data,
                        name=f"{shot}",
                        legendgroup=f"group_{shot}",
                        showlegend=show_legend,
                        mode="lines",
                        line=dict(color=color_map.get(shot, "gray"))
                    ), row=row, col=col)
                    
                    # Add discharge interval highlighting
                    if highlight_interval and shot in shot_durations:
                        dur_info = shot_durations[shot]
                        s_time = dur_info["start_time"]
                        e_time = dur_info["end_time"]
                        duration_val = dur_info["duration"]
                        
                        if s_time is not None and e_time is not None:
                            fig.add_vrect(
                                x0=s_time, x1=e_time,
                                fillcolor=color_map.get(shot, "pink"),
                                opacity=0.2,
                                line_width=0,
                                row=row, col=col
                            )
                            # Duration annotation
                            fig.add_annotation(
                                x=s_time + (e_time - s_time) / 2,
                                y=np.max(y_data) * 0.9 if len(y_data) > 0 else 0,
                                text=f"{duration_val:.1f} ms",
                                showarrow=False,
                                yshift=10,
                                font=dict(color=color_map.get(shot, "black"), size=10),
                                row=row, col=col
                            )

                except Exception as e:
                    pass

            # Update axes
            fig.update_yaxes(
                title_text=ylabels.get(sig, sig),
                row=row,
                col=col
            )
            
            fig.update_xaxes(
                title_text="Time (ms)",
                showticklabels=True,
                matches='x',
                row=row,
                col=col
            )
            
            if not data_found_for_sig:
                signals_not_found.append(sig)
            elif missing_shots:
                st.warning(f"Data not available for **{sig}** in Discharge(s): {', '.join(missing_shots)}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
            
        if signals_not_found:
            st.warning(f"No data available for the following signals: {', '.join(signals_not_found)}")
        
        # Update layout with relayout event handling for zoom
        fig.update_layout(
            height=total_height,
            margin=dict(t=80, b=100, l=60, r=60),
            showlegend=True,
            title_text="HXR Analysis Dashboard",
            hovermode="x unified",
            dragmode="zoom"  # Enable box zoom mode
        )
        
        # Display plot
        st.plotly_chart(fig, key="hxr_main_plot", use_container_width=True)
        
        # Info about zoom functionality
        st.info("💡 **Tip**: Click and drag on any plot to zoom in. Use the zoom tools in the top-right corner to pan, zoom, or reset the view.")
