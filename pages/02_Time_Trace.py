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
import zipfile

# Set page config
# st.set_page_config(layout="wide") # Commented out as likely handled by Home.py handling

st.title("Time Trace Analysis")

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
st.sidebar.title("Signal Dashboard")

# Main path for data files - relative to the workspace
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
#  load form .txt files
def load_signal(path, filename):
    """Loads signal data from a .txt file."""
    # Ensure filename is just the signal name (e.g., "IP1") and not "IP1.txt"
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
            names=["Time", signal_name.upper()] # Use signal_name.upper() as column name
        )
        return data
    except Exception as e:
        # st.error(f"Error loading {filename}: {e}")
        return pd.DataFrame()

def load_labr3_signal(base_path, shot_id, detector_name):
    """
    Loads LaBr3 data and returns a time-series of Count Rate.
    detector_name: 'LaBr3_1' or 'LaBr3_2'
    """
    try:
        # Try both .CSV and .csv extensions (case-insensitive)
        content = None
        for ext in ['.CSV', '.csv']:
            filename = f"{detector_name}{ext}"
            file_path = os.path.join(base_path, shot_id, filename)
            content = utils.read_github_file(file_path)
            if content:
                break
        
        if not content:
            return pd.DataFrame()

        # Check if file is small (heuristic)
        if len(content) < 100: 
             return pd.DataFrame()

        df = pd.read_csv(io.StringIO(content), delimiter=';', header=0)
        if df.empty or 'TIMETAG' not in df.columns:
            return pd.DataFrame()

        timetag_0 = df['TIMETAG'].values
        
        # User Constants
        bit = 2**12
        ts = 4e-9
        
        # time_sec = timetag_0 / bit * ts 
        time_sec = timetag_0 / 250e6 / 1000.0 / 4.0    # second
        time_ms = time_sec * 1000.0

        if len(time_ms) == 0:
            return pd.DataFrame()
        
        # Binning for Count Rate (1ms bin)
        # Limit to a reasonable time range to avoid creating huge DataFrames
        # Typical discharge is 0-1000ms, but we'll be generous with 0-2000ms
        # Only include data points within a reasonable window
        time_min = np.floor(np.min(time_ms))
        time_max = np.ceil(np.max(time_ms))
        
        # If data spans more than 2000ms, limit to reasonable discharge window
        # Most plasma discharges are < 1000ms
        max_duration = 2000.0  # ms
        
        if time_max - time_min > max_duration:
            # Find the region with most data (likely the actual discharge)
            # Use a sliding window to find the densest region
            # For simplicity, use the first significant cluster of data
            hist, edges = np.histogram(time_ms, bins=100)
            max_bin_idx = np.argmax(hist)
            center_time = (edges[max_bin_idx] + edges[max_bin_idx + 1]) / 2
            
            # Create window around the densest region
            start_t = max(0, center_time - max_duration/2)
            end_t = start_t + max_duration
        else:
            start_t = np.floor(time_min)
            end_t = np.ceil(time_max)
        
        if end_t <= start_t:
             # Handle single point or weird range
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
        st.error(f"Error loading LaBr3 {detector_name}: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()

def load_labr3_energy(base_path, shot_id, detector_name):
    """
    Loads LaBr3 energy data (keV) for the whole discharge.
    """
    try:
        # Try both .CSV and .csv extensions (case-insensitive)
        content = None
        for ext in ['.CSV', '.csv']:
            filename = f"{detector_name}{ext}"
            file_path = os.path.join(base_path, shot_id, filename)
            content = utils.read_github_file(file_path)
            if content:
                break
        

        df = pd.read_csv(io.StringIO(content), delimiter=';', header=0)
        if df.empty or 'ENERGY' not in df.columns:
            return np.array([])

        q = df['ENERGY'].values
        
        if len(q) == 0:
            return np.array([])
            
        if detector_name == "LaBr3_2":
            # Formula: E = 1.85*q - 57.44
            E = 1.85 * q - 57.44
        else:
            # Default / LaBr3_1 Formula: E = 2.8775*q - 66.0934
            E = 2.8775 * q - 66.0934
            
        return E
    except Exception as e:
        st.error(f"Error loading LaBr3 energy {detector_name}: {e}")
        import traceback
        st.code(traceback.format_exc())
        return np.array([])

def get_available_signals(base_path, sample_shot):
    """Retrieves a list of available signal files (TXT or CSV) for a given shot."""
    if not sample_shot:
        return []
    shot_path = os.path.join(base_path, sample_shot)
    
    files = utils.list_github_files(shot_path)
    if not files:
        return []

    signals = set()
    valid_exts = {".txt", ".csv", ".CSV", ".TXT"}  # Support both cases
    # Explicitly ensure no video files are picked up, though they shouldn't match .txt/.csv
    
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() in {".txt", ".csv"}:  # Normalize to lowercase for comparison
             signals.add(name)
             
    signals = list(signals)
 
    if "IP1" in signals and "Duration" not in signals: # Duration is based on IP1, so only add if IP1 exists
        signals.append("Duration")
    
    # Ensure LaBr3 are definitely included if present (redundant if they are .CSV, but keeps logic safe)
    # The loop above adds 'LaBr3_1' if 'LaBr3_1.CSV' exists. 
    # Just ensure we aren't doubling up or missing anything.

    return sorted(signals)

# Function to get all available shots (for search)
def get_all_available_shots(base_path):
    """Retrieves a list of all available shot directories."""
    shots = utils.list_github_dirs(base_path)
    
    # Try to sort numerically if possible, otherwise alphabetically
    try:
        return sorted(shots, key=lambda x: int(x) if x.isdigit() else x)
    except:
        return sorted(shots)



# --- Duration Calculation Function 
def cal_duration(data, time, start_threshold_ratio=0.05, end_threshold_ratio=0.05):
    """
    Calculates the duration based on start and end thresholds relative to max value.
    Start: First point > start_threshold_ratio * max
    End: First point < end_threshold_ratio * max (AFTER max)
    """
    if len(data) == 0 or np.all(data == 0): 
        return 0, -1, -1, None, None

    max_val = np.max(data)
    if max_val == 0: 
        return 0, -1, -1, None, None
        
    start_threshold = start_threshold_ratio * max_val
    end_threshold = end_threshold_ratio * max_val
    
    peak_idx = np.argmax(data)
    
    # Find start index (search from beginning to peak)
    # Using the first point that exceeds the start threshold
    pre_peak_indices = np.where(data[:peak_idx+1] > start_threshold)[0]
    if len(pre_peak_indices) > 0:
        start_idx = pre_peak_indices[0]
    else:
        start_idx = 0 # Fallback

    # Find end index (search from peak to end)
    # Using the first point that falls below the end threshold
    post_peak_indices = np.where(data[peak_idx:] < end_threshold)[0]
    if len(post_peak_indices) > 0:
        end_idx = peak_idx + post_peak_indices[0]
    else:
        end_idx = len(data) - 1 # Fallback to end of data

    start_time = time[start_idx]
    end_time = time[end_idx]
    duration = end_time - start_time
    
    return duration, start_idx, end_idx, start_time, end_time

# --- Session State ---
if "load_and_plot_clicked" not in st.session_state:
    st.session_state["load_and_plot_clicked"] = False


# --- GET AVAILABLE SHOTS ---
all_available_shots = get_all_available_shots(BASE_PATH)

if not all_available_shots:
    st.error(f"No discharge folders found in '{BASE_PATH}'. Please ensure there are folders containing data.")
else:
    # --- Step 1: Pick shots (Multiple Selection) ---
    st.sidebar.markdown("---") 
    st.sidebar.header("Plotting & Analysis Setup") 

    if st.sidebar.button("Clear selections"):
        st.session_state["selected_shots_ms"] = []
        st.session_state["selected_signals_ms"] = []
        st.rerun()

    # Determine default selection from session state
    default_selection = []
    if "current_shot" in st.session_state and st.session_state["current_shot"] in all_available_shots:
        default_selection = [st.session_state["current_shot"]]
    elif all_available_shots:
        default_selection = [all_available_shots[-1]]

    selected_shots = st.sidebar.multiselect(
        "Select Discharges",
        options=all_available_shots,
        default=default_selection,
        key="selected_shots_ms"
    )

    # --- Step 2: Pick signals (for plotting) ---
    signal_list = []
    if selected_shots:
        sample_shot_for_plotting_signals = selected_shots[0]
        signal_choices = get_available_signals(BASE_PATH, sample_shot_for_plotting_signals)
        
        if not signal_choices:
            st.sidebar.warning(f"No signal files found for shot '{sample_shot_for_plotting_signals}'.")
        else:
            # Define signal groups
            signal_groups = {
                "Default plot": ["IOH1", "IT1", "IV2", "IF2", "VP0", "IP2", "LaBr3_1", "LaBr3_2"],
                "Loop voltage": ["VP0", "VP1", "VP2", "VP3"],
                "Poloidal flux loops": ["F0", "F1", "F2", "F3", "F4"],
                "H-alpha": ["HA"],
                "HCN interferometer": ["HCN1", "HCN2", "HCN3"],
                "Tangential Magnetic fields": ["GBP1T", "GBP2T", "GBP3T", "GBP4T", "GBP5T", "GBP6T", 
                                                "GBP7T", "GBP8T", "GBP9T", "GBP10T", "GBP11T", "GBP12T"],
                "Normal Magnetic fields": ["GBP1N", "GBP2N", "GBP3N", "GBP4N", "GBP5N", "GBP6N",
                                           "GBP7N", "GBP8N", "GBP9N", "GBP10N", "GBP11N", "GBP12N"]
            }
            
            # Quick selection checkboxes (before multiselect)
            st.sidebar.markdown("**Quick Selection:**")
            
            # Track which checkboxes are selected and detect changes
            selected_groups = []
            checkbox_changed = False
            
            for group_name, group_signals in signal_groups.items():
                checkbox_key = f"checkbox_{group_name}"
                # Store previous state
                prev_state_key = f"prev_{checkbox_key}"
                
                checkbox_value = st.sidebar.checkbox(group_name, key=checkbox_key)
                
                # Check if checkbox state changed
                if prev_state_key in st.session_state:
                    if st.session_state[prev_state_key] != checkbox_value:
                        checkbox_changed = True
                
                # Update previous state
                st.session_state[prev_state_key] = checkbox_value
                
                if checkbox_value:
                    selected_groups.append(group_signals)
            
            # If checkboxes changed, update the selected signals
            if checkbox_changed and selected_groups:
                # Combine all selected groups and filter by available signals
                combined_signals = []
                for group in selected_groups:
                    combined_signals.extend(group)
                # Remove duplicates and filter by available signals
                combined_signals = [sig for sig in combined_signals if sig in signal_choices]
                # Remove duplicates while preserving order
                combined_signals = list(dict.fromkeys(combined_signals))
                st.session_state["selected_signals_ms"] = combined_signals
                st.rerun()
            
            # Determine default signals for initial load
            if "selected_signals_ms" not in st.session_state:
                defaults_req = signal_groups["Default plot"]
                default_signals = [sig for sig in defaults_req if sig in signal_choices]
                if not default_signals and len(signal_choices) >= 2:
                    default_signals = signal_choices[:2]
            else:
                # Filter previous session state to only include signals available in current discharge
                default_signals = [sig for sig in st.session_state["selected_signals_ms"] if sig in signal_choices]
                # If filtering removed all signals, use defaults
                if not default_signals:
                    defaults_req = signal_groups["Default plot"]
                    default_signals = [sig for sig in defaults_req if sig in signal_choices]
                    if not default_signals and len(signal_choices) >= 2:
                        default_signals = signal_choices[:2]
            
            signal_list = st.sidebar.multiselect(
                "Select signal (for plotting)", 
                signal_choices, 
                default=default_signals,
                key="selected_signals_ms"
            )
    else:
        st.sidebar.info("Please select at least one discharge.")


    # --- Time Range Inputs (for plotting) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Plotting Options")
    
    use_long_names = st.sidebar.toggle("Use Descriptive Signal Names", value=False)
    highlight_interval = st.sidebar.toggle("Highlight Discharge Interval", value=False)
    
    highlight_ref_signal = "IP2"
    if highlight_interval:
        highlight_ref_signal = st.sidebar.radio("Reference Signal for Interval", ["IP2", "IP1"])
    
    # Check if LaBr3 signals are selected to show the spectrum toggle
    has_labr3_selected = any(s.startswith("LaBr3") for s in signal_list)
    show_hxr_spectrum = False
    spectrum_scale = "Linear-Linear"
    
    if has_labr3_selected:
        show_hxr_spectrum = st.sidebar.toggle("Show HXR Spectrum (Whole discharge)", value=False)
        if show_hxr_spectrum:
            spectrum_scale = st.sidebar.radio("Spectrum Scale", ["Linear-Linear", "Log-Linear", "Log-Log"])
    
    t0 = st.sidebar.number_input("Plot Start time (ms)", value=250, key="plot_t0") # Changed default to match typical search
    t1 = st.sidebar.number_input("Plot End time (ms)", value=500, key="plot_t1")


    # --- Automatic Y-axis Labels and Subplot Titles from signals.yaml ---
    ylabels = {}
    st_titles = {}
    for sig in signal_list:
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


    # --- Subplot Layout Option  ---
    layout_option = st.sidebar.selectbox(
        "Subplot layout",
        options=["Grid (Auto)", "Vertical"]
    )

    # --- Consolidated and Corrected Layout Calculation ---
    num_plots = len(signal_list) 
    subplot_rows, subplot_cols = 1, 1 

    if num_plots > 0: 
        if layout_option == "Vertical":
            subplot_rows = num_plots
            subplot_cols = 1
        else:  # Grid (Auto)
            subplot_rows = (num_plots + 1) // 2 
            subplot_cols = 2 if num_plots > 1 else 1 
    
    # --- Buttons and Interaction ---
    st.sidebar.markdown("---")
    
    # Custom CSS to make the primary button orange
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ff9f1c; /* Orange */
        color: white; 
        border: none;
    }
    div.stButton > button:hover {
        background-color: #e08e1a;
        color: white;
    }
    </style>""", unsafe_allow_html=True)

    if st.sidebar.button("Load and Plot", type="primary"):
        st.session_state["load_and_plot_clicked"] = True

    # --- Plotting Logic ---
    if not st.session_state["load_and_plot_clicked"]:
        st.markdown("## 👋 Welcome to Time Trace Analysis")
        st.markdown("To start analyzing, please:")
        st.markdown("1. **Select Discharges** in the sidebar.")
        st.markdown("2. **Select Signals** you wish to plot.")
        st.markdown("3. Click **Load and Plot**.")
        
        st.divider()
        st.subheader("📋 Available Diagnostics Signals")
        
        # Table of signals
        table_data = []
        for sig, meta in signals_metadata.items():
            if isinstance(meta, dict):
                 table_data.append({
                     "Signal Name": f"`{sig}`",
                     "Unit": meta.get("unit", ""),
                     "Description": meta.get("long", meta.get("short", ""))
                 })
        
        if table_data:
            st.markdown(pd.DataFrame(table_data).to_markdown(index=False))

        st.markdown("---")
        st.info("ℹ️ For detailed technical specifications, please refer to the **Diagnostics Information** page in the sidebar.")

    elif st.session_state["load_and_plot_clicked"] and selected_shots and signal_list:
        st.write("### Plotting Signals")
        
        # Progress Bar Initialization
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Pre-calculate durations if highlight is ON
        shot_durations = {}
        if highlight_interval:
            status_text.text("Calculating discharge durations...")
            low_ip_shots = []
            for shot in selected_shots:
                # Load IP for duration calculation based on selection
                if highlight_ref_signal == "IP2":
                     ip_cols = ["IP2", "IP1", "IP"]
                else:
                     ip_cols = ["IP1", "IP2", "IP"]
                     
                loaded = False
                for ip_name in ip_cols:
                    filename = f"{ip_name}.txt"
                    file_path = os.path.join(BASE_PATH, shot)
                    # if os.path.exists(os.path.join(file_path, filename)): 
                    try:
                        df_ip = load_signal(file_path, filename)
                        if df_ip.empty:
                            continue
                            
                        df_ip = df_ip[(df_ip["Time"] >= t0) & (df_ip["Time"] <= t1)]
                        if not df_ip.empty:
                            time_ip = df_ip["Time"].to_numpy()
                            data_ip = df_ip[ip_name.upper() if ip_name.upper() in df_ip.columns else df_ip.columns[1]].to_numpy() # Handle col names safely
                            
                            # Check if max current >= 2 kA for highlighting
                            max_val = np.max(data_ip)
                            
                            threshold_amp = 2000
                            if max_val >= threshold_amp:
                                # Use default 0.05 (5%) and 0.05 (5%) thresholds
                                dur, s_idx, e_idx, s_time, e_time = cal_duration(data_ip, time_ip, start_threshold_ratio=0.05, end_threshold_ratio=0.05)
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
        
        # Calculate dynamic vertical spacing to prevent squeezing
        # Target gap ~80px between subplots
        height_per_row = 350
        total_height = height_per_row * subplot_rows
        
        # Plotly vertical_spacing is the fraction of total height distributed BETWEEN subplots.
        # However, for shared_xaxes=True it sometimes behaves differently. 
        # Let's use a more direct calculation.
        vertical_spacing_val = 0.1 # Default
        if subplot_rows > 1:
            # We want a gap of roughly 80 pixels
            # vertical_spacing is the normalized height of ONE gap
            vertical_spacing_val = 80.0 / total_height
            
            # Ensure it doesn't violate Plotly's constraint: spacing < 1/(rows-1)
            max_spacing = 1.0 / (subplot_rows - 1)
            if vertical_spacing_val >= max_spacing:
                vertical_spacing_val = max_spacing * 0.9

        fig = make_subplots(
            rows=subplot_rows,
            cols=subplot_cols,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing_val,
            horizontal_spacing=0.1, 
            subplot_titles=[st_titles.get(sig, sig) for sig in signal_list]
        )

        signals_not_found = []
        legend_shots_added = set()
        
        total_signals = len(signal_list)

        for idx, sig in enumerate(signal_list):
            
            # Update Status
            progress = (idx) / total_signals
            progress_bar.progress(progress)
            status_text.text(f"Loading signal: {sig} ({idx+1}/{total_signals})...")
            
            row = (idx // subplot_cols) + 1
            col = (idx % subplot_cols) + 1
            
            data_found_for_sig = False
            missing_shots = []

            for shot in selected_shots: 
                df = pd.DataFrame()
                
                if sig.upper().startswith("LABR3"):
                    df = load_labr3_signal(BASE_PATH, shot, sig)
                    if df.empty:
                        missing_shots.append(shot)
                        pass
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
                        
                    df = df[(df["Time"] >= t0) & (df["Time"] <= t1)] 
                    if df.empty:
                        continue
                    
                    data_found_for_sig = True

                    y_data = df[sig.upper()]
                    if "I" in sig.upper():
                        y_data = y_data / 1000

                    # Optimize Legend: Group by Shot
                    show_legend = False
                    if shot not in legend_shots_added:
                        show_legend = True
                        legend_shots_added.add(shot)

                    fig.add_trace(go.Scatter(
                        x=df["Time"],
                        y=y_data,
                        name=f"{shot}", # Consolidated name
                        legendgroup=f"group_{shot}", # Group by shot
                        showlegend=show_legend,
                        mode="lines",
                        line=dict(color=color_map.get(shot, "gray"))
                    ), row=row, col=col)
                    
                    # Apply Highlight if ON
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
                             # Centered annotation
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
                    # st.error(f"Error loading/processing {filename} for Shot {shot}: {e}")
                    pass

            # # Add limit line for LaBr3 ; Removed as per request
            # if sig.upper().startswith("LABR3"):
            #     fig.add_hline(
            #         y=300000,
            #         line_dash="dash",
            #         line_color="orange",
            #         annotation_text="threshold (300 kHz)",
            #         annotation_position="top right",
            #         row=row, col=col
            #     )

            fig.update_yaxes(
                title_text=ylabels.get(sig, " "),
                row=row,
                col=col
            )
            
            show_x_label = True # Label time on all graphs as requested
            x_title = "Time (ms)"
            
            fig.update_xaxes(
                title_text=x_title,
                showticklabels=show_x_label,
                matches='x',
                row=row,
                col=col
            )
            
            if not data_found_for_sig:
                signals_not_found.append(sig)
            elif missing_shots:
                 # Data found for some shots, but missing for others
                 st.warning(f"Data not available for **{sig}** in Discharge(s): {', '.join(missing_shots)}")
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
            
        if signals_not_found:
            st.warning(f"No data available for the following signals across selected shots: {', '.join(signals_not_found)}")
            
        fig.update_layout(
            height=height_per_row * subplot_rows,
            margin=dict(t=80, b=100, l=60, r=60),
            showlegend=True,
            title_text="Signal Dashboard",
            hovermode="x unified"
        )
        st.plotly_chart(fig)

        # --- HXR Spectrum Plotting ---
        if has_labr3_selected and show_hxr_spectrum:
            st.markdown("---")
            st.write("### HXR Spectrum (Whole Discharge)")
            
            labr3_signals = [s for s in signal_list if s.startswith("LaBr3")]
            overall_hxr_found = False
            
            # First pass: Collect all spectrum data to find maximum count across all detectors
            all_counts_by_sig = {}
            for sig in sorted(labr3_signals):
                all_counts_by_sig[sig] = []
                
                for shot in selected_shots:
                    energies = load_labr3_energy(BASE_PATH, shot, sig)
                    if len(energies) > 0:
                        min_e = 0.0
                        max_e = 2000.0
                        
                        if max_e >= min_e:
                            bins = np.arange(min_e - 0.5, max_e + 1.5, 1.0)
                            counts, bin_edges = np.histogram(energies, bins=bins)
                            all_counts_by_sig[sig].extend(counts)
            
            # Calculate global maximum count and round to next power of 10
            global_max_count = 0
            for counts_list in all_counts_by_sig.values():
                if len(counts_list) > 0:
                    global_max_count = max(global_max_count, np.max(counts_list))
            
            # Round to next power of 10
            if global_max_count > 0:
                y_axis_max = 10 ** np.ceil(np.log10(global_max_count))
            else:
                y_axis_max = 1
            
            # Separate plots for each LaBr3 detector
            # Sort to keep order consistent if typical names used
            for sig in sorted(labr3_signals): 
                 spec_fig = go.Figure()
                 has_data_for_sig = False
                 export_data = [] # For data download
                 
                 for shot in selected_shots:
                    energies = load_labr3_energy(BASE_PATH, shot, sig)
                    if len(energies) > 0:
                        has_data_for_sig = True
                        overall_hxr_found = True
                        
                        # Binning: 1 keV bins, centered at integers
                        # Edges should be at k-0.5 and k+0.5 for Integer center k
                        # min_e = np.floor(np.min(energies))
                        min_e = 0.0 # Start from 0 keV
                        # Optional: limit to physical non-negative energy if desired, 
                        # but formula creates negatives for low channels. 
                        # Previous code clamped min to 0. Let's keep consistent if intended.
                        # However, let's just bin what we have, but start binning from floor-0.5
                        if min_e < 0: min_e = 0 
                        # max_e = np.ceil(np.max(energies))
                        max_e = 2000.0 # Limit to 2000 keV (2 MeV) for practical purposes
                        
                        if max_e >= min_e:
                            # Bins edges at 0.5, 1.5, 2.5 etc.
                            # stop at max_e + 1.5 to ensuring covering max_e with center max_e
                            bins = np.arange(min_e - 0.5, max_e + 1.5, 1.0)
                            
                            counts, bin_edges = np.histogram(energies, bins=bins)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                            
                            # Show only points larger than zero (counts > 0)
                            mask = counts > 0
                            x_plot = bin_centers[mask]
                            y_plot = counts[mask]

                            # Store for export
                            for ev, cv in zip(x_plot, y_plot):
                                export_data.append({"Shot": shot, "Energy_keV": ev, "Count": cv})
                            
                            spec_fig.add_trace(go.Scatter(
                                x=x_plot,
                                y=y_plot,
                                mode='lines+markers',
                                name=f"{shot}",
                                line=dict(color=color_map.get(shot, "gray"))
                            ))

                 if has_data_for_sig:
                     spec_fig.update_layout(
                         title=f"HXR Energy Spectrum - {sig}",
                         xaxis_title="Energy (keV)",
                         yaxis_title="Counts",
                         yaxis_type="log" if spectrum_scale in ["Log-Log", "Log-Linear"] else "linear",
                         xaxis_type="log" if spectrum_scale == "Log-Log" else "linear",
                         height=400
                     )
                     
                     # Set range 1-2000 keV (0-2 MeV)
                     if spectrum_scale == "Log-Log":
                         # Log scale requires log10 of limits (1 to 2000)
                         spec_fig.update_xaxes(range=[np.log10(1), np.log10(2000)])
                     else:
                         # Linear x-axis for both Linear-Linear and Log-Linear
                         spec_fig.update_xaxes(range=[1, 2000])
                     
                     # Set y-axis range to global maximum (rounded to power of 10)
                     if spectrum_scale in ["Log-Log", "Log-Linear"]:
                         # For log scale, set reasonable range
                         spec_fig.update_yaxes(range=[0, np.log10(y_axis_max)])
                     else:
                         # For linear scale, set to global maximum
                         spec_fig.update_yaxes(range=[0, y_axis_max])
                         
                     st.plotly_chart(spec_fig, key=f"hxr_chart_{sig}")

                     # --- Download Buttons for HXR Data ---
                     if export_data:
                         st.markdown(f"**Download Processed HXR Data ({sig}):**")
                         df_all = pd.DataFrame(export_data)
                         
                         # Create a row of columns for buttons
                         # Limit columns to 4 per row for better UI
                         if len(selected_shots) > 0:
                            cols = st.columns(min(len(selected_shots), 4))
                            for i, shot in enumerate(selected_shots):
                                df_shot = df_all[df_all["Shot"] == shot]
                                if not df_shot.empty:
                                    # CSV
                                    csv = df_shot.to_csv(index=False).encode('utf-8')
                                    with cols[i % 4]:
                                        st.download_button(
                                            label=f"📥 {shot} (.csv)",
                                            data=csv,
                                            file_name=f"{sig}_{shot}_spectrum.csv",
                                            mime='text/csv',
                                            key=f"dl_hxr_{sig}_{shot}"
                                        )
            
            if not overall_hxr_found:
                 st.info("No LaBr3 energy data found for selected discharges.")

        # --- Download Section ---
        st.markdown("---")
        st.subheader("⬇️ Download Discharge Data")

        base_root = utils.get_data_root()
        
        # Grid layout for buttons
        cols_dl = st.columns(3) 

        for i, shot in enumerate(selected_shots):
            zip_filename = f"{shot}.zip"
            zip_path = f"{base_root}/{zip_filename}" if base_root else zip_filename
            
            # Check existence and get ID
            file_id = utils.get_id_from_path(zip_path)
            
            with cols_dl[i % 3]:
                if file_id:
                    # Direct download link for Google Drive file
                    url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    st.link_button(f"Download {shot}.zip", url)
                else:
                    st.warning(f"Zip not found: {shot}")
