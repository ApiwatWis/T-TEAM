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
        
        # Time conversion based on typical LaBr3 setup (checking Sample Scripts)
        # TIMETAG is shifted by 12 bits (4096) and clock is 250 MHz (4ns)
        # time_sec = TIMETAG / 4096 * 4e-9
        
        sampling_rate = 4e-9 
        bit_shift = 4000 # 2**12
        
        time_sec = timetag_0 / bit_shift * sampling_rate    
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
            
            start_t = np.floor(max(0, center_time - max_duration/2))
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
        
        rate = counts  / (bin_size / 1000.0) # Hz
        
        return pd.DataFrame({
            "Time": bin_centers,
            detector_name.upper(): rate,
            "Counts": counts
        })

    except Exception as e:
        return pd.DataFrame()

def load_labr3_energy(base_path, shot_id, detector_name, t_start=None, t_end=None):
    """
    Loads LaBr3 energy data (keV) filtered by time range.
    detector_name: 'LaBr3_1' or 'LaBr3_2'
    t_start, t_end: Time range in ms (optional)
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
            return np.array([])

        df = pd.read_csv(io.StringIO(content), delimiter=';', header=0)
        if df.empty or 'ENERGY' not in df.columns:
            return np.array([])

        # Filter by time if range is provided
        if t_start is not None and t_end is not None and 'TIMETAG' in df.columns:
            timetag_0 = df['TIMETAG'].values
            
            sampling_rate = 4e-9 
            bit_shift = 2**12
            
            time_sec = timetag_0 / bit_shift * sampling_rate
            time_ms = time_sec * 1000.0
            
            # Create time mask
            time_mask = (time_ms >= t_start) & (time_ms <= t_end)
            df = df[time_mask]
        
        if df.empty:
            return np.array([])
        
        q = df['ENERGY'].values
        
        if len(q) == 0:
            return np.array([])
            
        # Apply energy calibration
        if detector_name == "LaBr3_2":
            # Formula: E = 1.85*q - 57.44
            E = 1.85 * q - 57.44
        else:
            # Default / LaBr3_1 Formula: E = 2.8775*q - 66.0934
            E = 2.8775 * q - 66.0934
            
        return E
    except Exception as e:
        return np.array([])

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

if "hxr_spectrum_t0" not in st.session_state:
    st.session_state["hxr_spectrum_t0"] = 250

if "hxr_spectrum_t1" not in st.session_state:
    st.session_state["hxr_spectrum_t1"] = 500

if "hxr_spectrum_plot_clicked" not in st.session_state:
    st.session_state["hxr_spectrum_plot_clicked"] = False

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
    
    # Line style selector
    line_style_option = st.sidebar.radio(
        "Line Style",
        ["Line", "Dots + Line", "Dots"],
        key="hxr_line_style",
        horizontal=True
    )
    
    # Map line style option to plotly mode
    if line_style_option == "Dots":
        plot_mode = "markers"
    elif line_style_option == "Dots + Line":
        plot_mode = "lines+markers"
    else:  # Line
        plot_mode = "lines"
    
    # Time Range Inputs
    st.sidebar.markdown("---")
    st.sidebar.header("Time Range")
    
    # Reset button (placed before inputs to process first)
    if st.sidebar.button("Reset to Default (250-500 ms)", key="hxr_reset_time_range"):
        st.session_state["hxr_plot_t0"] = 250
        st.session_state["hxr_plot_t1"] = 500
        # Clear widget states to force update
        if "hxr_t0_input" in st.session_state:
            del st.session_state["hxr_t0_input"]
        if "hxr_t1_input" in st.session_state:
            del st.session_state["hxr_t1_input"]
        st.rerun()
    
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
    
    # HXR Spectrum Options
    st.sidebar.markdown("---")
    st.sidebar.header("HXR Spectrum Options")
    
    show_hxr_spectrum = st.sidebar.toggle(
        "Show HXR Energy Spectra", 
        value=False, 
        key="hxr_show_spectrum",
        help="Plot energy spectra for selected time interval"
    )
    
    spectrum_scale = "Semi-log (Linear-Log)"
    if show_hxr_spectrum:
        st.sidebar.markdown("**Spectrum Time Interval**")
        
        spec_t0 = st.sidebar.number_input(
            "Spectrum Start time (ms)",
            value=st.session_state["hxr_spectrum_t0"],
            key="hxr_spectrum_t0_input"
        )
        spec_t1 = st.sidebar.number_input(
            "Spectrum End time (ms)",
            value=st.session_state["hxr_spectrum_t1"],
            key="hxr_spectrum_t1_input"
        )
        
        # Update session state
        st.session_state["hxr_spectrum_t0"] = spec_t0
        st.session_state["hxr_spectrum_t1"] = spec_t1
        
        if st.sidebar.button("Re-plot Spectra", key="hxr_replot_spectrum"):
            st.session_state["hxr_spectrum_plot_clicked"] = True
        
        spectrum_scale = st.sidebar.radio(
            "Spectrum Plot Scale",
            ["Semi-log (Linear-Log)", "Linear-Linear", "Log-Log"],
            key="hxr_spectrum_scale",
            help="Semi-log: x-axis linear, y-axis log10 (default)"
        )

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

        # Create specs for secondary y-axis on LaBr3 plots
        specs = []
        for sig in all_signals:
            if sig.upper().startswith("LABR3"):
                specs.append([{"secondary_y": True}])
            else:
                specs.append([{"secondary_y": False}])

        # Create subplots
        fig = make_subplots(
            rows=subplot_rows,
            cols=subplot_cols,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing_val,
            subplot_titles=[st_titles.get(sig, sig) for sig in all_signals],
            specs=specs
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
            max_counts_for_scaling = 0

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

                    # Check if dual axis is needed
                    is_labr3 = sig.upper().startswith("LABR3")

                    # Legend optimization
                    show_legend = False
                    if shot not in legend_shots_added:
                        show_legend = True
                        legend_shots_added.add(shot)
                    
                    y_data_for_annot = None 

                    if is_labr3:
                        y_data_for_annot = df["Counts"]
                        current_max_counts = df["Counts"].max()
                        if current_max_counts > max_counts_for_scaling:
                            max_counts_for_scaling = current_max_counts
                        
                        # Plot Counts (Left Axis)
                        fig.add_trace(go.Scatter(
                            x=df["Time"],
                            y=df["Counts"],
                            name=f"{shot}",
                            legendgroup=f"group_{shot}",
                            showlegend=show_legend,
                            mode=plot_mode,
                            line=dict(color=color_map.get(shot, "gray")),
                            marker=dict(size=4, color=color_map.get(shot, "gray")) if "markers" in plot_mode else None
                        ), row=row, col=col, secondary_y=False)
                        
                        # Plot Rate (Right Axis)
                        fig.add_trace(go.Scatter(
                            x=df["Time"],
                            y=df[sig.upper()],
                            name=f"{shot} (cps)",
                            legendgroup=f"group_{shot}",
                            showlegend=False,
                            mode=plot_mode,
                            line=dict(color=color_map.get(shot, "gray")),
                            marker=dict(size=4, color=color_map.get(shot, "gray")) if "markers" in plot_mode else None
                        ), row=row, col=col, secondary_y=True)

                    else:
                        # Scale current signals to kA
                        y_data = df[sig.upper()]
                        if "I" in sig.upper():
                            y_data = y_data / 1000
                        
                        y_data_for_annot = y_data

                        # Add trace
                        fig.add_trace(go.Scatter(
                            x=df["Time"],
                            y=y_data,
                            name=f"{shot}",
                            legendgroup=f"group_{shot}",
                            showlegend=show_legend,
                            mode=plot_mode,
                            line=dict(color=color_map.get(shot, "gray")),
                            marker=dict(size=4, color=color_map.get(shot, "gray")) if "markers" in plot_mode else None
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
                            # Check if y_data_for_annot is valid
                            if y_data_for_annot is not None and len(y_data_for_annot) > 0:
                                y_max_val = np.max(y_data_for_annot)
                                fig.add_annotation(
                                    x=s_time + (e_time - s_time) / 2,
                                    y=y_max_val * 0.9,
                                    text=f"{duration_val:.1f} ms",
                                    showarrow=False,
                                    yshift=10,
                                    font=dict(color=color_map.get(shot, "black"), size=10),
                                    row=row, col=col
                                )

                except Exception as e:
                    pass

            # Update axes
            if sig.upper().startswith("LABR3"):
                # Synchronize axes ranges
                if max_counts_for_scaling > 0:
                    ymax = max_counts_for_scaling * 1.1 # 10% padding
                    
                    # Ensure range includes the limit line if it's close? 
                    # Limit is 300,000 cps -> 300 counts.
                    # If max count is > 300, we are good.
                    # If max count is small, say 10, then 300 is far above. 
                    # If we force range to data, line is off chart.
                    
                    # Primary (Counts)
                    fig.update_yaxes(
                        title_text="Counts", 
                        row=row, col=col, 
                        secondary_y=False,
                        range=[0, ymax]
                    )
                    
                    # Secondary (Rate)
                    # Rate = Counts * 1000
                    fig.update_yaxes(
                        title_text="Count Rate (cps)", 
                        row=row, col=col, 
                        secondary_y=True,
                        range=[0, ymax * 1000]
                    )
                else:
                     # Fallback if no data
                    fig.update_yaxes(title_text="Counts", row=row, col=col, secondary_y=False)
                    fig.update_yaxes(title_text="Count Rate (cps)", row=row, col=col, secondary_y=True)

                # Add Count Rate limit line
                fig.add_hline(
                    y=300000, 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text="Limit (300kcps)", 
                    annotation_position="top right",
                    row=row, 
                    col=col, 
                    secondary_y=True
                )
            else:
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
        
        # Display current data range
        st.markdown(f"**📊 Loaded Data Range:** {t0} - {t1} ms | **Duration:** {t1 - t0} ms")
        
        # Display plot with box selection event handling
        selection = st.plotly_chart(
            fig, 
            key="hxr_main_plot", 
            use_container_width=True,
            on_select="rerun",
            selection_mode="box"
        )
        
        # Handle box selection to update time range
        if selection:
            # Debug: show what we received
            # st.write("Selection object:", selection)
            
            if hasattr(selection, 'selection') and selection.selection:
                # st.write("Selection data:", selection.selection)
                
                # Try different ways to access box selection data
                try:
                    # Method 1: Direct box access
                    if 'box' in selection.selection:
                        boxes = selection.selection['box']
                        if boxes and len(boxes) > 0:
                            box = boxes[0]
                            if 'x' in box and len(box['x']) >= 2:
                                new_t0 = round(min(box['x']), 1)
                                new_t1 = round(max(box['x']), 1)
                                
                                if (abs(new_t0 - st.session_state["hxr_plot_t0"]) > 0.1 or 
                                    abs(new_t1 - st.session_state["hxr_plot_t1"]) > 0.1):
                                    st.session_state["hxr_plot_t0"] = new_t0
                                    st.session_state["hxr_plot_t1"] = new_t1
                                    st.rerun()
                    
                    # Method 2: Check for x0, x1 in selection
                    elif 'x0' in selection.selection and 'x1' in selection.selection:
                        new_t0 = round(min(selection.selection['x0'], selection.selection['x1']), 1)
                        new_t1 = round(max(selection.selection['x0'], selection.selection['x1']), 1)
                        
                        if (abs(new_t0 - st.session_state["hxr_plot_t0"]) > 0.1 or 
                            abs(new_t1 - st.session_state["hxr_plot_t1"]) > 0.1):
                            st.session_state["hxr_plot_t0"] = new_t0
                            st.session_state["hxr_plot_t1"] = new_t1
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error processing selection: {e}")
        
        # Info about zoom functionality
        st.info("💡 **Tip**: Use the Box Select tool (📦) from the toolbar, then drag on the plot to select a time range. The sidebar inputs will update automatically.")
        
        # --- HXR Energy Spectrum Plotting ---
        if show_hxr_spectrum:
            st.markdown("---")
            st.write("### HXR Energy Spectra (Time-Filtered)")
            
            # Use spectrum-specific time range
            spec_t0 = st.session_state["hxr_spectrum_t0"]
            spec_t1 = st.session_state["hxr_spectrum_t1"]
            
            st.caption(f"Energy spectra for events within time range: {spec_t0} - {spec_t1} ms")
            
            # Check if LaBr3 signals are available
            labr3_detectors = ["LaBr3_1", "LaBr3_2"]
            labr3_with_data = []
            
            # Check which detectors have data
            for detector in labr3_detectors:
                for shot in selected_shots:
                    energies = load_labr3_energy(BASE_PATH, shot, detector, spec_t0, spec_t1)
                    if len(energies) > 0:
                        labr3_with_data.append(detector)
                        break
            
            labr3_with_data = list(set(labr3_with_data))  # Remove duplicates
            
            if not labr3_with_data:
                st.warning("No LaBr3 energy data available for the selected discharges and time range.")
            else:
                # Plot spectrum for each detector separately
                for detector in sorted(labr3_with_data):
                    spec_fig = go.Figure()
                    has_data = False
                    
                    # Collect all counts to find global maximum for y-axis scaling
                    all_counts = []
                    
                    for shot in selected_shots:
                        energies = load_labr3_energy(BASE_PATH, shot, detector, spec_t0, spec_t1)
                        
                        if len(energies) > 0:
                            has_data = True
                            
                            # Energy binning: 1 keV bins from 0 to 2000 keV
                            min_e = 0.0
                            max_e = 2000.0
                            
                            bins = np.arange(min_e - 0.5, max_e + 1.5, 1.0)
                            counts, bin_edges = np.histogram(energies, bins=bins)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                            
                            # Filter to show only non-zero counts
                            mask = counts > 0
                            x_plot = bin_centers[mask]
                            y_plot = counts[mask]
                            
                            all_counts.extend(y_plot)
                            
                            # Add trace
                            spec_fig.add_trace(go.Scatter(
                                x=x_plot,
                                y=y_plot,
                                mode='lines+markers',
                                name=f"{shot}",
                                line=dict(color=color_map.get(shot, "gray")),
                                marker=dict(size=4)
                            ))
                    
                    if has_data:
                        # Calculate y-axis range
                        if len(all_counts) > 0:
                            global_max = np.max(all_counts)
                            y_axis_max = 10 ** np.ceil(np.log10(global_max)) if global_max > 0 else 10
                        else:
                            y_axis_max = 10
                        
                        # Determine axis types based on scale selection
                        if spectrum_scale == "Semi-log (Linear-Log)":
                            x_type = "linear"
                            y_type = "log"
                        elif spectrum_scale == "Log-Log":
                            x_type = "log"
                            y_type = "log"
                        else:  # Linear-Linear
                            x_type = "linear"
                            y_type = "linear"
                        
                        # Update layout
                        spec_fig.update_layout(
                            title=f"HXR Energy Spectrum - {detector} ({spectrum_scale})",
                            xaxis_title="Energy (keV)",
                            yaxis_title="Counts",
                            xaxis_type=x_type,
                            yaxis_type=y_type,
                            height=450,
                            hovermode="x unified"
                        )
                        
                        # Set axis ranges
                        if x_type == "log":
                            spec_fig.update_xaxes(range=[np.log10(1), np.log10(2000)])
                        else:
                            spec_fig.update_xaxes(range=[1, 2000])
                        
                        if y_type == "log":
                            spec_fig.update_yaxes(range=[0, np.log10(y_axis_max)])
                        else:
                            spec_fig.update_yaxes(range=[0, y_axis_max])
                        
                        # Display the plot
                        st.plotly_chart(spec_fig, key=f"hxr_spectrum_{detector}", use_container_width=True)
                
                st.info(f"📊 **Info**: Spectra calculated from events in time window {spec_t0} - {spec_t1} ms using 1 keV energy bins.")
