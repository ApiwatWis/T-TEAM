import streamlit as st
import os
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Set page config
# st.set_page_config(layout="wide") # Commented out as likely handled by Home.py handling

st.title("Time Trace Analysis")
st.sidebar.title("Signal Dashboard")

# Main path for data files - relative to the workspace
BASE_PATH = os.path.join(os.getcwd(), "data")

# -----------------------------
#  load form .txt files
def load_signal(path, filename):
    """Loads signal data from a .txt file."""
    # Ensure filename is just the signal name (e.g., "IP1") and not "IP1.txt"
    signal_name = filename.replace(".txt", "") 
    try:
        data = pd.read_csv(
            os.path.join(path, filename),
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
        # File is now directly in the shot folder with name "LaBr3_1.CSV" or "LaBr3_2.CSV"
        filename = f"{detector_name}.CSV"
        file_path = os.path.join(base_path, shot_id, filename)

        if not os.path.exists(file_path):
            return pd.DataFrame()

        # Check if file is empty or just header
        if os.stat(file_path).st_size < 100: # Heuristic for empty file
             # Try reading to be sure
             try:
                 df_test = pd.read_csv(file_path, delimiter=';', nrows=1)
                 if df_test.empty: return pd.DataFrame()
             except:
                 return pd.DataFrame()

        df = pd.read_csv(file_path, delimiter=';', header=0)
        if df.empty or 'TIMETAG' not in df.columns:
            return pd.DataFrame()

        timetag_0 = df['TIMETAG'].values
        
        # User Constants
        bit = 2**12
        ts = 4e-9
        
        time_sec = timetag_0 / bit * ts 
        time_ms = time_sec * 1000.0

        if len(time_ms) == 0:
            return pd.DataFrame()
        
        # Binning for Count Rate (1ms bin)
        start_t = np.floor(np.min(time_ms))
        end_t = np.ceil(np.max(time_ms))
        if end_t <= start_t:
             # Handle single point or weird range
             start_t = np.min(time_ms)
             end_t = np.max(time_ms) + 1.0

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
        # st.error(f"Error loading LaBr3: {e}")
        return pd.DataFrame()

def load_labr3_energy(base_path, shot_id, detector_name):
    """
    Loads LaBr3 energy data (keV) for the whole discharge.
    """
    try:
        filename = f"{detector_name}.CSV"
        file_path = os.path.join(base_path, shot_id, filename)

        if not os.path.exists(file_path):
            return np.array([])
            
        if os.stat(file_path).st_size < 100:
             return np.array([])

        df = pd.read_csv(file_path, delimiter=';', header=0)
        if df.empty or 'ENERGY' not in df.columns:
            return np.array([])

        q = df['ENERGY'].values
        # Formula: E = 2.8775*q - 66.0934
        if len(q) == 0:
            return np.array([])
            
        E = 2.8775 * q - 66.0934
        return E
    except Exception:
        return np.array([])

def get_available_signals(base_path, sample_shot):
    """Retrieves a list of available signal files (TXT) for a given shot."""
    if not sample_shot:
        return []
    shot_path = os.path.join(base_path, sample_shot)
    if not os.path.isdir(shot_path):
        return []

    files = os.listdir(shot_path)
    signals = [f[:-4] for f in files if f.endswith(".txt")]
    
 
    if "IP1" in signals and "Duration" not in signals: # Duration is based on IP1, so only add if IP1 exists
        signals.append("Duration")
    
    # Check LaBr3 in the shot folder
    for det in ["LaBr3_1", "LaBr3_2"]:
       fname = f"{det}.CSV"
       fpath = os.path.join(shot_path, fname)
       if os.path.exists(fpath):
           signals.append(det)

    return sorted(signals)

# Function to get all available shots (for search)
@st.cache_data # Cache this function as it lists directories
def get_all_available_shots(base_path):
    """Retrieves a list of all available shot directories."""
    if not os.path.isdir(base_path):
        return []
    
    shots = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
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

    selected_shots = st.sidebar.multiselect(
        "Select Discharges",
        options=all_available_shots,
        default=[all_available_shots[-1]] if all_available_shots else [],
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
            defaults_req = ["IOH1", "IT1", "IV2", "IF2", "VP0", "IP2", "HCN1", "HA", "LaBr3_1", "LaBr3_2"]
            default_signals = [sig for sig in defaults_req if sig in signal_choices]
            if not default_signals and len(signal_choices) >= 2:
                default_signals = signal_choices[:2]
            
            def set_defaults():
                st.session_state["selected_signals_ms"] = default_signals

            signal_list = st.sidebar.multiselect(
                "Select signal (for plotting)", 
                signal_choices, 
                default=default_signals,
                key="selected_signals_ms"
            )
            
            st.sidebar.button("Default plot", on_click=set_defaults)
    else:
        st.sidebar.info("Please select at least one discharge.")


    # --- Time Range Inputs (for plotting) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Plotting Options")
    
    use_long_names = st.sidebar.toggle("Use Descriptive Signal Names", value=True)
    highlight_interval = st.sidebar.toggle("Highlight Discharge Interval", value=False)
    
    # Check if LaBr3 signals are selected to show the spectrum toggle
    has_labr3_selected = any(s.startswith("LaBr3") for s in signal_list)
    show_hxr_spectrum = False
    if has_labr3_selected:
        show_hxr_spectrum = st.sidebar.toggle("Show HXR Spectrum (Whole discharge)", value=False)
    
    t0 = st.sidebar.number_input("Plot Start time (ms)", value=250, key="plot_t0") # Changed default to match typical search
    t1 = st.sidebar.number_input("Plot End time (ms)", value=500, key="plot_t1")


    # --- Automatic Y-axis Labels Dictionary  ---
    ylabels = {}
    for sig in signal_list:
        sig_upper = sig.upper()
        if not use_long_names:
            ylabels[sig] = sig_upper
        elif sig_upper.startswith("IP"):
            ylabels[sig] = "Plasma Current (kA)"
        elif sig_upper.startswith("IV"):
            ylabels[sig] = "Vertical Current (kA)"
        elif sig_upper.startswith("G"):
            ylabels[sig] = f"{sig} (T)"
        elif sig_upper.startswith("V"):
            ylabels[sig] = f"{sig} (V)"
        elif sig_upper.startswith("F"):
            ylabels[sig] = f"{sig} (Wb)"
        elif sig_upper.startswith("DIA"):
            ylabels[sig] = f"{sig} (Wb)"
        elif sig_upper.startswith("NE"): 
            ylabels[sig] = f"Electron Density (10^19 m^-3)" 
        elif sig_upper.startswith("VLOOP"): 
            ylabels[sig] = f"Loop Voltage (V)" 
        elif sig_upper.startswith("LABR3"):
            ylabels[sig] = f"Count Rate (Hz)"
        else:
            ylabels[sig] = f"{sig}"


    # --- Color Map for Shots ---
    unique_shots = list(set(selected_shots))
    base_colors = px.colors.qualitative.Plotly
    color_map = {shot: base_colors[i % len(base_colors)] for i, shot in enumerate(unique_shots)}


    # --- Subplot Layout Option  ---
    layout_option = st.sidebar.selectbox(
        "Subplot layout",
        options=["Grid (Auto)", "Vertical", "Horizontal"]
    )

    # --- Consolidated and Corrected Layout Calculation ---
    num_plots = len(signal_list) 
    subplot_rows, subplot_cols = 1, 1 

    if num_plots > 0: 
        if layout_option == "Vertical":
            subplot_rows = num_plots
            subplot_cols = 1
        elif layout_option == "Horizontal":
            subplot_rows = 1
            subplot_cols = num_plots
        else:  # Grid (Auto)
            subplot_rows = (num_plots + 1) // 2 
            subplot_cols = 2 if num_plots > 1 else 1 
    
    # --- Buttons and Interaction ---
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Load and Plot"):
        st.session_state["load_and_plot_clicked"] = True


    # --- Plotting Logic ---
    if st.session_state["load_and_plot_clicked"] and selected_shots and signal_list:
        st.write("### Plotting Signals")
        
        # Pre-calculate durations if highlight is ON
        shot_durations = {}
        if highlight_interval:
            for shot in selected_shots:
                # Load IP2 for duration calculation as requested
                ip_cols = ["IP2", "IP1", "IP"] 
                loaded = False
                for ip_name in ip_cols:
                    filename = f"{ip_name}.txt"
                    file_path = os.path.join(BASE_PATH, shot)
                    if os.path.exists(os.path.join(file_path, filename)):
                        try:
                            df_ip = load_signal(file_path, filename)
                            df_ip = df_ip[(df_ip["Time"] >= t0) & (df_ip["Time"] <= t1)]
                            if not df_ip.empty:
                                time_ip = df_ip["Time"].to_numpy()
                                data_ip = df_ip[ip_name.upper() if ip_name.upper() in df_ip.columns else df_ip.columns[1]].to_numpy() # Handle col names safely
                                
                                # Check if max current >= 2 kA for highlighting
                                # Assuming IP is in kA if absolute value > 100, wait, load_signal does not convert to kA by default. 
                                # But let's check values used in plotting. Plotting divides by 1000 if "I" in sig.
                                # Here we are using raw data. Let's assume raw data is in Amperes usually.
                                # Check if column name contains "IP".
                                max_val = np.max(data_ip)
                                # If signal is already in kA ? No, load_signal reads raw.
                                # User requirement: "below than 2 kA". Raw IP usually in Amperes.
                                # Let's assume 2000 A.
                                
                                threshold_amp = 2000
                                if max_val >= threshold_amp:
                                    # Use default 0.05 (5%) and 0.05 (5%) thresholds
                                    dur, s_idx, e_idx, s_time, e_time = cal_duration(data_ip, time_ip, start_threshold_ratio=0.05, end_threshold_ratio=0.05)
                                    shot_durations[shot] = {
                                        "duration": dur,
                                        "start_time": s_time,
                                        "end_time": e_time
                                    }
                                loaded = True
                                break
                        except Exception:
                            pass
        
        fig = make_subplots(
            rows=subplot_rows,
            cols=subplot_cols,
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.15,
            subplot_titles=[ylabels.get(sig.upper(), sig) for sig in signal_list]
        )

        for idx, sig in enumerate(signal_list):
            row = (idx // subplot_cols) + 1
            col = (idx % subplot_cols) + 1

            for shot in selected_shots: 
                df = pd.DataFrame()
                
                if sig.upper().startswith("LABR3"):
                    df = load_labr3_signal(BASE_PATH, shot, sig)
                    if df.empty:
                        # Optional: warn if data expected but missing. 
                        # But for LaBr3, user said it might be absent. 
                        # Sticking to valid data or warning.
                        # st.warning(f"No valid data for {sig} in Shot {shot}")
                        pass
                else:
                    filename = f"{sig}.txt"
                    file_path = os.path.join(BASE_PATH, shot)
                    full_path = os.path.join(file_path, filename)

                    if not os.path.exists(full_path):
                        st.warning(f"File not found: {full_path} for Shot {shot}, Signal {sig}")
                        continue

                    try:
                        df = load_signal(file_path, filename)
                    except Exception:
                        df = pd.DataFrame()

                try:
                    if df.empty:
                        continue
                        
                    df = df[(df["Time"] >= t0) & (df["Time"] <= t1)] 

                    y_data = df[sig.upper()]
                    if "I" in sig.upper():
                        y_data = y_data / 1000

                    fig.add_trace(go.Scatter(
                        x=df["Time"],
                        y=y_data,
                        name=f"{shot} - {sig.upper()}",
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

            fig.update_yaxes(
                title_text=ylabels.get(sig.upper(), " "),
                row=row,
                col=col
            )
            
            show_x_label = (row == subplot_rows)
            x_title = "Time (ms)" if show_x_label else None
            
            fig.update_xaxes(
                title_text=x_title,
                showticklabels=show_x_label,
                row=row,
                col=col
            )
            
        fig.update_layout(
            height=300 * subplot_rows,
            margin=dict(t=80, b=100, l=60, r=60),
            showlegend=True,
            title_text="Signal Dashboard",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- HXR Spectrum Plotting ---
        if has_labr3_selected and show_hxr_spectrum:
            st.markdown("---")
            st.write("### HXR Spectrum (Whole Discharge)")
            
            labr3_signals = [s for s in signal_list if s.startswith("LaBr3")]
            
            # Separate plots for each LaBr3 detector
            # Sort to keep order consistent if typical names used
            for sig in sorted(labr3_signals): 
                 spec_fig = go.Figure()
                 has_data_for_sig = False
                 
                 for shot in selected_shots:
                    energies = load_labr3_energy(BASE_PATH, shot, sig)
                    if len(energies) > 0:
                        has_data_for_sig = True
                        
                        # Binning: 1 keV bins, centered at integers
                        # Edges should be at k-0.5 and k+0.5 for Integer center k
                        min_e = np.floor(np.min(energies))
                        # Optional: limit to physical non-negative energy if desired, 
                        # but formula creates negatives for low channels. 
                        # Previous code clamped min to 0. Let's keep consistent if intended.
                        # However, let's just bin what we have, but start binning from floor-0.5
                        if min_e < 0: min_e = 0 
                        max_e = np.ceil(np.max(energies))
                        
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
                            
                            spec_fig.add_trace(go.Scatter(
                                x=x_plot,
                                y=y_plot,
                                mode='markers', # To plot as dots
                                name=f"{shot}",
                                marker=dict(color=color_map.get(shot, "gray"), size=4)
                            ))

                 if has_data_for_sig:
                    spec_fig.update_layout(
                        title=f"HXR Energy Spectrum - {sig}",
                        xaxis_title="Energy (keV)",
                        yaxis_title="Counts",
                        hovermode="closest",
                        height=400,
                        xaxis=dict(range=[0, 2000]) # Default 0-2000 keV as typical HXR range
                    )
                    st.plotly_chart(spec_fig, use_container_width=True)
                 else:
                    st.info(f"No HXR energy data available for {sig} in selected shots.")
