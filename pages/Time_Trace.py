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
        default=[all_available_shots[0]] if all_available_shots else [],
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
            defaults_req = ["IOH1", "IT1", "ITV1", "VP0", "IP1", "HCN1", "HA"]
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
    st.sidebar.header("Plotting Time Range")
    t0 = st.sidebar.number_input("Plot Start time (ms)", value=250, key="plot_t0") # Changed default to match typical search
    t1 = st.sidebar.number_input("Plot End time (ms)", value=500, key="plot_t1")


    # --- Automatic Y-axis Labels Dictionary  ---
    ylabels = {}
    for sig in signal_list:
        sig_upper = sig.upper()
        if sig_upper.startswith("IP"):
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
        else:
            ylabels[sig] = f"{sig}"


    # --- Color Map for Shots ---
    unique_shots = list(set(selected_shots))
    base_colors = px.colors.qualitative.Plotly
    color_map = {shot: base_colors[i % len(base_colors)] for i, shot in enumerate(unique_shots)}


    # --- Subplot Layout Option  ---
    layout_option = st.sidebar.selectbox(
        "Subplot layout",
        options=["Vertical", "Horizontal", "Grid (Auto)"]
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
    
    # Toggle for Highlight
    highlight_interval = st.sidebar.toggle("Highlight Discharge Interval", value=False)
    
    if st.sidebar.button("Load and Plot"):
        st.session_state["load_and_plot_clicked"] = True


    # --- Plotting Logic ---
    if st.session_state["load_and_plot_clicked"] and selected_shots and signal_list:
        st.write("### Plotting Signals")
        
        # Pre-calculate durations if highlight is ON
        shot_durations = {}
        if highlight_interval:
            for shot in selected_shots:
                # Load IP1 
                ip_cols = ["IP1", "IP"] 
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
                filename = f"{sig}.txt"
                file_path = os.path.join(BASE_PATH, shot)
                full_path = os.path.join(file_path, filename)

                if not os.path.exists(full_path):
                    st.warning(f"File not found: {full_path} for Shot {shot}, Signal {sig}")
                    continue

                try:
                    df = load_signal(file_path, filename)
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
