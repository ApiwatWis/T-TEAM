import streamlit as st
import cv2
import os
import numpy as np
import utils
import tempfile

if not utils.check_auth():
    st.stop()

st.title("CCD Movie Analysis")

# Main path for data files - relative to the repo root
BASE_PATH = utils.get_data_root()

# Function to get all available shots
@st.cache_data
def get_all_available_shots(base_path):
    shots = utils.list_github_dirs(base_path)
    numeric_shots = [s for s in shots if s.isdigit()]
    return sorted(numeric_shots, key=lambda x: int(x), reverse=True)


def Convert_frame_to_time_label(frame_index, fps, t0, shot_number):
    """
    Convert frame index to time label in ms.
    """
    time_ms = t0 + (frame_index / fps) * 1000  # Convert to ms
    text_time = f'Time: {time_ms:.1f} ms'
    if shot_number is not None:
        text = f'Shot: {shot_number}, ' + text_time
    else:
        text = f'Shot: Unknown, ' + text_time
    return text

def process_video(input_path, output_dir, output_base, t0, fps, shot_number):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, f"Could not open video file '{input_path}'", None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use H.264 codec for MP4 container (best for browser playback)
    # FORCE PLAYBACK FPS to 30 to avoid encoding errors with high acquisition FPS (e.g. 2000)
    playback_fps = 30.0 
    
    # Codec priority list: (FourCC, Extension)
    codecs = [
        ('avc1', '.mp4'),  # H.264 (Best for web)
        ('vp09', '.webm'), # VP9 (Good for web)
        ('VP80', '.webm'), # VP8 (Good for web)
        ('mp4v', '.mp4')   # MPEG-4 (Fallback, might not play in all browsers)
    ]

    out = None
    final_output_path = None

    for codec, ext in codecs:
        temp_path = os.path.join(output_dir, output_base + ext)
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Attempt to initialize writer
            test_out = cv2.VideoWriter(temp_path, fourcc, playback_fps, (frame_width, frame_height))
            if test_out.isOpened():
                out = test_out
                final_output_path = temp_path
                print(f"VideoWriter initialized with codec: {codec}")
                break
            else:
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass
        except Exception as e:
            if 'test_out' in locals() and test_out: test_out.release()
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass
            continue

    if not out or not out.isOpened():
        cap.release()
        return False, "Could not open video writer for output (no supported codec found).", None

    frame_count = 0
    
    # Get total frames approx for progress
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # --- Add text to the frame ---
        time_ms = t0 + ((frame_count - 1) / fps) * 1000
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)  # White color
        thickness = 2
        
        # Line 1: Thailand Tokamak-1
        cv2.putText(frame, "Thailand Tokamak-1", (20, 40), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        # Line 2: Time
        cv2.putText(frame, f"Time: {time_ms:.1f} ms", (20, 80), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        out.write(frame)
        
        if total_frames > 0:
             progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()
    return True, "Success", final_output_path

# --- Main Page Layout ---

all_shots = get_all_available_shots(BASE_PATH)

if not all_shots:
    st.error(f"No discharge folders found in '{BASE_PATH}'.")
else:
    selected_shot = st.selectbox("Select Discharge", all_shots)
    
    st.markdown("### Settings")
    col1, col2 = st.columns(2)
    with col1:
        t0 = st.number_input("Start Time (t0) [ms]", value=260.0, step=1.0)
    with col2:
        fps = st.number_input("Frame Rate (FPS)", value=2000.0, step=100.0)

    # Define Paths
    shot_dir = os.path.join(BASE_PATH, selected_shot)
    input_filename = f"{selected_shot}.avi"
    input_path_repo = os.path.join(shot_dir, input_filename)
                   
    output_base = f"{selected_shot}_time_label"
    
    # Use Temp Directory for Processing
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.TemporaryDirectory()
    
    # Use the persistent temp directory name
    temp_dir = st.session_state.temp_dir.name
    local_input_path = os.path.join(temp_dir, input_filename)
    
    # Check for existing processed video (mp4 or webm)
    existing_video = None
    for ext in ['.mp4', '.webm']:
        check_path = os.path.join(temp_dir, output_base + ext)
        if os.path.exists(check_path):
            existing_video = check_path
            break

    st.markdown("---")

    # Check for processed video in temp storage
    if existing_video:
        file_name = os.path.basename(existing_video)
        st.success(f"Processed video available: `{file_name}`")
        st.video(existing_video)
        with open(existing_video, "rb") as file:
            st.download_button("Download Processed Video", file, file_name=file_name, mime="video/mp4" if file_name.endswith(".mp4") else "video/webm")
        st.markdown("---")

    # Check existence on GitHub
    input_exists = utils.github_file_exists(input_path_repo)

    if input_exists:
        st.write(f"Original video source: `{input_filename}`")
        st.info("Note: Video will be downloaded from GitHub for processing.")
        
        btn_label = "Regenerate Video" if existing_video else "Generate Video"
        
        if st.button(btn_label):
            # Download first
            # Check if file needs to be downloaded (if not already there or to overwrite)
            if not os.path.exists(local_input_path):
                 with st.spinner("Downloading video from GitHub..."):
                     data = utils.read_github_file_binary(input_path_repo)
                     if data:
                         with open(local_input_path, "wb") as f: f.write(data)
                     else:
                         st.error("Download failed to local storage.")
                         st.stop()
            
            # Cleanup previous outputs before processing to avoid extension conflicts
            for ext in ['.mp4', '.webm']:
                prev_file = os.path.join(temp_dir, output_base + ext)
                if os.path.exists(prev_file):
                    os.remove(prev_file)

            with st.spinner("Processing video..."):
                success, msg, final_path = process_video(local_input_path, temp_dir, output_base, t0, fps, selected_shot)
                if success:
                    st.success(f"Video processed successfully! Saved as {os.path.basename(final_path)}")
                    st.rerun()
                else:
                    st.error(f"Error: {msg}")
    else:
        st.warning(f"No video file found for {selected_shot} in repository.")
