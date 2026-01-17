import streamlit as st
import cv2
import os
import numpy as np

st.title("CCD Movie Analysis")

# Main path for data files - relative to the workspace
BASE_PATH = os.path.join(os.getcwd(), "data")

# Function to get all available shots
@st.cache_data
def get_all_available_shots(base_path):
    if not os.path.isdir(base_path):
        return []
    shots = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    try:
        return sorted(shots, key=lambda x: int(x) if x.isdigit() else x)
    except:
        return sorted(shots)

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

def process_video(input_path, output_path, t0, fps, shot_number):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, f"Could not open video file '{input_path}'"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use H.264 codec for MP4 container (best for browser playback)
    # FORCE PLAYBACK FPS to 30 to avoid encoding errors with high acquisition FPS (e.g. 2000)
    playback_fps = 30.0 
    
    # Try avc1 (H.264) first, fallback to mp4v if needed
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, playback_fps, (frame_width, frame_height))
        if not out.isOpened():
             raise Exception("avc1 failed")
    except:
        # Fallback to mp4v (MPEG-4), might not play in all browsers but works in .mp4 container
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, playback_fps, (frame_width, frame_height))

    if not out.isOpened():
        cap.release()
        return False, "Could not open video writer for output."

    frame_count = 0
    progress_bar = st.progress(0)
    
    # Get total frames approx for progress
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        label_text = Convert_frame_to_time_label(frame_count - 1, fps=fps, t0=t0, shot_number=shot_number)

        # --- Add text to the frame ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)  # White color
        thickness = 2
        position = (20, 40)

        cv2.putText(frame, label_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
        out.write(frame)
        
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()
    return True, "Success"

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

    # Paths
    shot_dir = os.path.join(BASE_PATH, selected_shot)
    input_filename = f"{selected_shot}.avi"
    input_path = os.path.join(shot_dir, input_filename)
    output_filename = f"{selected_shot}_time_label.mp4"
    output_path = os.path.join(shot_dir, output_filename)

    st.markdown("---")

    # Check for processed video first
    if os.path.exists(output_path):
        st.success(f"Processed video found: `{output_filename}`")
        
        # Display video player
        st.video(output_path)
            
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name=output_filename,
                mime="video/mp4"
            )
        st.markdown("---")

    # Check for input video and generation options
    if os.path.exists(input_path):
        st.write(f"Original video source: `{input_filename}`")
        st.info("Note: The video format is .avi. Visual preview might not be supported in this browser.")
        
        button_label = "Regenerate Video with Timestamp" if os.path.exists(output_path) else "Generate Video with Timestamp"
        
        if st.button(button_label):
            with st.spinner("Processing video..."):
                success, msg = process_video(input_path, output_path, t0, fps, selected_shot)
                if success:
                    st.success(f"Video processed successfully! Saved to: `{output_path}`")
                    st.rerun() # Rerun to show the new video immediately
                else:
                    st.error(f"Error: {msg}")
    else:
        st.warning(f"No original movie file (.avi) found for discharge {selected_shot}. Expected path: `{input_path}`")
