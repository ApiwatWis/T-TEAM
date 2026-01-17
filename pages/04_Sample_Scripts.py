import streamlit as st

st.set_page_config(page_title="Sample Scripts", layout="wide")

st.title("📜 Sample Scripts")
st.write("Below are Python snippets to help you load and process TT-1 data locally.")

st.header("1. Loading Standard Signal (.txt)")
st.markdown("Most magnetic and current measurements are stored in `.txt` files with an 8-line header.")
st.code("""
import pandas as pd
import os

def load_signal(shot_id, signal_name):
    path = f"data/{shot_id}/{signal_name}.txt"
    # Standard TT-1 txt files have 8 lines of header
    df = pd.read_csv(path, skiprows=8, delimiter=r'\\s+', header=None, names=["Time", "Value"])
    return df

# Example usage
shot = "1641"
df_ip = load_signal(shot, "IP2")
print(f"Max Ip for shot {shot}: {df_ip['Value'].max()/1000:.2f} kA")
""", language="python")

st.header("2. Processing LaBr3 Data (.CSV)")
st.markdown("LaBr3 detectors output raw `TIMETAG` and `ENERGY` channels. Use the following constants for conversion.")
st.code("""
import pandas as pd
import numpy as np

def process_labr3(shot_id, detector="LaBr3_1"):
    path = f"data/{shot_id}/{detector}.CSV"
    df = pd.read_csv(path, delimiter=';')
    
    # Constants
    bit = 2**12
    sampling_rate = 4e-9 # 4 ns
    
    # Time conversion to milliseconds
    time_ms = (df['TIMETAG'] / bit * sampling_rate) * 1000.0
    
    # Energy conversion (keV) - Factory Calibration
    energy_kev = 2.8775 * df['ENERGY'] - 66.0934
    
    return time_ms, energy_kev

time, energy = process_labr3("1641")
""", language="python")

st.header("3. Plotting with Matplotlib")
st.code("""
import matplotlib.pyplot as plt

# Assuming df_ip is loaded from snippet #1
plt.figure(figsize=(10, 4))
plt.plot(df_ip['Time'], df_ip['Value']/1000, label='IP2')
plt.xlabel('Time (ms)')
plt.ylabel('Plasma Current (kA)')
plt.title('Discharge Trace')
plt.grid(True)
plt.legend()
plt.show()
""", language="python")

st.info("💡 You can also export processed data (like HXR spectra) directly from the **Time Trace** page using the 'Download' buttons.")
