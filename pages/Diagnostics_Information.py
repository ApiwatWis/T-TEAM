import streamlit as st
import os

def show_diagnostics_info():
    st.title("ℹ️ TT-1 Device & Diagnostics Information")

    # --- 1. Machine Specifications (Source: Suksaengpanomrung et al., 2025) ---
    st.header("1. Thailand Tokamak-1 (TT-1) Specifications")
    st.markdown("""
    **Thailand Tokamak-1 (TT-1)** is a small research tokamak operated by the Thailand Institute of Nuclear Technology (TINT). 
    It is a reconstruction of the **HT-6M** tokamak donated by ASIPP, China.
    """)

    # Display TT-1 Magnet System Figure
    image_path = "assets/SS01_TT1Cad.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, caption="Fig. 1. (a) CAD drawing illustrating the magnet system of TT-1. (b) Diagram showing the positioning of magnetic coils in the poloidal cross-section plane.", use_container_width=True)
        st.caption("Source: [Suksaengpanomrung et al., Fusion Engineering and Design (2024)](https://doi.org/10.1016/j.fusengdes.2024.114781)")
    else:
        st.info("💡 **Figure 1 (TT-1 Magnet System)** will appear here.  \n*Source: [Suksaengpanomrung et al. (2024)](https://doi.org/10.1016/j.fusengdes.2024.114781)*")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📏 Geometric Parameters")
        st.write("- **Major Radius ($R_0$):** 0.65 m")
        st.write("- **Minor Radius ($a$):** 0.20 m")
        st.write("- **Aspect Ratio ($R/a$):** 3.25")
        st.write("- **Vacuum Chamber:** Circular poloidal cross-section")
        st.write("- **Port Access:** 34 ports (14 horizontal, 20 vertical)")

    with col2:
        st.markdown("#### ⚡ Operational Parameters")
        st.write("- **Toroidal Magnetic Field ($B_t$):** ≤ 1.5 T (Rated)")
        st.write("- **Plasma Current ($I_p$):** ≤ 150 kA")
        st.write("- **Discharge Duration:** ~100 ms (Flattop)")
        st.write("- **Heating Method:** Ohmic Heating (Primary)")
        st.write("- **Limiter System:** 1 Fixed + 2 Movable Poloidal Limiters")

    st.divider()

    # --- 2. HXR Diagnostics (Source: Rongpuit et al., 2024) ---
    st.header("2. Hard X-Ray (HXR) Spectroscopy")
    st.markdown("""
    The HXR diagnostic system is designed to study **Runaway Electrons (RE)** and suprathermal electron dynamics.
    It utilizes a high-performance **LaBr$_3$(Ce)** scintillation detector.
    """)

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**🔍 Detector Type**")
        st.info("**LaBr$_3$(Ce)**\n(Cerium-doped Lanthanum Bromide)")
    
    with c2:
        st.markdown("**⚡ Key Performance**")
        st.write("- **Energy Resolution:** ~3% @ 662 keV (Excellent)")
        st.write("- **Decay Time:** ~16 ns (Very Fast)")
        st.write("- **Detection Range:** ~50 keV – Several MeV")

    with c3:
        st.markdown("**🎯 Scientific Goal**")
        st.write("Investigation of **Runaway Electron** generation and loss mechanisms during plasma disruptions and flat-top phases.")

    st.divider()

    # --- 3. Magnetic Diagnostics (General TT-1) ---
    st.header("3. Magnetic Diagnostics")
    st.write("The magnetic measurement system consists of various electromagnetic probes installed inside the vacuum vessel:")
    
    mag_data = {
        "Sensor Type": [
            "Rogowski Coils", 
            "Flux Loops", 
            "Poloidal Magnetic Probes (Mirnov)", 
            "Diamagnetic Loop"
        ],
        "Measurement": [
            "Plasma Current ($I_p$)", 
            "Loop Voltage ($V_{loop}$) & Poloidal Flux", 
            "Local Poloidal Magnetic Field ($B_\\theta$) & MHD Oscillations", 
            "Plasma Stored Energy ($W_{dia}$)"
        ]
    }
    st.table(mag_data)

# Allow standalone execution for testing
if __name__ == "__main__":
    show_diagnostics_info()