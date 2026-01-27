import streamlit as st
import os
import yaml
import numpy as np
import plotly.graph_objects as go
import utils

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

def get_cross_section_plot():
    # Load signals
    try:
        yaml_path = os.path.join(os.path.dirname(__file__), '../signals.yaml')
        with open(yaml_path, 'r') as f:
            signals = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading signals.yaml: {e}")
        return None

    fig = go.Figure()

    # --- 1. Vacuum Vessel (User Data) ---
    # RR: 0.650, aa: 0.254
    theta = np.linspace(0, 2*np.pi, 200)
    r_wall = 0.650 + 0.254 * np.cos(theta)
    z_wall = 0.254 * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=r_wall, y=z_wall,
        mode='lines',
        name='Vacuum Vessel',
        line=dict(color='black', width=3),
        hoverinfo='name'
    ))

    # --- 2. TF Coil (Approximation from config data) ---
    # mean radius approx: 0.386, thickness: 0.108
    thickness_tf = 0.0408  # mocked thickness
    r_tf_in = 0.650 + (0.386 - thickness_tf) * np.cos(theta)
    z_tf_in = (0.386 - thickness_tf) * np.sin(theta)
    r_tf_out = 0.650 + (0.386 + thickness_tf) * np.cos(theta[::-1])
    z_tf_out = (0.386 + thickness_tf) * np.sin(theta[::-1])
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([r_tf_in, r_tf_out, [r_tf_in[0]]]),
        y=np.concatenate([z_tf_in, z_tf_out, [z_tf_in[0]]]),
        fill="toself",
        mode='lines',
        name='TF Coil',
        line=dict(color='red', width=1),
        fillcolor='red',
        hoverinfo='name'
    ))

    # --- 3. PF Coils (VF & OH) ---
    # Hardcoded from machine configuration
    vf_coils = [
        {'R': 1.0855, 'Z': 0.8245, 'dR': 0.052, 'dZ': 0.300},
        {'R': 1.0855, 'Z': -0.8245, 'dR': 0.052, 'dZ': 0.300}
    ]
    oh_coils = [
        {'R': 0.140, 'Z': 0.000, 'dR': 0.111, 'dZ': 0.740},
        {'R': 0.270, 'Z': 0.510, 'dR': 0.090, 'dZ': 0.225},
        {'R': 0.270, 'Z': -0.510, 'dR': 0.090, 'dZ': 0.225},
        {'R': 0.923, 'Z': 0.860, 'dR': 0.0501, 'dZ': 0.080},
        {'R': 0.923, 'Z': -0.860, 'dR': 0.0501, 'dZ': 0.080}
    ]

    def add_rect_traces(coils, name, fill_color, line_color):
        x_pts, y_pts = [], []
        for coil in coils:
             R, Z, dR, dZ = coil['R'], coil['Z'], coil['dR'], coil['dZ']
             # Box with None to separate shapes in one trace
             x_pts.extend([R-dR/2, R+dR/2, R+dR/2, R-dR/2, R-dR/2, None])
             y_pts.extend([Z-dZ/2, Z-dZ/2, Z+dZ/2, Z+dZ/2, Z-dZ/2, None])
             
        fig.add_trace(go.Scatter(
            x=x_pts, y=y_pts,
            fill="toself",
            mode='lines',
            name=name,
            line=dict(color=line_color, width=2),
            fillcolor=fill_color
        ))

    # OH Coils: Dark Red
    add_rect_traces(oh_coils, 'Ohmic Coils', 'darkred', 'black')
    # VF Coils: Brown+Yellow (Brown fill, Yellow line to look like copper/brass)
    add_rect_traces(vf_coils, 'VF Coils', 'saddlebrown', 'gold')

    # --- 3.1 Limiters ---
    
    def add_limiter(start_deg, end_deg, name, color):
        theta_lim = np.linspace(np.radians(start_deg), np.radians(end_deg), 50)
        # Inner arc
        r_in = 0.650 + 0.200 * np.cos(theta_lim)
        z_in = 0.200 * np.sin(theta_lim)
        # Outer arc (reverse to close loop)
        r_out = 0.650 + (0.200 + 0.050) * np.cos(theta_lim[::-1])
        z_out = (0.200 + 0.050) * np.sin(theta_lim[::-1])
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([r_in, r_out, [r_in[0]]]),
            y=np.concatenate([z_in, z_out, [z_in[0]]]),
            fill="toself",
            mode='lines',
            name=name,
            line=dict(color='gray', width=1),
            fillcolor=color,
            hoverinfo='name'
        ))

    add_limiter(45, 135, 'Movable Limiter (Top)', 'silver')
    add_limiter(225, 315, 'Movable Limiter (Bottom)', 'silver')
    add_limiter(315, 405, 'Fixed Limiter (LFS)', 'darkgray')

    # --- 3.2 Plasma ---
    theta_p = np.linspace(0, 2*np.pi, 100)
    r_p = 0.650 + 0.20 * np.cos(theta_p)
    z_p = 0.20 * np.sin(theta_p)
    
    fig.add_trace(go.Scatter(
        x=r_p, y=z_p,
        fill="toself",
        mode='lines',
        name='Plasma',
        line=dict(color='mediumpurple', width=1),
        fillcolor='rgba(216, 191, 216, 0.7)',
        hoverinfo='name'
    ))

    # --- 4. Diagnostics from signals.yaml ---
    gbp_x, gbp_y, gbp_text = [], [], []
    flux_x, flux_y, flux_text = [], [], []
    
    for key, data in signals.items():
        if isinstance(data, dict) and 'r' in data and 'z' in data:
            if key.startswith('GBP') and key.endswith('T'):
                gbp_x.append(data['r'])
                gbp_y.append(data['z'])
                gbp_text.append(key)
            elif key.startswith('F') and key[1:].isdigit(): # F0, F1...
                flux_x.append(data['r'])
                flux_y.append(data['z'])
                flux_text.append(key)

    fig.add_trace(go.Scatter(
        x=gbp_x, y=gbp_y,
        mode='markers',
        name='GBP Probes',
        marker=dict(size=8, color='black', symbol='circle-open', line=dict(width=2)),
        text=gbp_text,
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=flux_x, y=flux_y,
        mode='markers',
        name='Flux Loops',
        marker=dict(size=8, color='purple', symbol='diamond'),
        text=flux_text,
        hoverinfo='text'
    ))

    # --- 4.1 HCN Interferometry ---
    hcn_chords = [
        {'name': 'HCN1', 'r1': 0.610, 'z1': 0.400, 'r2': 0.610, 'z2': -0.400, 'wavelength': 337E-6},
        {'name': 'HCN2', 'r1': 0.650, 'z1': 0.400, 'r2': 0.650, 'z2': -0.400, 'wavelength': 337E-6},
        {'name': 'HCN3', 'r1': 0.690, 'z1': 0.400, 'r2': 0.690, 'z2': -0.400, 'wavelength': 337E-6},
    ]

    for hcn in hcn_chords:
        text_info = (f"{hcn['name']}<br>"
                     f"R = {hcn['r1']} m<br>"
                     f"Z = {hcn['z2']} to {hcn['z1']} m<br>"
                     f"Wavelength = {hcn['wavelength']*1e6:.0f} μm")
        fig.add_trace(go.Scatter(
            x=[hcn['r1'], hcn['r2']],
            y=[hcn['z1'], hcn['z2']],
            mode='lines',
            name=hcn['name'],
            line=dict(color='violet', width=2, dash='dashdot'),
            text=[text_info, text_info],
            hoverinfo='text'
        ))

    # Add reference line at R=0
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="TT-1 Poloidal Cross-Section",
        xaxis_title="Radial Position R (m)",
        yaxis_title="Vertical Position Z (m)",
        width=700, height=700,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        hovermode='closest',
        xaxis=dict(range=[0, 1.5]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig

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
    
    st.markdown("**Signals:** `LaBr3_1`, `LaBr3_2` (Unit: counts)")

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

    # --- 3. Magnetic Diagnostics ---
    st.header("3. Magnetic Diagnostics")
    st.write("The magnetic measurement system consists of various electromagnetic probes installed inside the vacuum vessel.")

    # Interactive Plot
    st.subheader("📊 Interactive Diagnostics Map")
    st.info("Click on the legend items to toggle visibility of components.")
    fig = get_cross_section_plot()
    if fig:
        st.plotly_chart(fig)

    # 3.1 Poloidal Magnetic Probes
    st.subheader("3.1 Poloidal Magnetic Probes (Mirnov Coils)")
    st.write("Measure local poloidal magnetic field perturbations ($B_\\theta$).")
    
    # Create lists for GBP probes
    gbp_list = ", ".join([f"`GBP{i}T`" for i in range(1, 13)])
    gbp_n_list = ", ".join([f"`GBP{i}N`" for i in range(1, 13)])
    
    st.markdown(f"**Tangential Probes ($B_t$):** {gbp_list}")
    st.markdown(f"**Normal Probes ($B_n$):** {gbp_n_list}")
    st.caption("**Unit:** Tesla (T)")

    # 3.2 Flux Loops & Voltage
    st.subheader("3.2 Flux Loops & Loop Voltage")
    st.write(r"Measure poloidal magnetic flux ($\Psi$) and loop voltage ($V_{loop}$).")
    flux_data = {
        "Signal Name": ["`F0`, `F1`, `F2`, `F3`, `F4`", "`VP0`, `VP1`, `VP2`, `VP3`"],
        "Unit": ["Wb", "V"],
        "Description": ["Poloidal Flux Loops", "Flux Loop Voltages"]
    }
    st.table(flux_data)

    # 3.3 Diamagnetic Loops
    st.subheader("3.3 Diamagnetic & Compensation Loops")
    st.write("Measure plasma stored energy (Diamagnetic Flux).")
    dia_data = {
        "Signal Name": ["`DIACD1A`, `DIAKL1B`", "`DIACD1C`, `DIAKL1C`"],
        "Unit": ["Wb", "Wb"],
        "Description": ["Diamagnetic Flux (Active)", "Compensation Flux (Background)"]
    }
    st.table(dia_data)

    st.divider()

    # --- 4. Coil & Plasma Currents ---
    st.header("4. Coil & Plasma Currents")
    st.write("Measurements of currents in the plasma and external magnetic coils.")
    curr_data = {
        "Signal Name": ["`IP1`, `IP2`", "`IOH1`, `IOH2`", "`IT1`, `IT2`", "`IV1`, `IV2`", "`IF1`, `IF2`, `HF_V`"],
        "Unit": ["A", "A", "A", "A", "A"],
        "Description": ["Plasma Current ($I_p$)", "Ohmic Heating Current", "Toroidal Field Current", "Vertical Field Current", "Feedback Control Current"]
    }
    st.table(curr_data)

    st.divider()

    # --- 5. Optical & Density Diagnostics ---
    st.header("5. Optical & Density Diagnostics")
    opt_data = {
        "Signal Name": ["`HA`", "`HCN1`, `HCN2`, `HCN3`", "`GP`"],
        "Unit": ["V", "V", "V"],
        "Description": [
            "H-Alpha Spectroscopy", 
            "HCN Interferometer (High/Mid/Low)", 
            "Gas Puffing Signal"
        ]
    }
    st.table(opt_data)

# Allow standalone execution for testing
if __name__ == "__main__":
    show_diagnostics_info()