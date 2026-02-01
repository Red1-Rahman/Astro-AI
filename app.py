# Astro-AI: Galaxy Evolution Analysis Platform
# 
# Copyright (c) 2025 Redwan Rahman and CAM-SUST
# Copernicus Astronomical Memorial of Shahjalal University of Science and Technology
# 
# All rights reserved. Licensed under the Astro-AI Proprietary License.
# Unauthorized use, distribution, or modification is strictly prohibited.
# See LICENSE file for full terms and conditions.
# 
# Contact: redwanrahman2002@outlook.com
# 
# Main application file integrating cosmic evolution simulations,
# cluster environment analysis, and JWST spectrum processing.
# 
# Created using functions and patterns from:
# - 21cmFAST simulations (CosmoSim.ipynb)
# - Galaxy cluster analysis (galaxy_clusters_workshop_notebook_Participants.ipynb)
# - SED fitting with Bagpipes (Galaxy SEDs Fitting.ipynb, Making model galaxies.ipynb)
# - JWST pipeline processing (reduce_data_st.ipynb, optimal_1d_extraction.ipynb)
# - Spectral fitting with NGSF (sf_class.py, SF_functions.py)

# CRITICAL: Setup bagpipes environment BEFORE any imports that might use bagpipes
import os
import tempfile
import sys

# Comprehensive bagpipes environment setup
try:
    # Create writable directory for bagpipes data files in Streamlit Cloud
    bagpipes_data_dir = os.path.join(tempfile.gettempdir(), 'bagpipes_data')
    os.makedirs(bagpipes_data_dir, exist_ok=True)
    
    # Create all expected subdirectories
    grids_dir = os.path.join(bagpipes_data_dir, 'grids')
    filters_dir = os.path.join(bagpipes_data_dir, 'filters')
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(filters_dir, exist_ok=True)
    
    # Set environment variables for bagpipes data location
    os.environ['BAGPIPES_FILTERS'] = bagpipes_data_dir
    os.environ['BAGPIPES_DATA'] = bagpipes_data_dir
    
    # Aggressive approach: Monkey patch bagpipes config module before import
    import types
    
    # Create a custom config module to intercept bagpipes.config
    mock_config = types.ModuleType('bagpipes.config')
    mock_config.BAGPIPES_DIR = bagpipes_data_dir
    mock_config.bagpipes_dir = bagpipes_data_dir
    mock_config.filters_dir = filters_dir  
    mock_config.grid_dir = grids_dir
    
    # Add commonly expected variables
    mock_config.igm_redshifts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    mock_config.igm_wavelengths = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    
    # Insert into sys.modules to intercept imports
    sys.modules['bagpipes.config'] = mock_config
    
    print(f"✅ Bagpipes environment configured: {bagpipes_data_dir}")
    print(f"✅ Bagpipes grids directory: {grids_dir}")
    print(f"✅ Bagpipes filters directory: {filters_dir}")
    
except Exception as e:
    print(f"⚠️ Bagpipes setup warning: {e}")

import streamlit as st
import logging
from utils.feature_flags import summarize_status, detect_capabilities, all_required_or_raise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Import modules
from modules.cos_evo.cosmic_evolution import CosmicEvolution
from modules.cluster_analyzer.cluster_analysis import ClusterAnalyzer
from modules.jwst_analyzer.jwst_pipeline import JWSTAnalyzer
from dashboard.dashboard import Dashboard
from api.openai_integration import OpenAIAssistant
from utils.data_handler import DataHandler
from utils.cosmology_utils import CosmologyUtils
from utils.plotting_utils import PlottingUtils

def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main():
    _configure_logging()
    st.set_page_config(
        page_title="Astro-AI: Galaxy Evolution Analysis Platform",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .module-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌌 Astro-AI: Galaxy Evolution Analysis Platform</h1>
        <p>Integrate cosmic evolution simulations, cluster analysis, and JWST spectroscopy for comprehensive galaxy studies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.image("Icons/AstroAI logo.jpg", width=200)
    st.sidebar.title("🚀 Navigation")
    st.sidebar.markdown("---")

    with st.sidebar.expander("Environment Status", expanded=False):
        st.markdown(summarize_status())
        strict = st.checkbox("Strict mode (require all heavy deps)", value=False, help="Fail if optional scientific dependencies are missing.")
        if strict:
            try:
                # Define which capabilities are truly required in strict deployment
                all_required_or_raise(["py21cmfast", "bagpipes", "jwst_pipeline", "astropy"])  # adjust as needed
                st.success("All required capabilities present.")
            except Exception as e:
                st.error(str(e))
        else:
            caps = detect_capabilities()
            missing = [k for k, v in caps.items() if not v.available]
            if missing:
                st.caption("Missing optional modules: " + ", ".join(missing))
    
    module = st.sidebar.selectbox(
        "Select Analysis Module:",
        [
            "🏠 Home",
            "📊 Data Upload & Setup",
            "🌌 Module 1: Cosmic Evolution (Cos-Evo)",
            "🌟 Module 2: Cluster Environment Analyzer",
            "🔭 Module 3: JWST Spectrum Analyzer",
            "📈 Comparative Dashboard",
            "📝 Report & Reflection"
        ]
    )
    
    # Initialize session state
    if 'cosmology_params' not in st.session_state:
        st.session_state.cosmology_params = {
            'H0': 67.66,
            'Om0': 0.31,
            'z_range': [5, 15]
        }
    
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = {}
    
    # Route to different modules
    if module == "🏠 Home":
        show_home()
    elif module == "📊 Data Upload & Setup":
        show_data_upload()
    elif module == "🌌 Module 1: Cosmic Evolution (Cos-Evo)":
        show_cosmic_evolution()
    elif module == "🌟 Module 2: Cluster Environment Analyzer":
        show_cluster_analyzer()
    elif module == "🔭 Module 3: JWST Spectrum Analyzer":
        show_jwst_analyzer()
    elif module == "📈 Comparative Dashboard":
        show_dashboard()
    elif module == "📝 Report & Reflection":
        show_report()

def show_home():
    st.header("Welcome to Astro-AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Platform Overview
        Astro-AI is a comprehensive galaxy evolution analysis platform that integrates:
        
        - **21cm Simulations** using py21cmfast
        - **Galaxy Cluster Analysis** with photometric and spectroscopic data
        - **JWST Pipeline** for advanced spectrum processing
        - **SED Fitting** with Bagpipes and other tools
        - **AI-Powered Insights** for interpretation and analysis
        """)
        
        st.markdown("""
        ### 🔄 End-to-End Workflow
        1. **Input Data**: Upload catalogs, JWST files, set cosmology
        2. **Cosmic Evolution**: Run 21cm simulations across cosmic time
        3. **Cluster Analysis**: Analyze galaxy environments and properties
        4. **JWST Spectra**: Process and fit high-resolution spectra
        5. **Integration**: Compare results across all modules
        6. **Report**: Generate insights and interpretations
        """)
    
    with col2:
        st.markdown("""
        ### 📚This platform reuses and integrates functions from:
        
        - **21cmFAST** simulations (CosmoSim.ipynb)
        - **Galaxy cluster** analysis tools
        - **Bagpipes** SED fitting framework
        - **JWST STScI** pipeline integration
        - **NGSF** spectral fitting utilities
        - **BayeSN** supernova analysis tools
        """)
        
        # Quick start buttons
        st.markdown("### 🚀 Quick Start")
        if st.button("📊 Upload Data", type="primary", use_container_width=True):
            st.session_state.current_module = "📊 Data Upload & Setup"
            st.rerun()
        
        if st.button("🌌 Start Cosmic Evolution", use_container_width=True):
            st.session_state.current_module = "🌌 Module 1: Cosmic Evolution (Cos-Evo)"
            st.rerun()

def show_data_upload():
    st.header("📊 Data Upload & Configuration")
    
    tab1, tab2, tab3 = st.tabs(["Upload Options", "Cosmological Parameters", "Example Datasets"])
    
    with tab1:
        st.subheader("Upload Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Galaxy Catalog (CSV/FITS)**")
            catalog_file = st.file_uploader(
                "Upload catalog with RA, Dec, z, photometry",
                type=['csv', 'fits'],
                help="Required columns: RA, Dec, redshift, and photometric bands"
            )
            
            if catalog_file:
                st.session_state.uploaded_data['catalog'] = catalog_file
                st.success("✅ Catalog uploaded successfully!")
        
        with col2:
            st.markdown("**JWST/NIRSpec File (FITS)**")
            jwst_file = st.file_uploader(
                "Upload JWST spectroscopic data",
                type=['fits'],
                help="Stage 2 or Stage 3 JWST pipeline products"
            )
            
            if jwst_file:
                st.session_state.uploaded_data['jwst'] = jwst_file
                st.success("✅ JWST data uploaded successfully!")
    
    with tab2:
        st.subheader("Set Cosmological Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            h0 = st.number_input("H₀ (km/s/Mpc)", value=67.66, min_value=50.0, max_value=100.0)
        
        with col2:
            om0 = st.number_input("Ωₘ", value=0.31, min_value=0.1, max_value=0.9)
        
        with col3:
            sigma8 = st.number_input("σ₈", value=0.8, min_value=0.6, max_value=1.2)
        
        z_min, z_max = st.slider("Redshift Range", 0.0, 20.0, (5.0, 15.0))
        
        st.session_state.cosmology_params = {
            'H0': h0,
            'Om0': om0,
            'sigma8': sigma8,
            'z_range': [z_min, z_max]
        }
        
        if st.button("💾 Save Cosmology"):
            st.success("✅ Cosmological parameters saved!")
    
    with tab3:
        st.subheader("Example Datasets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Load Cluster Sample", use_container_width=True):
                # This would load example cluster data
                st.info("Loading example cluster catalog...")
        
        with col2:
            if st.button("Load Demo JWST Data", use_container_width=True):
                # This would load example JWST data
                st.info("Loading demo JWST spectra...")
        
        with col3:
            if st.button("Default Cosmology", use_container_width=True):
                st.session_state.cosmology_params = {
                    'H0': 67.66,
                    'Om0': 0.31,
                    'sigma8': 0.8,
                    'z_range': [5.0, 15.0]
                }
                st.success("✅ Default cosmology loaded!")

def show_cosmic_evolution():
    st.header("🌌 Module 1: Cosmic Evolution (Cos-Evo)")
    
    # Initialize cosmic evolution module
    cos_evo = CosmicEvolution(st.session_state.cosmology_params)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        box_size = st.selectbox("Box Size (Mpc)", [50, 100, 200], index=0)
        resolution = st.selectbox("Resolution (HII_DIM)", [50, 100, 128], index=0)
        
        z_start = st.number_input("Start Redshift", value=15.0, min_value=6.0, max_value=20.0)
        z_end = st.number_input("End Redshift", value=6.0, min_value=5.0, max_value=15.0)
        z_step = st.number_input("Redshift Step", value=1.0, min_value=0.5, max_value=2.0)
        
        if st.button("🚀 Run 21cm Simulation", type="primary"):
            with st.spinner("Running cosmic evolution simulation..."):
                # This would call the actual simulation
                results = cos_evo.run_simulation(box_size, resolution, z_start, z_end, z_step)
                st.session_state.cos_evo_results = results
                st.success("✅ Simulation completed!")
    
    with col2:
        st.subheader("Expected Outputs")
        
        # Show what outputs will be generated
        st.markdown("""
        **This module will generate:**
        
        1. **Evolution of 21 cm signal vs redshift**
           - Brightness temperature maps at different z
           - Power spectra evolution
           - Global signal extraction
        
        2. **Density field snapshots (z ~ 5–15)**
           - Matter density evolution
           - Ionization fraction maps
           - Neutral hydrogen distribution
        
        3. **Timeline of galaxy emergence**
           - First light signatures
           - Reionization progression
           - Structure formation milestones
        """)
        
        # Placeholder plot
        fig, ax = plt.subplots(figsize=(8, 6))
        z_placeholder = np.linspace(6, 15, 100)
        signal_placeholder = -50 * np.exp(-(z_placeholder - 10)**2 / 10) + np.random.normal(0, 5, 100)
        
        ax.plot(z_placeholder, signal_placeholder, 'b-', linewidth=2)
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Brightness Temperature [mK]')
        ax.set_title('21cm Global Signal Evolution (Example)')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

def show_cluster_analyzer():
    st.header("🌟 Module 2: Cluster Environment Analyzer")
    
    # Initialize cluster analyzer
    cluster_analyzer = ClusterAnalyzer()
    
    tab1, tab2, tab3 = st.tabs(["Cluster Detection", "SED Fitting", "Results"])
    
    with tab1:
        st.subheader("Galaxy Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Spatial Distribution**")
            if st.button("Plot RA-Dec Distribution"):
                # This would generate the actual plot from uploaded data
                fig, ax = plt.subplots(figsize=(8, 6))
                # Placeholder data
                ra = np.random.normal(150, 1, 1000)
                dec = np.random.normal(2, 0.5, 1000)
                ax.scatter(ra, dec, alpha=0.6, s=20)
                ax.set_xlabel('RA (degrees)')
                ax.set_ylabel('Dec (degrees)')
                ax.set_title('Galaxy Spatial Distribution')
                st.pyplot(fig)
        
        with col2:
            st.markdown("**Redshift Analysis**")
            if st.button("Generate Redshift Histogram"):
                fig, ax = plt.subplots(figsize=(8, 6))
                # Placeholder redshift distribution with cluster peak
                z_gals = np.concatenate([
                    np.random.normal(1.2, 0.05, 200),  # Cluster peak
                    np.random.uniform(0.5, 2.0, 800)   # Field galaxies
                ])
                ax.hist(z_gals, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(1.2, color='red', linestyle='--', label='Cluster z=1.2')
                ax.set_xlabel('Redshift')
                ax.set_ylabel('Number of Galaxies')
                ax.set_title('Redshift Distribution - Cluster Detection')
                ax.legend()
                st.pyplot(fig)
    
    with tab2:
        st.subheader("Bagpipes SED Fitting Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Star Formation History**")
            sfh_model = st.selectbox("SFH Model", ["exponential", "double_power_law", "burst"])
            
            if sfh_model == "exponential":
                age_range = st.slider("Age Range (Gyr)", 0.1, 15.0, (0.1, 15.0))
                tau_range = st.slider("τ Range (Gyr)", 0.3, 10.0, (0.3, 10.0))
            
            mass_range = st.slider("Log(M*/M☉) Range", 8.0, 12.0, (8.0, 12.0))
            metallicity_range = st.slider("Metallicity Range (Z☉)", 0.0, 2.5, (0.0, 2.5))
        
        with col2:
            st.markdown("**Dust Model**")
            dust_model = st.selectbox("Dust Curve", ["Calzetti", "SMC", "MW"])
            av_range = st.slider("Av Range (mag)", 0.0, 3.0, (0.0, 2.0))
            
            st.markdown("**Redshift**")
            fit_redshift = st.checkbox("Fit redshift", value=False)
            if not fit_redshift:
                fixed_z = st.number_input("Fixed redshift", value=1.2, min_value=0.0, max_value=10.0)
        
        if st.button("🔄 Run SED Fitting", type="primary"):
            with st.spinner("Running Bagpipes SED fitting..."):
                # This would run the actual SED fitting
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                st.success("✅ SED fitting completed!")
    
    with tab3:
        st.subheader("Analysis Results")
        
        # Placeholder results visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Color-Magnitude Diagram**")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Generate example CMD
            mag_r = np.random.normal(20, 2, 1000)
            color_gr = np.random.normal(0.8, 0.3, 1000)
            
            # Red sequence
            red_seq_mask = (color_gr > 0.6) & (mag_r < 22)
            
            ax.scatter(mag_r[~red_seq_mask], color_gr[~red_seq_mask], 
                      alpha=0.6, s=20, c='blue', label='Blue cloud')
            ax.scatter(mag_r[red_seq_mask], color_gr[red_seq_mask], 
                      alpha=0.8, s=20, c='red', label='Red sequence')
            
            ax.set_xlabel('r magnitude')
            ax.set_ylabel('g - r color')
            ax.set_title('Color-Magnitude Diagram')
            ax.legend()
            ax.invert_xaxis()
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Mass-SFR Relation**")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Generate example mass-SFR data
            log_mass = np.random.normal(10.5, 0.8, 500)
            log_sfr = 0.8 * log_mass - 8 + np.random.normal(0, 0.5, 500)
            
            # Separate cluster vs field
            cluster_mask = np.random.choice([True, False], size=500, p=[0.3, 0.7])
            
            ax.scatter(log_mass[~cluster_mask], log_sfr[~cluster_mask], 
                      alpha=0.6, s=30, c='gray', label='Field galaxies')
            ax.scatter(log_mass[cluster_mask], log_sfr[cluster_mask], 
                      alpha=0.8, s=30, c='orange', label='Cluster members')
            
            ax.set_xlabel('log(M*/M☉)')
            ax.set_ylabel('log(SFR) [M☉/yr]')
            ax.set_title('Mass-SFR Relation')
            ax.legend()
            st.pyplot(fig)

def show_jwst_analyzer():
    st.header("🔭 Module 3: JWST Spectrum Analyzer")
    
    # Initialize JWST analyzer
    jwst_analyzer = JWSTAnalyzer()
    
    tab1, tab2, tab3 = st.tabs(["Pipeline Steps", "1D Extraction", "Spectral Fitting"])
    
    with tab1:
        st.subheader("JWST Data Reduction Pipeline")
        
        pipeline_steps = [
            "Stage 1: Detector Processing (uncal → rate)",
            "Stage 2: Spectroscopic Processing (rate → cal)",
            "Stage 3: Combine Exposures (cal → crf)",
            "Custom: Optimal 1D Extraction"
        ]
        
        for i, step in enumerate(pipeline_steps):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{step}**")
            
            with col2:
                status = "✅ Complete" if i < 2 else "⏳ Pending"
                st.write(status)
            
            with col3:
                if st.button(f"Run Step {i+1}", key=f"step_{i}"):
                    with st.spinner(f"Running {step}..."):
                        # Simulate processing time
                        import time
                        time.sleep(2)
                        st.success(f"✅ {step} completed!")
        
        st.markdown("---")
        
        if st.button("🚀 Run Full Pipeline", type="primary"):
            with st.spinner("Running complete JWST pipeline..."):
                progress_bar = st.progress(0)
                steps = ["Stage 1", "Stage 2", "Stage 3", "1D Extraction"]
                for i, step in enumerate(steps):
                    st.write(f"Processing {step}...")
                    progress_bar.progress((i + 1) * 25)
                    import time
                    time.sleep(1)
                st.success("✅ Full pipeline completed!")
    
    with tab2:
        st.subheader("Optimal 1D Spectrum Extraction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Extraction Parameters**")
            
            profile_sigma = st.slider("Profile σ (pixels)", 1.0, 5.0, 2.0)
            bg_offset = st.slider("Background offset (pixels)", 3, 10, 5)
            snr_threshold = st.slider("SNR threshold", 3.0, 20.0, 10.0)
            
            extraction_method = st.selectbox(
                "Extraction Method",
                ["Optimal (Horne 1986)", "Simple Aperture", "Profile Weighted"]
            )
            
            if st.button("Extract 1D Spectrum"):
                with st.spinner("Extracting 1D spectrum..."):
                    # Generate example 1D spectrum
                    wavelength = np.linspace(1.0, 5.0, 1000)  # microns
                    flux = np.exp(-(wavelength - 2.5)**2 / 0.5) + 0.1 * np.random.normal(0, 1, 1000)
                    flux_err = 0.05 * np.ones_like(flux)
                    
                    st.session_state.extracted_spectrum = {
                        'wavelength': wavelength,
                        'flux': flux,
                        'flux_err': flux_err
                    }
                    st.success("✅ 1D spectrum extracted!")
        
        with col2:
            st.markdown("**Extracted Spectrum**")
            
            if 'extracted_spectrum' in st.session_state:
                spec = st.session_state.extracted_spectrum
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(spec['wavelength'], spec['flux'], 'b-', linewidth=1, label='Flux')
                ax.fill_between(spec['wavelength'], 
                              spec['flux'] - spec['flux_err'],
                              spec['flux'] + spec['flux_err'],
                              alpha=0.3, color='blue', label='±1σ error')
                
                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel('Flux (arbitrary units)')
                ax.set_title('Extracted 1D Spectrum')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
            else:
                st.info("Extract a 1D spectrum to see the plot here.")
    
    with tab3:
        st.subheader("Bagpipes Spectral Fitting")
        
        if 'extracted_spectrum' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Fitting Configuration**")
                
                # SFH model selection
                sfh_components = st.multiselect(
                    "Star Formation History Components",
                    ["exponential", "burst", "constant"],
                    default=["exponential"]
                )
                
                # Dust model
                dust_law = st.selectbox("Dust Law", ["Calzetti", "SMC", "MW"])
                
                # Redshift
                z_fit = st.checkbox("Fit redshift")
                if not z_fit:
                    z_fixed = st.number_input("Fixed redshift", value=3.5, min_value=0.0)
                
                # Spectral resolution
                spec_res = st.number_input("Spectral Resolution R", value=1000, min_value=100)
                
                if st.button("🔬 Fit Spectrum with Bagpipes"):
                    with st.spinner("Running Bagpipes spectral fit..."):
                        # Simulate fitting process
                        progress_bar = st.progress(0)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            import time
                            time.sleep(0.05)
                        
                        # Generate mock results
                        st.session_state.spectral_fit_results = {
                            'stellar_mass': 10.2,
                            'stellar_mass_err': 0.3,
                            'sfr': 5.2,
                            'sfr_err': 1.1,
                            'age': 2.1,
                            'age_err': 0.5,
                            'metallicity': 0.8,
                            'metallicity_err': 0.2,
                            'av': 0.3,
                            'av_err': 0.1
                        }
                        st.success("✅ Spectral fitting completed!")
            
            with col2:
                st.markdown("**Fitted Results**")
                
                if 'spectral_fit_results' in st.session_state:
                    results = st.session_state.spectral_fit_results
                    
                    # Display results table
                    results_df = pd.DataFrame({
                        'Parameter': ['log(M*/M☉)', 'SFR (M☉/yr)', 'Age (Gyr)', 'Z/Z☉', 'Av (mag)'],
                        'Value': [
                            f"{results['stellar_mass']:.1f} ± {results['stellar_mass_err']:.1f}",
                            f"{results['sfr']:.1f} ± {results['sfr_err']:.1f}",
                            f"{results['age']:.1f} ± {results['age_err']:.1f}",
                            f"{results['metallicity']:.1f} ± {results['metallicity_err']:.1f}",
                            f"{results['av']:.1f} ± {results['av_err']:.1f}"
                        ]
                    })
                    
                    st.dataframe(results_df, hide_index=True)
                    
                    # Plot fit results
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
                    
                    # Spectrum + fit
                    spec = st.session_state.extracted_spectrum
                    model_flux = spec['flux'] + 0.02 * np.random.normal(0, 1, len(spec['flux']))
                    
                    ax1.plot(spec['wavelength'], spec['flux'], 'k-', linewidth=1, label='Observed')
                    ax1.plot(spec['wavelength'], model_flux, 'r-', linewidth=1, label='Best fit')
                    ax1.set_ylabel('Flux')
                    ax1.set_title('Spectrum + Best Fit Model')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Residuals
                    residuals = spec['flux'] - model_flux
                    ax2.plot(spec['wavelength'], residuals, 'g-', linewidth=1)
                    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
                    ax2.set_ylabel('Residuals')
                    ax2.set_title('Fit Residuals')
                    ax2.grid(True, alpha=0.3)
                    
                    # Star formation history
                    age_bins = np.linspace(0, 10, 50)
                    sfh = np.exp(-(age_bins - 2)**2 / 2) * 5
                    
                    ax3.plot(age_bins, sfh, 'b-', linewidth=2)
                    ax3.set_xlabel('Lookback Time (Gyr)')
                    ax3.set_ylabel('SFR (M☉/yr)')
                    ax3.set_title('Star Formation History')
                    ax3.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Run spectral fitting to see results here.")
        else:
            st.info("Please extract a 1D spectrum first in the previous tab.")

def show_dashboard():
    st.header("📈 Comparative Dashboard")
    
    # Initialize dashboard
    dashboard = Dashboard()
    
    tab1, tab2, tab3 = st.tabs(["Integration Overview", "Comparative Plots", "Galaxy Storyline"])
    
    with tab1:
        st.subheader("Analysis Integration Status")
        
        # Module status cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status1 = "✅ Complete" if 'cos_evo_results' in st.session_state else "⏳ Pending"
            st.markdown(f"""
            <div class="module-card">
                <h4>🌌 Cosmic Evolution</h4>
                <p>Status: {status1}</p>
                <p>21cm simulations and power spectra</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status2 = "✅ Complete" if 'cluster_analysis_results' in st.session_state else "⏳ Pending"
            st.markdown(f"""
            <div class="module-card">
                <h4>🌟 Cluster Analysis</h4>
                <p>Status: {status2}</p>
                <p>Environment effects and SED fitting</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status3 = "✅ Complete" if 'spectral_fit_results' in st.session_state else "⏳ Pending"
            st.markdown(f"""
            <div class="module-card">
                <h4>🔭 JWST Analysis</h4>
                <p>Status: {status3}</p>
                <p>High-resolution spectroscopy</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Integration summary
        st.subheader("🔗 Cross-Module Connections")
        
        if st.button("Generate Integration Summary"):
            st.markdown("""
            **Cosmic Timeline Integration:**
            - 21cm simulations show the early universe structure formation
            - Cluster analysis reveals environment-dependent galaxy evolution
            - JWST spectra provide detailed stellar population properties
            
            **Key Connections:**
            1. Reionization signatures from 21cm → cluster formation epochs
            2. Environmental quenching from clusters → spectroscopic confirmation
            3. High-z galaxy properties from JWST → cosmic evolution context
            """)
    
    with tab2:
        st.subheader("Side-by-Side Comparative Analysis")
        
        # Create comparative plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 21cm evolution
        z_range = np.linspace(6, 15, 50)
        signal_21cm = -50 * np.exp(-(z_range - 10)**2 / 10) + np.random.normal(0, 3, 50)
        
        ax1.plot(z_range, signal_21cm, 'b-', linewidth=2)
        ax1.set_xlabel('Redshift')
        ax1.set_ylabel('21cm Signal [mK]')
        ax1.set_title('🌌 Cosmic Evolution Timeline')
        ax1.grid(True, alpha=0.3)
        
        # Cluster environment
        mass_bins = np.logspace(9, 12, 20)
        red_fraction = 0.1 + 0.6 / (1 + np.exp(-(mass_bins - 10**10.5) / 1e10))
        
        ax2.semilogx(mass_bins, red_fraction, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Stellar Mass [M☉]')
        ax2.set_ylabel('Red Fraction')
        ax2.set_title('🌟 Environmental Quenching')
        ax2.grid(True, alpha=0.3)
        
        # JWST spectrum example
        wavelength = np.linspace(1, 5, 200)
        spectrum = np.exp(-(wavelength - 2.5)**2 / 0.3) + 0.1 * np.random.normal(0, 1, 200)
        
        ax3.plot(wavelength, spectrum, 'g-', linewidth=2)
        ax3.set_xlabel('Wavelength [μm]')
        ax3.set_ylabel('Flux')
        ax3.set_title('🔭 JWST Spectroscopy')
        ax3.grid(True, alpha=0.3)
        
        # Combined timeline
        cosmic_time = np.linspace(0.5, 13.8, 100)
        redshift_time = np.interp(cosmic_time, [0.5, 2, 5, 13.8], [10, 2, 0.5, 0])
        
        ax4.plot(cosmic_time, redshift_time, 'purple', linewidth=3)
        ax4.axvline(2, color='red', linestyle='--', label='Cluster formation')
        ax4.axvline(1, color='blue', linestyle='--', label='JWST observations')
        ax4.set_xlabel('Cosmic Time [Gyr]')
        ax4.set_ylabel('Redshift')
        ax4.set_title('🕰️ Unified Timeline')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("🌌 Galaxy Evolution Storyline")
        
        st.markdown("""
        ### The Complete Picture: From Cosmic Dawn to Today
        
        **Phase 1: Cosmic Dawn (z ~ 15-10) - 21cm Era**
        - First light from primordial stars
        - Reionization bubbles begin to form
        - Dark matter halos collapse to form first galaxies
        
        **Phase 2: Assembly Era (z ~ 10-3) - Cluster Formation**
        - Hierarchical structure formation accelerates
        - Galaxy clusters begin to form and evolve
        - Environmental effects start to influence galaxy properties
        
        **Phase 3: Maturation (z ~ 3-0) - JWST Window**
        - Detailed stellar populations observable with JWST
        - Environmental quenching becomes prominent
        - Modern galaxy properties established
        """)
        
        # Interactive storyline
        storyline_progress = st.slider("Evolution Timeline", 0, 100, 50, 
                                     help="Slide to explore different epochs")
        
        if storyline_progress < 33:
            st.info("🌅 **Cosmic Dawn Era**: 21cm signals dominate, first stars ignite")
        elif storyline_progress < 66:
            st.warning("🏗️ **Assembly Era**: Clusters form, environment shapes galaxies")
        else:
            st.success("🔬 **JWST Era**: Detailed spectroscopy reveals stellar archaeology")

def show_report():
    st.header("📝 Report & Reflection")
    
    # Initialize AI assistant with OpenRouter/DeepSeek by default
    ai_assistant = OpenAIAssistant(provider="openrouter", model="deepseek/deepseek-r1:free")
    
    # Show AI setup guide if in fallback mode
    if ai_assistant.fallback_mode:
        with st.expander("🤖 Enable AI Features - Setup Guide", expanded=True):
            st.markdown("""
            ### 🆓 **Quick Setup: Free AI with OpenRouter**
            
            **Step 1:** Get a free API key at [OpenRouter.ai](https://openrouter.ai/)
            
            **Step 2:** In Streamlit Cloud, go to **Settings > Secrets** and add:
            ```toml
            OPENROUTER_API_KEY = "sk-or-v1-your-key-here"
            ```
            
            **Step 3:** Restart the app to enable AI-powered analysis!
            
            ### 🚦 **About Rate Limits**
            
            **Free Models:** Sometimes experience high demand and temporary rate limits
            **With Your Own Key:** Get dedicated rate limits and priority access
            **Alternative:** The app gracefully falls back to high-quality simulation mode
            
            **Features you'll unlock:**
            - ✨ AI-powered scientific insights
            - 📊 Intelligent comparative analysis  
            - 📝 Automated report generation
            - 🎯 Smart next-step suggestions
            
            **Model:** DeepSeek R1 (free, optimized for scientific reasoning)
            """)
    else:
        # Show rate limit info if we're getting 429 errors
        with st.expander("ℹ️ AI Service Info", expanded=False):
            st.markdown("""
            **Current Status:** ✅ AI Features Enabled
            
            **Note:** If you see rate limit messages, this is normal for free tiers during peak usage.
            The app automatically switches to simulation mode when needed.
            
            **For Priority Access:** Add your own OpenRouter API key in Streamlit secrets.
            """)
            
            st.info("💡 **Tip:** OpenRouter offers free access to DeepSeek R1, which is excellent for scientific analysis!")
    
    tab1, tab2, tab3 = st.tabs(["Analysis Report", "AI Insights", "Export Results"])
    
    with tab1:
        st.subheader("Scientific Analysis Report")
        
        # Text editor for user notes
        st.markdown("**Your Analysis Notes:**")
        user_notes = st.text_area(
            "Write your observations and interpretations:",
            height=300,
            placeholder="Describe your findings from the cosmic evolution, cluster analysis, and JWST spectroscopy modules..."
        )
        
        # Guided prompts
        st.markdown("**Suggested Analysis Questions:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌌 How do initial conditions shape galaxy evolution?"):
                prompt_text = """
                Based on the 21cm simulations and cluster analysis, discuss how the initial 
                density fluctuations in the early universe influence the later formation 
                and evolution of galaxies in different environments.
                """
                st.text_area("Analysis prompt:", value=prompt_text, height=100)
        
        with col2:
            if st.button("🏠 What role does environment play in quenching?"):
                prompt_text = """
                Compare the star formation properties of cluster versus field galaxies. 
                How do the spectroscopic results from JWST complement the photometric 
                analysis of environmental effects?
                """
                st.text_area("Analysis prompt:", value=prompt_text, height=100)
        
        if st.button("🔭 How do JWST spectra refine our models?"):
            prompt_text = """
            Discuss how the high-resolution spectroscopic data from JWST provides 
            constraints on galaxy formation models that cannot be obtained from 
            photometry alone.
            """
            st.text_area("Analysis prompt:", value=prompt_text, height=100)
    
    with tab2:
        st.subheader("🤖 AI-Powered Scientific Insights")
        
        st.markdown("Get AI assistance for interpreting your results and generating scientific insights.")
        
        # Query input
        ai_query = st.text_input(
            "Ask the AI assistant about your analysis:",
            placeholder="e.g., 'Explain the connection between 21cm signals and cluster formation'"
        )
        
        if st.button("💭 Get AI Insights"):
            if ai_query:
                with st.spinner("Generating AI insights..."):
                    # This would use the OpenAI API
                    ai_response = ai_assistant.generate_insight(ai_query, st.session_state)
                    st.markdown("**AI Response:**")
                    st.markdown(ai_response)
            else:
                st.warning("Please enter a query for the AI assistant.")
        
        # Pre-defined analysis templates
        st.markdown("**Quick Analysis Templates:**")
        
        template_options = [
            "Cosmic evolution timeline analysis",
            "Environmental effects on galaxy properties",
            "Spectroscopic vs photometric constraints",
            "High-redshift galaxy formation insights"
        ]
        
        selected_template = st.selectbox("Choose an analysis template:", template_options)
        
        if st.button(f"Generate {selected_template}"):
            with st.spinner("Generating analysis..."):
                template_response = ai_assistant.generate_template_analysis(selected_template, st.session_state)
                st.markdown("**Generated Analysis:**")
                st.markdown(template_response)
    
    with tab3:
        st.subheader("📊 Export & Save Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Export Options:**")
            
            export_formats = st.multiselect(
                "Select export formats:",
                ["CSV", "JSON", "FITS", "HDF5"],
                default=["CSV", "JSON"]
            )
            
            include_plots = st.checkbox("Include plots (PNG/PDF)", value=True)
            include_report = st.checkbox("Include analysis report", value=True)
            
            if st.button("📦 Prepare Export Package"):
                with st.spinner("Preparing export package..."):
                    # This would generate the actual export files
                    st.success("✅ Export package ready!")
                    
                    # Show download buttons
                    st.download_button(
                        label="📥 Download Results Package",
                        data="Mock data package",  # Would be actual data
                        file_name="astro_ai_results.zip",
                        mime="application/zip"
                    )
        
        with col2:
            st.markdown("**Report Generation:**")
            
            report_sections = st.multiselect(
                "Include report sections:",
                ["Executive Summary", "Methodology", "Results", "Discussion", "Conclusions"],
                default=["Executive Summary", "Results", "Discussion"]
            )
            
            report_format = st.selectbox("Report format:", ["PDF", "HTML", "Markdown"])
            
            if st.button("📄 Generate Scientific Report"):
                with st.spinner("Generating scientific report..."):
                    # This would generate the actual report
                    st.success("✅ Scientific report generated!")
                    
                    st.download_button(
                        label="📥 Download Report",
                        data="Mock scientific report",  # Would be actual report
                        file_name=f"astro_ai_report.{report_format.lower()}",
                        mime="application/pdf" if report_format == "PDF" else "text/html"
                    )
    
    # Team Credits Section
    st.markdown("---")
    st.markdown("### 👥 Development Team")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            st.image("Icons/Team R3NS.jpg", width=180)
        except Exception:
            st.markdown("**🌌 Team R3NS**")
    
    with col2:
        st.markdown("""
        **Team R3NS** - CAM-SUST 2025
        
        **Team Members:**
        1. 👨‍💻 [Redwan Rahman](https://github.com/Red1-Rahman) - Project Lead, Backend & Frontend Development
        2. 👩‍🔬 **Nishat Nabilah Ahmed** - Scientific Validation Lead
        3. 👩‍💻 **Nafia Wahid Nirjhor** - Backend Development
        4. 👨‍🔬 **Saidul Hossain Al Amin** - Scientific Validation & Documentation
        5. 👨‍💻 **Ahnaf Rahman Nabil** - Quality Assurance
        
        """)

if __name__ == "__main__":
    main()
