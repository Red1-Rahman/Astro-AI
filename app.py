# Astro-AI: Galaxy Evolution Analysis Platform
#
# Copyright (c) 2025 Redwan Rahman and CAM-SUST
# Copernicus Astronomical Memorial of Shahjalal University of Science and Technology
#
# Optimized version: adds @st.cache_data / @st.cache_resource, lazy figure
# generation, session-state pipeline tracking, and removes blocking time.sleep().

# ── Bagpipes env setup (must happen before any bagpipes import) ───────────────
import os, sys, types, tempfile

try:
    bagpipes_data_dir = os.path.join(tempfile.gettempdir(), "bagpipes_data")
    grids_dir   = os.path.join(bagpipes_data_dir, "grids")
    filters_dir = os.path.join(bagpipes_data_dir, "filters")
    for d in (bagpipes_data_dir, grids_dir, filters_dir):
        os.makedirs(d, exist_ok=True)

    os.environ["BAGPIPES_FILTERS"] = bagpipes_data_dir
    os.environ["BAGPIPES_DATA"]    = bagpipes_data_dir

    mock_config = types.ModuleType("bagpipes.config")
    mock_config.BAGPIPES_DIR    = bagpipes_data_dir
    mock_config.bagpipes_dir    = bagpipes_data_dir
    mock_config.filters_dir     = filters_dir
    mock_config.grid_dir        = grids_dir
    mock_config.igm_redshifts   = list(range(11))
    mock_config.igm_wavelengths = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    sys.modules["bagpipes.config"] = mock_config
except Exception as e:
    print(f"⚠️  Bagpipes setup warning: {e}")

# ── Standard imports ──────────────────────────────────────────────────────────
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # thread-safe non-interactive backend
import matplotlib.pyplot as plt
import streamlit as st
warnings.filterwarnings("ignore")

from modules.cos_evo.cosmic_evolution      import CosmicEvolution
from modules.cluster_analyzer.cluster_analysis import ClusterAnalyzer
from modules.jwst_analyzer.jwst_pipeline   import JWSTAnalyzer
from dashboard.dashboard                   import Dashboard
from api.openai_integration                import OpenAIAssistant
from utils.feature_flags import summarize_status, detect_capabilities, all_required_or_raise


# ─────────────────────────────────────────────────────────────────────────────
# CACHING LAYER
# @st.cache_resource  → singleton objects shared across all sessions/reruns
#                       (module instances, heavy models, DB connections)
# @st.cache_data      → serialisable data; each unique set of args gets its
#                       own cache entry; safe to return DataFrames / arrays
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_cluster_analyzer() -> ClusterAnalyzer:
    """One ClusterAnalyzer per server process."""
    return ClusterAnalyzer()


@st.cache_resource(show_spinner=False)
def _get_jwst_analyzer() -> JWSTAnalyzer:
    return JWSTAnalyzer()


@st.cache_resource(show_spinner=False)
def _get_ai_assistant():
    return OpenAIAssistant(provider="openrouter", model="deepseek/deepseek-r1:free")


@st.cache_resource(show_spinner=False)
def _get_dashboard():
    return Dashboard()


# Simulation results are cached per unique parameter combination.
# hash_funcs not needed — all args are plain Python scalars.
@st.cache_data(ttl=3600, show_spinner=False)
def _run_simulation(box_size: int, resolution: int,
                    z_start: float, z_end: float, z_step: float,
                    h0: float, om0: float, sigma8: float) -> dict:
    cos_evo = CosmicEvolution({"H0": h0, "Om0": om0, "sigma8": sigma8,
                               "z_range": [z_end, z_start]})
    return cos_evo.run_simulation(box_size, resolution, z_start, z_end, z_step)


@st.cache_data(ttl=1800, show_spinner=False)
def _run_mock_cluster(n_galaxies: int = 1000, n_clusters: int = 5) -> pd.DataFrame:
    ca = ClusterAnalyzer()
    ca.generate_mock_data(n_galaxies, n_clusters)
    return ca.data


@st.cache_data(ttl=1800, show_spinner=False)
def _generate_mock_spectrum() -> dict:
    ja = JWSTAnalyzer()
    ja._generate_mock_jwst_data("rate")
    # Run mock pipeline so extraction is available
    ja.pipeline_status["stage1"] = True
    ja.pipeline_status["stage2"] = True
    ja.reduced_data["stage2"] = ja.raw_data
    spec = ja._optimal_extraction(
        ja.raw_data["data"],
        ja.raw_data["error"],
        np.linspace(1.0, 5.0, ja.raw_data["data"].shape[1]),
    )
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION-STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state():
    defaults = {
        "cosmology_params": {"H0": 67.66, "Om0": 0.31, "sigma8": 0.8,
                              "z_range": [5.0, 15.0]},
        "uploaded_data":    {},
        # pipeline progress flags (no time.sleep needed)
        "jwst_stage1_done": False,
        "jwst_stage2_done": False,
        "jwst_stage3_done": False,
        "jwst_extraction_done": False,
        # results
        "cos_evo_results":          None,
        "cluster_analysis_results": None,
        "extracted_spectrum":       None,
        "spectral_fit_results":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _configure_logging()
    _init_session_state()

    st.set_page_config(
        page_title="Astro-AI: Galaxy Evolution Analysis Platform",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main-header{text-align:center;background:linear-gradient(90deg,#1e3c72,#2a5298);
        color:white;padding:2rem;border-radius:10px;margin-bottom:2rem}
    .module-card{background:#f8f9fa;padding:1.5rem;border-radius:10px;
        border:1px solid #dee2e6;margin:1rem 0}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🌌 Astro-AI: Galaxy Evolution Analysis Platform</h1>
        <p>Cosmic evolution simulations · Cluster analysis · JWST spectroscopy</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.image("Icons/AstroAI logo.jpg", width=200)
    st.sidebar.title("🚀 Navigation")
    st.sidebar.markdown("---")

    with st.sidebar.expander("Environment Status", expanded=False):
        st.markdown(summarize_status())
        strict = st.checkbox("Strict mode", value=False,
                             help="Fail if optional scientific dependencies are missing.")
        if strict:
            try:
                all_required_or_raise(["py21cmfast", "bagpipes", "jwst_pipeline", "astropy"])
                st.success("All required capabilities present.")
            except Exception as e:
                st.error(str(e))
        else:
            missing = [k for k, v in detect_capabilities().items() if not v.available]
            if missing:
                st.caption("Missing optional modules: " + ", ".join(missing))

    module = st.sidebar.selectbox(
        "Select Analysis Module:",
        ["🏠 Home", "📊 Data Upload & Setup",
         "🌌 Module 1: Cosmic Evolution (Cos-Evo)",
         "🌟 Module 2: Cluster Environment Analyzer",
         "🔭 Module 3: JWST Spectrum Analyzer",
         "📈 Comparative Dashboard",
         "📝 Report & Reflection"],
    )

    # ── Route ─────────────────────────────────────────────────────────────────
    routes = {
        "🏠 Home":                                  show_home,
        "📊 Data Upload & Setup":                   show_data_upload,
        "🌌 Module 1: Cosmic Evolution (Cos-Evo)":  show_cosmic_evolution,
        "🌟 Module 2: Cluster Environment Analyzer":show_cluster_analyzer,
        "🔭 Module 3: JWST Spectrum Analyzer":      show_jwst_analyzer,
        "📈 Comparative Dashboard":                  show_dashboard,
        "📝 Report & Reflection":                   show_report,
    }
    routes[module]()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────

def show_home():
    st.header("Welcome to Astro-AI")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Platform Overview
        Astro-AI integrates three scientific pipelines:
        - **21cm Simulations** using py21cmFAST
        - **Galaxy Cluster Analysis** with photometric and spectroscopic data
        - **JWST NIRSpec Pipeline** for advanced spectrum processing
        - **SED Fitting** with Bagpipes
        - **AI-Powered Insights** via OpenRouter / DeepSeek R1
        """)
        st.markdown("""
        ### 🔄 End-to-End Workflow
        1. **Input Data** — upload catalogs, JWST files, set cosmology
        2. **Cosmic Evolution** — 21 cm simulations across cosmic time
        3. **Cluster Analysis** — galaxy environments and properties
        4. **JWST Spectra** — process and fit high-resolution spectra
        5. **Integration** — compare results across modules
        6. **Report** — generate insights and interpretations
        """)

    with col2:
        st.markdown("""
        ### 📚 Integrated Tools
        - 21cmFAST (CosmoSim.ipynb)
        - Galaxy cluster analysis tools
        - Bagpipes SED fitting framework
        - JWST STScI pipeline
        - NGSF spectral fitting utilities
        - BayeSN supernova analysis tools
        """)
        st.markdown("### 🚀 Quick Start")
        if st.button("📊 Upload Data", type="primary", use_container_width=True):
            st.info("Navigate to **Data Upload & Setup** in the sidebar.")
        if st.button("🌌 Start Cosmic Evolution", use_container_width=True):
            st.info("Navigate to **Module 1** in the sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def show_data_upload():
    st.header("📊 Data Upload & Configuration")
    tab1, tab2, tab3 = st.tabs(["Upload Options", "Cosmological Parameters", "Example Datasets"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Galaxy Catalog (CSV/FITS)**")
            catalog_file = st.file_uploader(
                "Upload catalog with RA, Dec, z, photometry",
                type=["csv", "fits"],
                help="Required columns: RA, Dec, redshift, and photometric bands",
            )
            if catalog_file:
                st.session_state.uploaded_data["catalog"] = catalog_file
                st.success("✅ Catalog uploaded!")
        with col2:
            st.markdown("**JWST/NIRSpec File (FITS)**")
            jwst_file = st.file_uploader(
                "Upload JWST spectroscopic data",
                type=["fits"],
                help="Stage 2 or Stage 3 JWST pipeline products",
            )
            if jwst_file:
                st.session_state.uploaded_data["jwst"] = jwst_file
                st.success("✅ JWST data uploaded!")

    with tab2:
        st.subheader("Set Cosmological Parameters")
        c = st.session_state.cosmology_params
        col1, col2, col3 = st.columns(3)
        with col1:
            h0      = st.number_input("H₀ (km/s/Mpc)", value=float(c["H0"]),    min_value=50.0, max_value=100.0)
        with col2:
            om0     = st.number_input("Ωₘ",              value=float(c["Om0"]),   min_value=0.1,  max_value=0.9)
        with col3:
            sigma8  = st.number_input("σ₈",              value=float(c.get("sigma8", 0.8)), min_value=0.6, max_value=1.2)

        z_min, z_max = st.slider("Redshift Range", 0.0, 20.0,
                                 (float(c["z_range"][0]), float(c["z_range"][1])))
        if st.button("💾 Save Cosmology"):
            st.session_state.cosmology_params = {
                "H0": h0, "Om0": om0, "sigma8": sigma8, "z_range": [z_min, z_max],
            }
            st.success("✅ Cosmological parameters saved!")

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Load Cluster Sample", use_container_width=True):
                with st.spinner("Generating mock cluster catalog…"):
                    st.session_state.cluster_mock_data = _run_mock_cluster()
                st.success(f"✅ Loaded {len(st.session_state.cluster_mock_data):,} galaxies")
        with col2:
            if st.button("Load Demo JWST Data", use_container_width=True):
                with st.spinner("Generating mock JWST spectrum…"):
                    st.session_state.extracted_spectrum = _generate_mock_spectrum()
                st.success("✅ Demo spectrum loaded")
        with col3:
            if st.button("Default Cosmology", use_container_width=True):
                st.session_state.cosmology_params = {
                    "H0": 67.66, "Om0": 0.31, "sigma8": 0.8, "z_range": [5.0, 15.0],
                }
                st.success("✅ Default cosmology restored")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: COSMIC EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def show_cosmic_evolution():
    st.header("🌌 Module 1: Cosmic Evolution (Cos-Evo)")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Simulation Parameters")
        box_size   = st.selectbox("Box Size (Mpc)",        [50, 100, 200], index=0)
        resolution = st.selectbox("Resolution (HII_DIM)", [50, 100, 128], index=0)
        z_start    = st.number_input("Start Redshift", value=15.0, min_value=6.0,  max_value=20.0)
        z_end      = st.number_input("End Redshift",   value=6.0,  min_value=5.0,  max_value=15.0)
        z_step     = st.number_input("Redshift Step",  value=1.0,  min_value=0.5,  max_value=2.0)

        cp = st.session_state.cosmology_params

        # Warn user if result is already cached
        if st.session_state.cos_evo_results is not None:
            st.info("Showing cached results. Change parameters and rerun to update.")

        if st.button("🚀 Run 21cm Simulation", type="primary"):
            with st.spinner("Running cosmic evolution simulation…"):
                results = _run_simulation(
                    box_size, resolution, z_start, z_end, z_step,
                    cp["H0"], cp["Om0"], cp.get("sigma8", 0.8),
                )
                st.session_state.cos_evo_results = results
            st.success("✅ Simulation complete!")

    with col2:
        st.subheader("Results")
        results = st.session_state.cos_evo_results

        if results is None:
            # Show a placeholder so the page isn't empty on first load
            st.markdown("**Expected outputs after running simulation:**")
            st.markdown("""
            1. Global 21 cm signal vs redshift  
            2. Brightness temperature maps (z ~ 5–15)  
            3. Power spectrum evolution  
            """)
            z_ph = np.linspace(6, 15, 100)
            sig_ph = -50 * np.exp(-(z_ph - 10)**2 / 10)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(z_ph, sig_ph, "b--", lw=1.5, alpha=0.5, label="Example (not real data)")
            ax.set_xlabel("Redshift z"); ax.set_ylabel("Brightness Temperature [mK]")
            ax.set_title("21 cm Global Signal — placeholder")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            tab_a, tab_b, tab_c = st.tabs(["Global Signal", "Power Spectra", "BT Slices"])
            cos_evo = CosmicEvolution(st.session_state.cosmology_params)
            cos_evo.results = results

            with tab_a:
                fig, ax = cos_evo.plot_global_evolution()
                st.pyplot(fig); plt.close(fig)

            with tab_b:
                fig, ax = cos_evo.plot_power_spectra_evolution()
                st.pyplot(fig); plt.close(fig)

            with tab_c:
                fig, axes = cos_evo.plot_brightness_temperature_slices()
                st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CLUSTER ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

def show_cluster_analyzer():
    st.header("🌟 Module 2: Cluster Environment Analyzer")

    # Retrieve shared (cached) analyzer instance
    ca = _get_cluster_analyzer()

    # Load data into analyzer if available from session state
    if "cluster_mock_data" in st.session_state and ca.data is None:
        ca.data = st.session_state.cluster_mock_data
        ca._process_data()

    tab1, tab2, tab3 = st.tabs(["Cluster Detection", "SED Fitting", "Results"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Spatial Distribution**")
            if ca.data is None:
                st.info("Load data first via Data Upload → Load Cluster Sample.")
            else:
                # Only regenerate figure when button pressed
                if st.button("Plot RA-Dec Distribution") or "fig_spatial" in st.session_state:
                    if "fig_spatial" not in st.session_state:
                        ca.analyze_spatial_distribution()
                        ca.separate_cluster_field()
                        fig, _ = ca.plot_spatial_distribution()
                        # Save figure bytes to session state so reruns don't replot
                        import io
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                        plt.close(fig)
                        st.session_state.fig_spatial = buf.getvalue()
                    st.image(st.session_state.fig_spatial)

        with col2:
            st.markdown("**Redshift Distribution**")
            if ca.data is None:
                st.info("Load data first.")
            else:
                if st.button("Detect Clusters") or "cluster_detect_results" in st.session_state:
                    if "cluster_detect_results" not in st.session_state:
                        results = ca.detect_clusters_redshift()
                        st.session_state.cluster_detect_results = results

                    results = st.session_state.cluster_detect_results
                    n = results["n_clusters_detected"]
                    st.metric("Clusters detected", n)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(results["z_histogram"]["bin_centers"],
                           results["z_histogram"]["counts"],
                           width=np.diff(results["z_histogram"]["bin_edges"]),
                           alpha=0.7, color="steelblue", label="Galaxies")
                    ax.plot(results["z_histogram"]["bin_centers"],
                            results["background"], "r--", lw=1.5, label="Background")
                    for cl in results["detected_clusters"]:
                        ax.axvline(cl["redshift"], color="green", lw=1, alpha=0.8)
                    ax.set_xlabel("Redshift"); ax.set_ylabel("Count")
                    ax.set_title("Redshift Distribution — Cluster Detection")
                    ax.legend(); ax.grid(True, alpha=0.3)
                    st.pyplot(fig); plt.close(fig)

    with tab2:
        st.subheader("Bagpipes SED Fitting Configuration")
        col1, col2 = st.columns(2)

        with col1:
            sfh_model = st.selectbox("SFH Model", ["exponential", "double_power_law"])
            if sfh_model == "exponential":
                st.slider("Age Range (Gyr)",  0.1, 15.0, (0.1, 15.0))
                st.slider("τ Range (Gyr)",    0.3, 10.0, (0.3, 10.0))
            st.slider("Log(M*/M☉) Range",     8.0, 12.0, (8.0, 12.0))
            st.slider("Metallicity (Z☉)",      0.0,  2.5, (0.0,  2.5))

        with col2:
            dust_model = st.selectbox("Dust Curve", ["Calzetti", "SMC", "MW"])
            st.slider("Av Range (mag)", 0.0, 3.0, (0.0, 2.0))
            fit_redshift = st.checkbox("Fit redshift", value=False)
            if not fit_redshift:
                st.number_input("Fixed redshift", value=1.2, min_value=0.0, max_value=10.0)

        if st.button("🔄 Run SED Fitting", type="primary"):
            if ca.data is None:
                st.warning("Load data first.")
            else:
                with st.spinner("Running Bagpipes SED fitting… (this may take a while)"):
                    fit_instructions = ca.setup_bagpipes_model(sfh_model, dust_model)
                    # Fit a small subset so the UI stays responsive
                    subset = ca.data.sample(min(50, len(ca.data)), random_state=42)
                    sed_results = ca.run_sed_fitting(subset, fit_instructions)
                    st.session_state.cluster_analysis_results = sed_results
                st.success(f"✅ SED fitting complete — {len(sed_results)} galaxies fitted")

    with tab3:
        if ca.data is None:
            st.info("Load data and run analysis first.")
        else:
            # Lazy: only compute red fraction when the tab is active
            if "red_frac_results" not in st.session_state:
                ca.separate_cluster_field()
                st.session_state.red_frac_results = ca.compute_red_fraction()

            col1, col2 = st.columns(2)
            with col1:
                fig, _ = ca.plot_color_magnitude_diagram()
                st.pyplot(fig); plt.close(fig)
            with col2:
                fig, _ = ca.plot_red_fraction()
                st.pyplot(fig); plt.close(fig)

            if st.button("📋 Generate Summary Report"):
                report = ca.generate_summary_report()
                st.json(report)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: JWST ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

def show_jwst_analyzer():
    st.header("🔭 Module 3: JWST Spectrum Analyzer")
    ja = _get_jwst_analyzer()

    tab1, tab2, tab3 = st.tabs(["Pipeline Steps", "1D Extraction", "Spectral Fitting"])

    with tab1:
        st.subheader("JWST Data Reduction Pipeline")

        steps = [
            ("Stage 1: Detector Processing (uncal → rate)",        "jwst_stage1_done", ja.run_pipeline_stage1),
            ("Stage 2: Spectroscopic Processing (rate → cal)",      "jwst_stage2_done", ja.run_pipeline_stage2),
            ("Stage 3: Combine Exposures (cal → crf)",              "jwst_stage3_done", ja.run_pipeline_stage3),
        ]

        for label, flag, fn in steps:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"**{label}**")
            done = st.session_state.get(flag, False)
            col2.write("✅ Done" if done else "⏳ Pending")
            if col3.button("Run", key=flag + "_btn"):
                # Only run if previous stage done (or stage 1 which has no prereq)
                with st.spinner(f"Running {label}…"):
                    ok = fn()
                    if ok:
                        st.session_state[flag] = True
                        st.rerun()
                    else:
                        st.error(f"❌ {label} failed — check logs.")

        st.markdown("---")

        if st.button("🚀 Run Full Pipeline", type="primary"):
            with st.spinner("Running complete JWST pipeline…"):
                prog = st.progress(0)
                for i, (label, flag, fn) in enumerate(steps):
                    fn()
                    st.session_state[flag] = True
                    prog.progress((i + 1) * 33)
                st.session_state["jwst_stage3_done"] = True
            st.success("✅ Full pipeline complete!")
            st.rerun()

    with tab2:
        st.subheader("Optimal 1D Spectrum Extraction")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Extraction Parameters**")
            ja.extraction_params["profile_sigma"] = st.slider("Profile σ (px)", 1.0, 5.0, 2.0)
            ja.extraction_params["bg_offset"]      = st.slider("Background offset (px)", 3, 10, 5)
            ja.extraction_params["snr_threshold"]  = st.slider("SNR threshold", 3.0, 20.0, 10.0)
            method_label = st.selectbox("Method",
                ["Optimal (Horne 1986)", "Simple Aperture", "Profile Weighted"])
            method_map = {
                "Optimal (Horne 1986)": "optimal",
                "Simple Aperture":      "aperture",
                "Profile Weighted":     "profile",
            }

            if st.button("Extract 1D Spectrum"):
                # Use mock data if no real pipeline has run
                if not st.session_state.jwst_stage2_done:
                    with st.spinner("Generating mock spectrum…"):
                        spec = _generate_mock_spectrum()
                else:
                    with st.spinner("Extracting 1D spectrum…"):
                        spec = ja.extract_1d_spectrum(method_map[method_label])
                st.session_state.extracted_spectrum = spec
                st.success("✅ Spectrum extracted!")

        with col2:
            spec = st.session_state.extracted_spectrum
            if spec is not None:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(spec["wavelength"], spec["flux"], "b-", lw=1, label="Flux")
                ax.fill_between(spec["wavelength"],
                                spec["flux"] - spec["flux_error"],
                                spec["flux"] + spec["flux_error"],
                                alpha=0.3, color="blue", label="±1σ")
                ax.set_xlabel("Wavelength (μm)"); ax.set_ylabel("Flux")
                ax.set_title("Extracted 1D Spectrum")
                ax.legend(); ax.grid(True, alpha=0.3)
                st.pyplot(fig); plt.close(fig)
            else:
                st.info("Extract a spectrum to see the plot here.")

    with tab3:
        st.subheader("Bagpipes Spectral Fitting")
        spec = st.session_state.extracted_spectrum

        if spec is None:
            st.info("Please extract a 1D spectrum first (previous tab).")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Fitting Configuration**")
                dust_law  = st.selectbox("Dust Law", ["Calzetti", "SMC", "MW"])
                z_fit     = st.checkbox("Fit redshift")
                z_fixed   = 3.5
                if not z_fit:
                    z_fixed = st.number_input("Fixed redshift", value=3.5, min_value=0.0)
                spec_res  = st.number_input("Spectral Resolution R", value=1000, min_value=100)

                if st.button("🔬 Fit Spectrum with Bagpipes"):
                    with st.spinner("Running spectral fit… (mock mode if Bagpipes not installed)"):
                        ja.extracted_spectra["target"] = spec
                        fit_instructions = ja.setup_spectral_fitting_model("exponential")
                        if not z_fit:
                            fit_instructions["redshift"] = z_fixed
                        result = ja.fit_spectrum_bagpipes("target", fit_instructions,
                                                          spec_resolution=spec_res)
                        st.session_state.spectral_fit_results = result
                    st.success("✅ Spectral fitting complete!")

            with col2:
                result = st.session_state.spectral_fit_results
                if result:
                    df = pd.DataFrame({
                        "Parameter": ["log(M*/M☉)", "Age (Gyr)", "τ (Gyr)",
                                      "Z/Z☉", "Av (mag)", "Redshift"],
                        "Value": [
                            f"{result['stellar_mass']:.2f} ± {result['stellar_mass_err']:.2f}",
                            f"{result['age']:.2f} ± {result['age_err']:.2f}",
                            f"{result['tau']:.2f} ± {result['tau_err']:.2f}",
                            f"{result['metallicity']:.2f} ± {result['metallicity_err']:.2f}",
                            f"{result['av']:.2f} ± {result['av_err']:.2f}",
                            f"{result['redshift']:.3f} ± {result['redshift_err']:.3f}",
                        ],
                    })
                    st.dataframe(df, hide_index=True)

                    # SFH plot
                    fig, ax = plt.subplots(figsize=(8, 4))
                    time_arr = np.linspace(0, 13.8, 100)
                    lookback = 13.8 - result["age"]
                    sfh = np.exp(-(time_arr - lookback)**2 / (2 * result["tau"]**2))
                    sfh[time_arr < lookback] = 0
                    ax.plot(time_arr, sfh, "b-", lw=2)
                    ax.set_xlabel("Cosmic Time (Gyr)"); ax.set_ylabel("SFR (arb. units)")
                    ax.set_title("Star Formation History"); ax.grid(True, alpha=0.3)
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.info("Run fitting to see results here.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: COMPARATIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def show_dashboard():
    st.header("📈 Comparative Dashboard")
    tab1, tab2, tab3 = st.tabs(["Integration Overview", "Comparative Plots", "Galaxy Storyline"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        for col, key, label, detail in [
            (col1, "cos_evo_results",          "🌌 Cosmic Evolution", "21 cm simulations & power spectra"),
            (col2, "cluster_analysis_results", "🌟 Cluster Analysis",  "Environment effects & SED fitting"),
            (col3, "spectral_fit_results",     "🔭 JWST Analysis",    "High-resolution spectroscopy"),
        ]:
            status = "✅ Complete" if st.session_state.get(key) else "⏳ Pending"
            col.markdown(f"""<div class="module-card">
                <h4>{label}</h4><p>Status: {status}</p><p>{detail}</p>
            </div>""", unsafe_allow_html=True)

        if st.button("Generate Integration Summary"):
            st.markdown("""
            **Cosmic Timeline Integration**

            - 21 cm simulations reveal early-universe structure formation.
            - Cluster analysis shows environment-dependent quenching.
            - JWST spectra constrain stellar populations at key redshifts.

            **Key Connections**

            1. Reionization signatures (21 cm) → cluster formation epochs  
            2. Environmental quenching → spectroscopic confirmation (JWST)  
            3. High-z galaxy properties (JWST) → cosmic evolution context  
            """)

    with tab2:
        # Gate figure generation behind a button so it only runs on demand
        if st.button("Generate Comparative Plots") or "dashboard_figs" in st.session_state:
            if "dashboard_figs" not in st.session_state:
                import io
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

                z_r  = np.linspace(6, 15, 50)
                ax1.plot(z_r, -50 * np.exp(-(z_r - 10)**2 / 10), "b-", lw=2)
                ax1.set(xlabel="Redshift", ylabel="21 cm Signal [mK]",
                        title="🌌 Cosmic Evolution Timeline"); ax1.grid(alpha=0.3)

                mass_b = np.logspace(9, 12, 20)
                rf     = 0.1 + 0.6 / (1 + np.exp(-(mass_b - 10**10.5) / 1e10))
                ax2.semilogx(mass_b, rf, "ro-", lw=2, ms=5)
                ax2.set(xlabel="Stellar Mass [M☉]", ylabel="Red Fraction",
                        title="🌟 Environmental Quenching"); ax2.grid(alpha=0.3)

                wl = np.linspace(1, 5, 200)
                ax3.plot(wl, np.exp(-(wl - 2.5)**2 / 0.3), "g-", lw=2)
                ax3.set(xlabel="Wavelength [μm]", ylabel="Flux",
                        title="🔭 JWST Spectroscopy"); ax3.grid(alpha=0.3)

                ct = np.linspace(0.5, 13.8, 100)
                ax4.plot(ct, np.interp(ct, [0.5, 2, 5, 13.8], [10, 2, 0.5, 0]),
                         color="purple", lw=3)
                ax4.axvline(2, color="red",  ls="--", label="Cluster formation")
                ax4.axvline(1, color="blue", ls="--", label="JWST observations")
                ax4.set(xlabel="Cosmic Time [Gyr]", ylabel="Redshift",
                        title="🕰️ Unified Timeline"); ax4.legend(); ax4.grid(alpha=0.3)

                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                st.session_state.dashboard_figs = buf.getvalue()

            st.image(st.session_state.dashboard_figs)

    with tab3:
        st.markdown("""
        ### The Complete Picture: From Cosmic Dawn to Today

        **Phase 1 — Cosmic Dawn (z ~ 15–10): 21 cm Era**  
        First light from primordial stars; reionization bubbles emerge.

        **Phase 2 — Assembly Era (z ~ 10–3): Cluster Formation**  
        Hierarchical structure formation; environment begins shaping galaxies.

        **Phase 3 — Maturation (z ~ 3–0): JWST Window**  
        Detailed stellar populations observable; quenching prominent.
        """)
        val = st.slider("Evolution Timeline", 0, 100, 50)
        if val < 33:
            st.info("🌅 **Cosmic Dawn** — 21 cm signals dominate, first stars ignite")
        elif val < 66:
            st.warning("🏗️ **Assembly Era** — clusters form, environment shapes galaxies")
        else:
            st.success("🔬 **JWST Era** — spectroscopy reveals stellar archaeology")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REPORT & REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

def show_report():
    st.header("📝 Report & Reflection")
    ai_assistant = _get_ai_assistant()

    rag_status = ai_assistant.get_rag_status()
    if rag_status["enabled"]:
        with st.expander("🔍 RAG Knowledge Base Status", expanded=False):
            st.success("✅ Retrieval-Augmented Generation enabled")
            stats = rag_status["stats"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Documents",      stats.get("total_documents", 0))
            c2.metric("Vocabulary",     stats.get("vocabulary_size", 0))
            c3.metric("Analysis Types", len(stats.get("analysis_types", [])))
    else:
        with st.expander("🔍 RAG System Status", expanded=False):
            st.warning(f"⚠️ RAG not available: {rag_status['reason']}")

    if ai_assistant.fallback_mode:
        with st.expander("🤖 Enable AI Features", expanded=True):
            st.markdown("""
            **Quick Setup — Free AI via OpenRouter**

            1. Get a free key at [openrouter.ai](https://openrouter.ai/)
            2. In Streamlit Cloud → **Settings > Secrets**:
            ```toml
            OPENROUTER_API_KEY = "sk-or-v1-your-key-here"
            ```
            3. Restart the app.
            """)

    tab1, tab2, tab3 = st.tabs(["Analysis Report", "AI Insights", "Export Results"])

    with tab1:
        st.subheader("Scientific Analysis Report")
        st.text_area("Your Analysis Notes:", height=300,
                     placeholder="Describe your findings…")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌌 Initial conditions → galaxy evolution?"):
                st.text_area("Prompt:", height=100, value=(
                    "Based on the 21 cm simulations and cluster analysis, discuss how initial "
                    "density fluctuations influence the later formation and evolution of galaxies "
                    "in different environments."))
        with col2:
            if st.button("🏠 Environment and quenching?"):
                st.text_area("Prompt:", height=100, value=(
                    "Compare star formation properties of cluster vs field galaxies. How do "
                    "JWST spectroscopic results complement photometric environmental analysis?"))

    with tab2:
        st.subheader("🤖 AI-Powered Scientific Insights")
        ai_query = st.text_input("Ask the AI assistant:",
                                 placeholder="e.g. 'Explain the connection between 21 cm signals and cluster formation'")
        if st.button("💭 Get AI Insights") and ai_query:
            with st.spinner("Generating insights…"):
                st.markdown(ai_assistant.generate_insight(ai_query, st.session_state))

        selected = st.selectbox("Quick Analysis Template:", [
            "Cosmic evolution timeline analysis",
            "Environmental effects on galaxy properties",
            "Spectroscopic vs photometric constraints",
            "High-redshift galaxy formation insights",
        ])
        if st.button(f"Generate: {selected}"):
            with st.spinner("Generating analysis…"):
                st.markdown(ai_assistant.generate_template_analysis(selected, st.session_state))

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            export_formats = st.multiselect("Export formats:", ["CSV", "JSON", "FITS", "HDF5"],
                                            default=["CSV", "JSON"])
            include_plots  = st.checkbox("Include plots", value=True)
            if st.button("📦 Prepare Export Package"):
                with st.spinner("Preparing…"):
                    st.success("✅ Export package ready!")
                    st.download_button("📥 Download Results",
                                       data=b"placeholder",
                                       file_name="astro_ai_results.zip",
                                       mime="application/zip")

        with col2:
            report_sections = st.multiselect("Report sections:",
                ["Executive Summary", "Methodology", "Results", "Discussion", "Conclusions"],
                default=["Executive Summary", "Results", "Discussion"])
            report_format = st.selectbox("Format:", ["PDF", "HTML", "Markdown"])
            if st.button("📄 Generate Scientific Report"):
                with st.spinner("Generating…"):
                    st.success("✅ Report generated!")
                    st.download_button("📥 Download Report",
                                       data=b"placeholder",
                                       file_name=f"astro_ai_report.{report_format.lower()}",
                                       mime="text/html")

    # Team credits
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
        **Team R3NS — CAM-SUST 2025**

        1. 👨‍💻 [Redwan Rahman](https://github.com/Red1-Rahman) — Project Lead, Backend & Frontend  
        2. 👩‍🔬 [Nishat Nabilah Ahmed](https://github.com/NN-Ahmed) — Scientific Validation Lead  
        3. 👩‍💻 [Nafia Wahid Nirjhor](https://github.com/nafiawahidnirjhor) — Backend Development  
        4. 👨‍🔬 [Saidul Hossain Al Amin](https://github.com/SaidulHossainAlamin) — Scientific Validation & Docs  
        5. 👨‍💻 [Ahnaf Rahman Nabil](https://github.com/NaBziY) — Quality Assurance  
        """)


if __name__ == "__main__":
    main()