# Cluster Environment Analyzer Module for Astro-AI
#
# Copyright (c) 2025 Redwan Rahman and CAM-SUST
#
# Mock galaxy properties grounded in published observations:
#   - Stellar mass function (SMF):
#       Baldry et al. 2012 (MNRAS 421, 621)       — field SMF
#       Vulcani et al. 2013 (A&A 550, A58)        — cluster SMF enhancement
#   - Red fraction vs mass and environment:
#       Peng et al. 2010 (ApJ 721, 193)           — mass & environment quenching
#   - Star-forming main sequence (SFR–M* relation):
#       Speagle et al. 2014 (ApJS 214, 15)        — redshift-dependent SFMS
#   - Color bimodality:
#       Bell et al. 2004 (ApJ 608, 752)           — red/blue separation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import warnings
import logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import os, tempfile
    bagpipes_data_dir = os.path.join(tempfile.gettempdir(), "bagpipes_data")
    os.makedirs(bagpipes_data_dir, exist_ok=True)
    os.environ.setdefault("BAGPIPES_FILTERS", bagpipes_data_dir)
    os.environ.setdefault("BAGPIPES_DATA",    bagpipes_data_dir)
    import bagpipes as pipes
    HAVE_BAGPIPES = True
    logger.info("Bagpipes loaded successfully.")
except (ImportError, PermissionError, OSError) as e:
    HAVE_BAGPIPES = False
    logger.warning("Bagpipes not available: %s. Using simulation mode.", e)

try:
    from astropy.io import fits
    from astropy.table import Table
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False


# ---------------------------------------------------------------------------
# Physical constants / reference parameters
# ---------------------------------------------------------------------------

# Baldry+2012 Table 1 — field double-Schechter parameters (all galaxies)
_SCHECHTER_FIELD = dict(
    log_M_star = 10.66,   # characteristic mass log(M*/M☉)
    phi1       = 3.96e-3, # Mpc⁻³ dex⁻¹  (faint-end component)
    alpha1     = -0.35,
    phi2       = 0.79e-3,
    alpha2     = -1.47,
)

# Vulcani+2013 — cluster SMF has enhanced massive-galaxy fraction
# Implemented as a tilt of the characteristic mass by +0.15 dex
_SCHECHTER_CLUSTER = dict(
    log_M_star = 10.81,
    phi1       = 4.50e-3,
    alpha1     = -0.20,
    phi2       = 0.50e-3,
    alpha2     = -1.30,
)

# Peng+2010 eq. 12 — mass-quenching efficiency εm
# f_red(M*, env) = 1 - (1 - f_q_mass)(1 - f_q_env)
_PENG_MASS_QUENCH_SLOPE  = 1.0        # log-linear slope vs log(M*)
_PENG_MASS_QUENCH_PIVOT  = 10.5       # log(M*/M☉) where εm ~ 0.5
_PENG_ENV_DELTA_RED_FRAC = 0.15       # additional red fraction in clusters (Peng+2010 §4)

# Speagle+2014 eq. 28 — SFMS at cosmic noon z~1 (t~5.9 Gyr)
# log(SFR) = (0.84 - 0.026·t)·log(M*) - (6.51 - 0.11·t)
_SFMS_T_REF = 5.9    # Gyr (z~1, typical cluster study epoch)

# g-r color cut for red/blue separation (Bell+2004)
_COLOR_CUT = 0.65


# ---------------------------------------------------------------------------
# Schechter / Peng utility functions
# ---------------------------------------------------------------------------

def _double_schechter_logm(log_m: np.ndarray, params: dict) -> np.ndarray:
    """
    Double Schechter function in log-mass units (Baldry+2012 eq. 4).

    Returns φ [Mpc⁻³ dex⁻¹] at each log_m.
    """
    dm    = log_m - params["log_M_star"]
    term1 = params["phi1"] * np.log(10) * 10**(dm * (1 + params["alpha1"])) * np.exp(-10**dm)
    term2 = params["phi2"] * np.log(10) * 10**(dm * (1 + params["alpha2"])) * np.exp(-10**dm)
    return term1 + term2


def _peng_red_fraction(log_m: float, in_cluster: bool = False) -> float:
    """
    Red fraction as a function of stellar mass and environment.

    Based on Peng et al. 2010 mass-quenching + environment-quenching model.
    Returns f_red ∈ [0, 1].
    """
    # Mass quenching: logistic centred at pivot mass
    eps_mass = 1.0 / (1.0 + np.exp(-1.5 * (log_m - _PENG_MASS_QUENCH_PIVOT)))
    # Combined: f_red = 1 - (1 - eps_mass)(1 - eps_env)
    eps_env  = _PENG_ENV_DELTA_RED_FRAC if in_cluster else 0.0
    f_red    = 1.0 - (1.0 - eps_mass) * (1.0 - eps_env)
    return float(np.clip(f_red, 0.0, 1.0))


def _speagle_sfr(log_m: float, t_gyr: float = _SFMS_T_REF,
                 scatter_dex: float = 0.3) -> float:
    """
    Star-forming main sequence SFR [M☉/yr] from Speagle+2014 eq. 28.

    Parameters
    ----------
    log_m       : log10(M*/M☉)
    t_gyr       : cosmic time [Gyr]  (use age of universe at target z)
    scatter_dex : intrinsic scatter of the SFMS [dex]
    """
    log_sfr_mean = ((0.84 - 0.026 * t_gyr) * log_m
                    - (6.51 - 0.11 * t_gyr))
    # Add realistic lognormal scatter
    log_sfr = log_sfr_mean + np.random.normal(0.0, scatter_dex)
    return float(10 ** log_sfr)


def _sample_log_mass(n: int, params: dict,
                     log_m_min: float = 9.0,
                     log_m_max: float = 12.0) -> np.ndarray:
    """
    Draw stellar masses from a double-Schechter function via rejection sampling.
    """
    log_m_grid = np.linspace(log_m_min, log_m_max, 500)
    phi        = _double_schechter_logm(log_m_grid, params)
    phi_max    = phi.max()

    samples = []
    rng     = np.random.default_rng(seed=42)
    while len(samples) < n:
        batch   = rng.uniform(log_m_min, log_m_max, n * 5)
        accept  = rng.uniform(0, phi_max, n * 5)
        phi_val = _double_schechter_logm(batch, params)
        samples.extend(batch[accept < phi_val])

    return np.array(samples[:n])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClusterAnalyzer:
    """
    Galaxy Cluster Environment Analysis.

    Mock data are generated from published empirical relations so that
    results are scientifically meaningful even without real observations.
    All mock quantities are flagged clearly in plots and reports.
    """

    def __init__(self):
        self.data: pd.DataFrame | None = None
        self.cluster_members: dict     = {}
        self.sed_results: dict         = {}
        self.analysis_results: dict    = {}

        self.column_mapping = {
            "cluster_id":       "cl_id",
            "halo_mass":        "halo_mass",
            "stellar_mass":     "stellar_mass",
            "log10_stellar_mass": None,
            "g_mag":            "g_mag",
            "r_mag":            "r_mag",
            "redshift":         "redshift",
            "ra":               "ra",
            "dec":              "dec",
        }

        self.color_cut = _COLOR_CUT
        self.mass_bins = np.arange(9.0, 12.2, 0.3)
        self._mock_data = False

    # -----------------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------------

    def load_data(self, data_source, file_format: str = "csv") -> bool:
        try:
            if isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
            elif file_format.lower() == "csv":
                self.data = pd.read_csv(data_source)
            elif file_format.lower() == "fits" and HAVE_ASTROPY:
                with fits.open(data_source) as hdul:
                    self.data = Table(hdul[1].data).to_pandas()
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            self._process_data()
            self._mock_data = False
            logger.info("Loaded %d galaxies from file.", len(self.data))
            return True
        except Exception as e:
            logger.error("Error loading data: %s", e)
            return False

    def _process_data(self):
        for key, col in self.column_mapping.items():
            if col and col in self.data.columns:
                self.data[key] = self.data[col]

        if "log10_stellar_mass" in self.data.columns:
            self.data["log10M"] = self.data["log10_stellar_mass"]
        elif "stellar_mass" in self.data.columns:
            self.data["log10M"] = np.log10(self.data["stellar_mass"].clip(lower=1e4))

        if "g_mag" in self.data.columns and "r_mag" in self.data.columns:
            self.data["g_r"]    = self.data["g_mag"] - self.data["r_mag"]
            self.data["is_red"] = self.data["g_r"] >= self.color_cut

        for col in ["halo_mass", "redshift", "ra", "dec", "log10M"]:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

    # -----------------------------------------------------------------------
    # MOCK DATA — grounded in Baldry+2012, Peng+2010, Speagle+2014
    # -----------------------------------------------------------------------

    def generate_mock_data(self, n_galaxies: int = 1000,
                           n_clusters: int = 5,
                           z_cluster: float = 1.0) -> bool:
        """
        Generate mock galaxy catalog with physically motivated properties.

        Galaxy masses are drawn from the Baldry+2012 double-Schechter SMF
        (field) or the Vulcani+2013 cluster-enhanced SMF.  Red fractions
        follow Peng+2010.  SFRs follow the Speagle+2014 SFMS at the
        specified cluster redshift.

        Parameters
        ----------
        n_galaxies  : total number of galaxies
        n_clusters  : number of galaxy clusters
        z_cluster   : mean redshift of clusters (affects SFMS normalisation)
        """
        rng = np.random.default_rng(seed=42)

        # Approximate cosmic time at z_cluster (flat ΛCDM, rough formula)
        # t(z) ≈ 13.8 / (1+z)^1.5 Gyr  — good to ~20% for 0.5<z<3
        t_gyr = 13.8 / (1.0 + z_cluster) ** 1.5

        n_cluster_gals = int(n_galaxies * 0.30)
        n_field_gals   = n_galaxies - n_cluster_gals

        # --- cluster positions ---
        cl_ra  = rng.uniform(149.5, 150.5, n_clusters)
        cl_dec = rng.uniform(1.8,   2.2,   n_clusters)
        cl_z   = rng.normal(z_cluster, 0.05, n_clusters)
        cl_log_mhalo = rng.uniform(14.0, 15.0, n_clusters)  # log(M_halo/M☉)

        rows = []

        # --- cluster members ---
        log_m_cluster = _sample_log_mass(n_cluster_gals, _SCHECHTER_CLUSTER,
                                          log_m_min=9.5)
        for i in range(n_cluster_gals):
            cl_idx   = rng.integers(0, n_clusters)
            log_m    = log_m_cluster[i]
            in_cl    = True
            f_red    = _peng_red_fraction(log_m, in_cluster=True)
            is_red   = rng.random() < f_red
            sfr      = (_speagle_sfr(log_m, t_gyr, scatter_dex=0.25) * 0.3
                        if is_red
                        else _speagle_sfr(log_m, t_gyr, scatter_dex=0.25))
            sfr      = max(sfr, 1e-3)

            # Photometry from mass-to-light + dust (simplified)
            r_mag, g_r = _mock_photometry(log_m, is_red, rng)

            rows.append({
                "galaxy_id":    i,
                "cl_id":        f"cluster_{cl_idx}",
                "halo_mass":    10 ** cl_log_mhalo[cl_idx],
                "stellar_mass": 10 ** log_m,
                "log10M":       log_m,
                "ra":           rng.normal(cl_ra[cl_idx],  0.08),
                "dec":          rng.normal(cl_dec[cl_idx], 0.08),
                "redshift":     rng.normal(cl_z[cl_idx],   0.02),
                "sfr":          sfr,
                "g_mag":        g_r + r_mag,
                "r_mag":        r_mag,
                "g_r":          g_r,
                "is_red":       is_red,
            })

        # --- field galaxies ---
        log_m_field = _sample_log_mass(n_field_gals, _SCHECHTER_FIELD,
                                        log_m_min=8.5)
        for i in range(n_field_gals):
            log_m  = log_m_field[i]
            f_red  = _peng_red_fraction(log_m, in_cluster=False)
            is_red = rng.random() < f_red
            sfr    = (_speagle_sfr(log_m, t_gyr, scatter_dex=0.3) * 0.1
                      if is_red
                      else _speagle_sfr(log_m, t_gyr, scatter_dex=0.3))
            sfr    = max(sfr, 1e-3)
            r_mag, g_r = _mock_photometry(log_m, is_red, rng)

            rows.append({
                "galaxy_id":    n_cluster_gals + i,
                "cl_id":        "field",
                "halo_mass":    10 ** rng.uniform(12.0, 14.0),
                "stellar_mass": 10 ** log_m,
                "log10M":       log_m,
                "ra":           rng.uniform(149.0, 151.0),
                "dec":          rng.uniform(1.5,   2.5),
                "redshift":     rng.normal(z_cluster, 0.5),
                "sfr":          sfr,
                "g_mag":        g_r + r_mag,
                "r_mag":        r_mag,
                "g_r":          g_r,
                "is_red":       is_red,
            })

        self.data       = pd.DataFrame(rows)
        self._mock_data = True
        logger.info("Generated %d mock galaxies (Baldry+2012 SMF, Peng+2010 f_red, "
                    "Speagle+2014 SFMS).", len(self.data))
        return True

    # -----------------------------------------------------------------------
    # ANALYSIS METHODS
    # -----------------------------------------------------------------------

    def analyze_spatial_distribution(self) -> dict:
        if self.data is None:
            raise ValueError("No data loaded.")

        H, xe, ye = np.histogram2d(
            self.data["ra"], self.data["dec"], bins=20, density=True)

        results = {
            "ra_range":         [self.data["ra"].min(),  self.data["ra"].max()],
            "dec_range":        [self.data["dec"].min(), self.data["dec"].max()],
            "density_map":      H,
            "ra_edges":         xe,
            "dec_edges":        ye,
            "overdense_regions": H > np.percentile(H, 80),
            "n_galaxies":       len(self.data),
        }
        self.analysis_results["spatial"] = results
        return results

    def detect_clusters_redshift(self, z_bins: int = 50,
                                  significance_threshold: float = 3.0) -> dict:
        if self.data is None:
            raise ValueError("No data loaded.")

        counts, edges = np.histogram(self.data["redshift"], bins=z_bins)
        centers       = (edges[1:] + edges[:-1]) / 2.0
        background    = gaussian_filter1d(counts.astype(float), sigma=2)
        residuals     = counts - background
        significance  = residuals / np.sqrt(background + 1)

        peaks = [i for i in range(1, len(significance) - 1)
                 if (significance[i] > significance_threshold
                     and significance[i] > significance[i-1]
                     and significance[i] > significance[i+1])]

        detected = []
        for j, pi in enumerate(peaks):
            z_cl = centers[pi]
            mask = np.abs(self.data["redshift"] - z_cl) < 0.05
            detected.append({
                "cluster_id":  f"detected_{j}",
                "redshift":    z_cl,
                "n_members":   int(mask.sum()),
                "significance": float(significance[pi]),
                "members":     self.data[mask].copy(),
            })

        results = {
            "z_histogram":      {"counts": counts, "bin_centers": centers,
                                 "bin_edges": edges},
            "background":       background,
            "significance":     significance,
            "detected_clusters": detected,
            "n_clusters_detected": len(detected),
        }
        self.analysis_results["redshift_clustering"] = results
        return results

    def separate_cluster_field(self, cluster_z_tolerance: float = 0.05) -> dict:
        if self.data is None:
            raise ValueError("No data loaded.")

        if "cl_id" in self.data.columns:
            cl_gals   = self.data[self.data["cl_id"] != "field"]
            field_gals = self.data[self.data["cl_id"] == "field"]
        elif "redshift_clustering" in self.analysis_results:
            cl_list = self.analysis_results["redshift_clustering"]["detected_clusters"]
            cl_gals = pd.concat([c["members"] for c in cl_list], ignore_index=True)
            field_gals = self.data[~self.data.index.isin(cl_gals.index)]
        else:
            # No cluster info: use overdensity threshold on RA-Dec
            logger.warning("No cluster membership info — treating all as field.")
            cl_gals    = self.data.iloc[:0]
            field_gals = self.data

        results = {
            "cluster_members": cl_gals,
            "field_galaxies":  field_gals,
            "n_cluster":       len(cl_gals),
            "n_field":         len(field_gals),
            "cluster_fraction": len(cl_gals) / max(len(self.data), 1),
        }
        self.cluster_members = results
        return results

    def setup_bagpipes_model(self, sfh_model: str = "exponential",
                              dust_model: str = "Calzetti") -> dict:
        if sfh_model == "exponential":
            sfh_comp = {"exponential": {
                "age": (0.1, 15.0), "tau": (0.3, 10.0),
                "massformed": (1.0, 15.0), "metallicity": (0.0, 2.5),
            }}
        elif sfh_model == "double_power_law":
            sfh_comp = {"dblplaw": {
                "tau": (0.3, 10.0), "alpha": (0.01, 1000.0), "beta": (0.01, 1000.0),
                "massformed": (1.0, 15.0), "metallicity": (0.0, 2.5),
            }}
        else:
            raise ValueError(f"Unsupported SFH model: {sfh_model}")

        return {
            "redshift": (0.0, 10.0),
            "dust":     {"type": dust_model, "Av": (0.0, 2.0)},
            **sfh_comp,
        }

    def run_sed_fitting(self, galaxy_subset=None,
                        fit_instructions=None, n_live: int = 400) -> dict:
        if galaxy_subset is None:
            galaxy_subset = self.data
        if fit_instructions is None:
            fit_instructions = self.setup_bagpipes_model()

        if not HAVE_BAGPIPES:
            return self._mock_sed_fitting_results(galaxy_subset)

        results = {}
        for _, galaxy in galaxy_subset.iterrows():
            try:
                photometry = self._prepare_photometry(galaxy)
                gobj = pipes.galaxy(str(galaxy["galaxy_id"]),
                                    lambda x: photometry,
                                    spectrum_exists=False)
                fit = pipes.fit(gobj, fit_instructions)
                fit.fit(verbose=False, n_live=n_live)
                post = fit.posterior
                results[galaxy["galaxy_id"]] = {
                    "stellar_mass":     post.samples["stellar_mass"].mean(),
                    "stellar_mass_err": post.samples["stellar_mass"].std(),
                    "sfr":              post.samples.get("sfr", np.array([0])).mean(),
                    "sfr_err":          post.samples.get("sfr", np.array([0])).std(),
                    "age":              post.samples["age"].mean(),
                    "age_err":          post.samples["age"].std(),
                    "metallicity":      post.samples["metallicity"].mean(),
                    "metallicity_err":  post.samples["metallicity"].std(),
                    "av":               post.samples["dust:Av"].mean(),
                    "av_err":           post.samples["dust:Av"].std(),
                    "mock":             False,
                }
            except Exception as e:
                logger.error("Error fitting galaxy %s: %s", galaxy["galaxy_id"], e)
        self.sed_results = results
        return results

    def _mock_sed_fitting_results(self, galaxy_subset: pd.DataFrame) -> dict:
        """
        Mock SED fitting using the Speagle+2014 SFMS and Peng+2010 f_red.

        Stellar mass recovery: Gaussian scatter of 0.15 dex around input mass
        (typical Bagpipes photometric accuracy, Carnall+2019).
        Age: derived from sSFR assuming exponential SFH.
        """
        rng     = np.random.default_rng(seed=1)
        results = {}

        for _, galaxy in galaxy_subset.iterrows():
            log_m  = float(galaxy.get("log10M", 10.0))
            is_red = bool(galaxy.get("is_red", False))
            z      = float(galaxy.get("redshift", 1.0))
            t_gyr  = 13.8 / (1.0 + z) ** 1.5

            # Stellar mass: 0.15 dex scatter (Carnall+2019 typical uncertainty)
            stellar_mass_fit = log_m + rng.normal(0, 0.15)
            mass_err         = 0.15

            # SFR from SFMS with quenching
            if is_red:
                # Quenched: sSFR 2 dex below SFMS
                log_sfr = _speagle_sfr.__wrapped__(log_m, t_gyr) if hasattr(
                    _speagle_sfr, "__wrapped__") else (
                    (0.84 - 0.026 * t_gyr) * log_m - (6.51 - 0.11 * t_gyr) - 2.0)
                sfr     = max(10 ** (float(log_sfr) + rng.normal(0, 0.3)), 1e-3)
                sfr_err = sfr * 0.5
                # Old age: typical quenching lookback time ~2-5 Gyr
                age     = rng.uniform(3.0, min(t_gyr, 8.0))
                age_err = age * 0.25
            else:
                sfr_mean = _speagle_sfr(log_m, t_gyr, scatter_dex=0.0)
                sfr      = max(sfr_mean * 10 ** rng.normal(0, 0.3), 1e-2)
                sfr_err  = sfr * 0.3
                # Young age: consistent with ongoing star formation
                age     = rng.uniform(0.5, min(t_gyr * 0.6, 5.0))
                age_err = age * 0.2

            # Metallicity: mass-metallicity relation (Tremonti+2004 simplified)
            # 12+log(O/H) = -0.185*(log_m - 10.5)^2 + 9.07  → normalised to Z☉
            met_solar = np.clip(
                10 ** (-0.185 * (log_m - 10.5)**2 + 9.07 - 8.69), 0.1, 2.5)
            met_err   = met_solar * 0.15

            # Dust: Av from IRX-β for star-forming; low for quenched
            av     = (rng.exponential(0.4) if not is_red
                      else rng.exponential(0.1))
            av_err = max(av * 0.2, 0.05)

            results[galaxy["galaxy_id"]] = {
                "stellar_mass":     stellar_mass_fit,
                "stellar_mass_err": mass_err,
                "sfr":              sfr,
                "sfr_err":          sfr_err,
                "age":              age,
                "age_err":          age_err,
                "metallicity":      met_solar,
                "metallicity_err":  met_err,
                "av":               av,
                "av_err":           av_err,
                "mock":             True,
            }

        self.sed_results = results
        return results

    def _prepare_photometry(self, galaxy) -> np.ndarray:
        bands  = ["g", "r"]
        fluxes, errs = [], []
        for band in bands:
            mag_col = f"{band}_mag"
            if mag_col in galaxy.index and not pd.isna(galaxy[mag_col]):
                flux = 10 ** (-0.4 * (float(galaxy[mag_col]) - 25.0))
            else:
                flux = float(np.random.lognormal(0, 1))
            err = flux * 0.1
            fluxes.append(flux); errs.append(err)
        return np.column_stack([fluxes, errs])

    # -----------------------------------------------------------------------
    # STELLAR MASS FUNCTIONS
    # -----------------------------------------------------------------------

    def analyze_stellar_mass_functions(self, mass_bins=None) -> dict:
        if mass_bins is None:
            mass_bins = self.mass_bins
        if not self.cluster_members:
            self.separate_cluster_field()

        cl_gals    = self.cluster_members["cluster_members"]
        field_gals = self.cluster_members["field_galaxies"]

        def _smf(gals, bins):
            if "log10M" not in gals.columns or len(gals) == 0:
                return np.zeros(len(bins) - 1), (bins[:-1] + bins[1:]) / 2
            counts, _ = np.histogram(gals["log10M"].dropna(), bins=bins)
            centers   = (bins[:-1] + bins[1:]) / 2
            phi       = counts / np.diff(bins)  # per dex (volume=1 for relative comparison)
            return phi, centers

        phi_cl,  mc = _smf(cl_gals,    mass_bins)
        phi_fi,  _  = _smf(field_gals, mass_bins)

        # Reference SMF from Baldry+2012 (for overplotting)
        phi_ref_field   = _double_schechter_logm(mc, _SCHECHTER_FIELD)
        phi_ref_cluster = _double_schechter_logm(mc, _SCHECHTER_CLUSTER)

        def _smf_color(gals, bins, red: bool):
            if "is_red" not in gals.columns or len(gals) == 0:
                return np.zeros(len(bins) - 1)
            sub = gals[gals["is_red"] == red]
            counts, _ = np.histogram(sub["log10M"].dropna(), bins=bins)
            return counts / np.diff(bins)

        results = {
            "mass_bins":    mass_bins,
            "mass_centers": mc,
            "cluster": {
                "total": phi_cl,
                "red":   _smf_color(cl_gals,    mass_bins, True),
                "blue":  _smf_color(cl_gals,    mass_bins, False),
            },
            "field": {
                "total": phi_fi,
                "red":   _smf_color(field_gals, mass_bins, True),
                "blue":  _smf_color(field_gals, mass_bins, False),
            },
            "reference": {
                "baldry2012_field":   phi_ref_field,
                "vulcani2013_cluster": phi_ref_cluster,
            },
            "mock": self._mock_data,
        }
        self.analysis_results["stellar_mass_functions"] = results
        return results

    # -----------------------------------------------------------------------
    # RED FRACTION
    # -----------------------------------------------------------------------

    def compute_red_fraction(self, mass_bins=None) -> dict:
        if mass_bins is None:
            mass_bins = self.mass_bins
        if not self.cluster_members:
            self.separate_cluster_field()

        cl_gals    = self.cluster_members["cluster_members"]
        field_gals = self.cluster_members["field_galaxies"]
        mc         = (mass_bins[:-1] + mass_bins[1:]) / 2

        def _rf(gals, bins):
            frac, err = [], []
            for i in range(len(bins) - 1):
                mask = ((gals["log10M"] >= bins[i]) & (gals["log10M"] < bins[i+1])
                        if "log10M" in gals.columns else np.zeros(len(gals), bool))
                sub  = gals[mask]
                n    = len(sub)
                if n > 0 and "is_red" in sub.columns:
                    nr = sub["is_red"].sum()
                    f  = nr / n
                    e  = np.sqrt(f * (1 - f) / n)
                else:
                    f, e = 0.0, 0.0
                frac.append(f); err.append(e)
            return np.array(frac), np.array(err)

        rf_cl,  e_cl  = _rf(cl_gals,    mass_bins)
        rf_fi,  e_fi  = _rf(field_gals, mass_bins)

        # Peng+2010 theoretical prediction for comparison
        peng_cluster = np.array([_peng_red_fraction(m, in_cluster=True)  for m in mc])
        peng_field   = np.array([_peng_red_fraction(m, in_cluster=False) for m in mc])

        results = {
            "mass_bins":    mass_bins,
            "mass_centers": mc,
            "cluster": {"red_fraction": rf_cl, "red_fraction_err": e_cl},
            "field":   {"red_fraction": rf_fi, "red_fraction_err": e_fi},
            "environmental_effect": {
                "delta_red_fraction":     rf_cl - rf_fi,
                "delta_red_fraction_err": np.sqrt(e_cl**2 + e_fi**2),
            },
            "reference": {
                "peng2010_cluster": peng_cluster,
                "peng2010_field":   peng_field,
            },
            "mock": self._mock_data,
        }
        self.analysis_results["red_fraction"] = results
        return results

    # -----------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------

    def _mock_label(self, ax):
        if self._mock_data:
            ax.text(0.98, 0.02,
                    "SIMULATED DATA — Baldry+2012 / Peng+2010 / Speagle+2014",
                    transform=ax.transAxes, fontsize=7, color="darkorange",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                              ec="darkorange", alpha=0.9))

    def plot_spatial_distribution(self, save_path=None):
        if "spatial" not in self.analysis_results:
            self.analyze_spatial_distribution()
        if not self.cluster_members:
            self.separate_cluster_field()

        res       = self.analysis_results["spatial"]
        cl_gals   = self.cluster_members["cluster_members"]
        fi_gals   = self.cluster_members["field_galaxies"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(fi_gals["ra"], fi_gals["dec"],
                    alpha=0.5, s=10, c="steelblue", label=f"Field ({len(fi_gals)})")
        ax1.scatter(cl_gals["ra"], cl_gals["dec"],
                    alpha=0.8, s=20, c="tomato",    label=f"Cluster ({len(cl_gals)})")
        ax1.set_xlabel("RA [°]"); ax1.set_ylabel("Dec [°]")
        ax1.set_title("Galaxy Spatial Distribution"); ax1.legend(); ax1.grid(alpha=0.3)
        self._mock_label(ax1)

        im = ax2.imshow(res["density_map"].T, origin="lower", cmap="viridis",
                        extent=[res["ra_edges"][0], res["ra_edges"][-1],
                                res["dec_edges"][0], res["dec_edges"][-1]],
                        aspect="auto")
        ax2.set_xlabel("RA [°]"); ax2.set_ylabel("Dec [°]")
        ax2.set_title("Galaxy Density Map")
        fig.colorbar(im, ax=ax2).set_label("Number Density")
        self._mock_label(ax2)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, (ax1, ax2)

    def plot_color_magnitude_diagram(self, save_path=None):
        if not self.cluster_members:
            self.separate_cluster_field()

        cl_gals = self.cluster_members["cluster_members"]
        fi_gals = self.cluster_members["field_galaxies"]

        fig, ax = plt.subplots(figsize=(9, 7))

        for gals, color, label in [
            (fi_gals, "steelblue", f"Field ({len(fi_gals)})"),
            (cl_gals, "tomato",    f"Cluster ({len(cl_gals)})"),
        ]:
            if "r_mag" in gals.columns and "g_r" in gals.columns:
                ax.scatter(gals["r_mag"], gals["g_r"],
                           alpha=0.5, s=15, c=color, label=label)

        # Red sequence and color cut
        r_range = np.linspace(18, 26, 100)
        ax.plot(r_range, 0.65 + 0.02 * (r_range - 22), "k--", alpha=0.6,
                label="Red sequence (illustrative)")
        ax.axhline(self.color_cut, color="darkorange", ls=":", alpha=0.8,
                   label=f"g–r = {self.color_cut} (Bell+2004 cut)")

        ax.set_xlabel("r [mag]"); ax.set_ylabel("g – r")
        ax.set_title("Color–Magnitude Diagram")
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.invert_xaxis()
        self._mock_label(ax)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_stellar_mass_functions(self, save_path=None):
        if "stellar_mass_functions" not in self.analysis_results:
            self.analyze_stellar_mass_functions()

        res = self.analysis_results["stellar_mass_functions"]
        mc  = res["mass_centers"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Total SMF
        for label, phi, color in [
            ("Cluster (measured)", res["cluster"]["total"], "tomato"),
            ("Field (measured)",   res["field"]["total"],   "steelblue"),
        ]:
            mask = phi > 0
            ax1.semilogy(mc[mask], phi[mask], "o-", color=color, label=label, ms=5)

        # Baldry/Vulcani reference lines
        ax1.semilogy(mc, res["reference"]["baldry2012_field"],   "b--",
                     alpha=0.5, label="Baldry+2012 (field)")
        ax1.semilogy(mc, res["reference"]["vulcani2013_cluster"], "r--",
                     alpha=0.5, label="Vulcani+2013 (cluster)")

        ax1.set_xlabel("log(M*/M☉)"); ax1.set_ylabel("φ [Mpc⁻³ dex⁻¹]")
        ax1.set_title("Stellar Mass Functions"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
        self._mock_label(ax1)

        # By color
        for label, phi, ls, color in [
            ("Cluster red",  res["cluster"]["red"],  "-",  "tomato"),
            ("Cluster blue", res["cluster"]["blue"], "--", "tomato"),
            ("Field red",    res["field"]["red"],    "-",  "steelblue"),
            ("Field blue",   res["field"]["blue"],   "--", "steelblue"),
        ]:
            mask = phi > 0
            if mask.sum() > 0:
                ax2.semilogy(mc[mask], phi[mask], ls, color=color,
                             label=label, lw=2)

        ax2.set_xlabel("log(M*/M☉)"); ax2.set_ylabel("φ [Mpc⁻³ dex⁻¹]")
        ax2.set_title("SMF by Color"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        self._mock_label(ax2)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, (ax1, ax2)

    def plot_red_fraction(self, save_path=None):
        if "red_fraction" not in self.analysis_results:
            self.compute_red_fraction()

        res = self.analysis_results["red_fraction"]
        mc  = res["mass_centers"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Measured
        for env, color in [("cluster", "tomato"), ("field", "steelblue")]:
            ax1.errorbar(mc,
                         res[env]["red_fraction"],
                         yerr=res[env]["red_fraction_err"],
                         fmt="o-", color=color, ms=5, capsize=3,
                         label=f"{env.capitalize()} (measured)")

        # Peng+2010 reference
        ax1.plot(mc, res["reference"]["peng2010_cluster"], "r--",
                 alpha=0.6, label="Peng+2010 (cluster)")
        ax1.plot(mc, res["reference"]["peng2010_field"],   "b--",
                 alpha=0.6, label="Peng+2010 (field)")

        ax1.set_xlabel("log(M*/M☉)"); ax1.set_ylabel("Red Fraction")
        ax1.set_title("Red Fraction vs Stellar Mass (Peng+2010)")
        ax1.set_ylim(0, 1); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
        self._mock_label(ax1)

        # Environmental effect
        ax2.errorbar(mc,
                     res["environmental_effect"]["delta_red_fraction"],
                     yerr=res["environmental_effect"]["delta_red_fraction_err"],
                     fmt="go-", ms=5, capsize=3, label="Δ f_red (measured)")
        ax2.axhline(_PENG_ENV_DELTA_RED_FRAC, color="green", ls="--", alpha=0.6,
                    label=f"Peng+2010 prediction (+{_PENG_ENV_DELTA_RED_FRAC:.2f})")
        ax2.axhline(0, color="black", ls="-", lw=0.5, alpha=0.4)

        ax2.set_xlabel("log(M*/M☉)"); ax2.set_ylabel("Δ f_red = cluster − field")
        ax2.set_title("Environmental Quenching Excess")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        self._mock_label(ax2)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, (ax1, ax2)

    # -----------------------------------------------------------------------
    # SUMMARY REPORT
    # -----------------------------------------------------------------------

    def generate_summary_report(self) -> dict:
        if self.data is None:
            raise ValueError("No data loaded.")
        if not self.cluster_members:
            self.separate_cluster_field()

        cl_gals = self.cluster_members["cluster_members"]
        fi_gals = self.cluster_members["field_galaxies"]

        report = {
            "data_provenance": {
                "mode": "mock — Baldry+2012 SMF / Peng+2010 f_red / Speagle+2014 SFMS"
                         if self._mock_data else "real observations",
                "references": [
                    "Baldry et al. 2012, MNRAS 421, 621",
                    "Vulcani et al. 2013, A&A 550, A58",
                    "Peng et al. 2010, ApJ 721, 193",
                    "Speagle et al. 2014, ApJS 214, 15",
                    "Bell et al. 2004, ApJ 608, 752",
                ],
            },
            "data_summary": {
                "total_galaxies":   len(self.data),
                "cluster_members":  len(cl_gals),
                "field_galaxies":   len(fi_gals),
                "cluster_fraction": len(cl_gals) / max(len(self.data), 1),
            },
        }

        if "is_red" in self.data.columns:
            report["color_statistics"] = {
                "cluster_red_fraction":  float(cl_gals["is_red"].mean()) if len(cl_gals) > 0 else 0,
                "field_red_fraction":    float(fi_gals["is_red"].mean()) if len(fi_gals) > 0 else 0,
                "environmental_excess":  float(cl_gals["is_red"].mean() - fi_gals["is_red"].mean())
                                         if len(cl_gals) > 0 and len(fi_gals) > 0 else 0,
                "peng2010_prediction":   _PENG_ENV_DELTA_RED_FRAC,
            }

        if "log10M" in self.data.columns:
            report["mass_statistics"] = {
                "cluster_median_log_mass": float(cl_gals["log10M"].median()) if len(cl_gals) > 0 else 0,
                "field_median_log_mass":   float(fi_gals["log10M"].median()) if len(fi_gals) > 0 else 0,
                "log_mass_range":          [float(self.data["log10M"].min()),
                                            float(self.data["log10M"].max())],
            }

        if self.sed_results:
            masses = [r["stellar_mass"] for r in self.sed_results.values()]
            sfrs   = [r["sfr"]          for r in self.sed_results.values()]
            mocks  = [r.get("mock", True) for r in self.sed_results.values()]
            report["sed_fitting"] = {
                "n_galaxies_fitted":   len(self.sed_results),
                "mean_log_mass":       float(np.mean(masses)),
                "mean_sfr_msun_yr":    float(np.mean(sfrs)),
                "mode":                "mock" if any(mocks) else "bagpipes",
            }

        return report


# ---------------------------------------------------------------------------
# Module-level photometry helper
# ---------------------------------------------------------------------------

def _mock_photometry(log_m: float, is_red: bool,
                     rng: np.random.Generator) -> tuple[float, float]:
    """
    Approximate r-band magnitude and g-r color from stellar mass.

    Mass-to-light ratio from Bell+2003 (ApJS 149, 289):
      log(M/L_r) ≈ -0.306 + 1.097*(g-r)
    Combined with a luminosity-distance appropriate for z~1 (illustrative).
    """
    # Color: bimodal distribution (Bell+2004)
    if is_red:
        g_r = float(rng.normal(0.80, 0.08))      # red sequence
    else:
        g_r = float(rng.normal(0.35, 0.12))      # blue cloud

    g_r = np.clip(g_r, -0.2, 1.4)

    # log(M/L_r) from Bell+2003
    log_ml = -0.306 + 1.097 * g_r
    # Absolute magnitude in r: M_r = M_sun_r - 2.5*(log_m - log_ml)
    # M_sun_r ≈ 4.65
    M_r = 4.65 - 2.5 * (log_m - log_ml)
    # Apparent magnitude at z~1 (DM~44.1 for standard ΛCDM)
    DM   = 44.1
    r_mag = float(M_r + DM + rng.normal(0, 0.3))
    return r_mag, g_r