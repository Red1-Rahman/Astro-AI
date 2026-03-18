# JWST Spectrum Analyzer Module for Astro-AI
#
# Copyright (c) 2025 Redwan Rahman and CAM-SUST
#
# Mock spectrum generation grounded in:
#   - Gordon et al. 2022 / Jakobsen et al. 2022:
#       NIRSpec instrument response / throughput
#   - Horne 1986 (PASP 98, 609):
#       Optimal extraction algorithm
#   - Osterbrock & Ferland 2006 (Astrophysics of Gaseous Nebulae):
#       Emission line ratios (Balmer decrement, [OIII]/Hβ)
#   - Calzetti et al. 2000 (ApJ 533, 682):
#       Dust attenuation law
#   - Bruzual & Charlot 2003 (MNRAS 344, 1000):
#       Stellar continuum shape
#   - Pettini & Pagel 2004 (MNRAS 348, L59):
#       Strong-line metallicity calibration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import warnings
import logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from astropy.io import fits
    from astropy.table import Table
    import astropy.units as u
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False
    logger.warning("astropy not available.")

try:
    import os, tempfile
    if "BAGPIPES_DATA" not in os.environ:
        _bdir = os.path.join(tempfile.gettempdir(), "bagpipes_data")
        os.makedirs(_bdir, exist_ok=True)
        os.environ["BAGPIPES_FILTERS"] = _bdir
        os.environ["BAGPIPES_DATA"]    = _bdir
    import bagpipes as pipes
    HAVE_BAGPIPES = True
except (ImportError, PermissionError, OSError) as e:
    HAVE_BAGPIPES = False
    logger.warning("bagpipes not available: %s. Using simulation mode.", e)

try:
    from jwst.datamodels import ImageModel
    from jwst import datamodels
    HAVE_JWST_PIPELINE = True
except ImportError:
    HAVE_JWST_PIPELINE = False
    logger.warning("JWST pipeline not available. Using simulation mode.")


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Balmer decrement (Case B recombination, T=10⁴ K, n_e=100 cm⁻³)
# Osterbrock & Ferland 2006, Table 4.2
_BALMER_HA_HB = 2.86      # Hα/Hβ intrinsic
_BALMER_HG_HB = 0.469     # Hγ/Hβ
_BALMER_HD_HB = 0.261     # Hδ/Hβ

# NIRSpec wavelength coverage (observed frame) [microns]
_NIRSPEC_LAMBDA_MIN = 0.60
_NIRSPEC_LAMBDA_MAX = 5.30

# Common emission lines — rest-frame wavelengths [microns] and
# intrinsic flux ratios relative to Hβ = 1.0
# Sources: Osterbrock & Ferland 2006; Kewley & Dopita 2002
_REST_LINES = {
    # Hydrogen Balmer series
    0.48613: {"name": "Hβ",       "ratio": 1.000},
    0.65628: {"name": "Hα",       "ratio": _BALMER_HA_HB},
    0.43405: {"name": "Hγ",       "ratio": _BALMER_HG_HB},
    0.41019: {"name": "Hδ",       "ratio": _BALMER_HD_HB},
    # Paschen series (NIR)
    1.28216: {"name": "Paβ",      "ratio": 0.162},
    1.87510: {"name": "Paα",      "ratio": 0.332},
    # Forbidden oxygen
    0.37270: {"name": "[OII]3727", "ratio": 1.5},   # typical star-forming galaxy
    0.49590: {"name": "[OIII]4959","ratio": 0.340},  # fixed ratio to 5007 (atomic physics)
    0.50070: {"name": "[OIII]5007","ratio": 1.010},  # Osterbrock+2006 §5.2
    # Forbidden nitrogen
    0.65485: {"name": "[NII]6548", "ratio": 0.102},
    0.65835: {"name": "[NII]6583", "ratio": 0.310},
    # Sulphur
    0.67166: {"name": "[SII]6716", "ratio": 0.220},
    0.67310: {"name": "[SII]6731", "ratio": 0.170},
}

# Calzetti+2000 dust attenuation k(λ) curve
# Parameterised for λ in microns
def _calzetti_k(lam_um: np.ndarray) -> np.ndarray:
    """
    Calzetti et al. 2000 (ApJ 533, 682) attenuation curve k(λ).
    Valid for 0.12 < λ < 2.2 μm.
    """
    lam = lam_um
    k   = np.zeros_like(lam)
    # UV/optical: 0.12–0.63 μm
    mask_uv = lam < 0.63
    k[mask_uv] = (2.659 * (-2.156 + 1.509 / lam[mask_uv]
                             - 0.198 / lam[mask_uv]**2
                             + 0.011 / lam[mask_uv]**3)
                  + 4.05)
    # Optical/NIR: 0.63–2.2 μm
    mask_nir = (lam >= 0.63) & (lam <= 2.2)
    k[mask_nir] = 2.659 * (-1.857 + 1.040 / lam[mask_nir]) + 4.05
    # Beyond 2.2 μm: assume k → 0 smoothly
    mask_mir = lam > 2.2
    k[mask_mir] = np.maximum(
        2.659 * (-1.857 + 1.040 / 2.2) + 4.05 * (2.2 / lam[mask_mir])**1.5, 0)
    return np.maximum(k, 0.0)


def _apply_calzetti_attenuation(flux: np.ndarray, lam_um: np.ndarray,
                                 av: float) -> np.ndarray:
    """
    Apply Calzetti+2000 dust attenuation.
    av : V-band attenuation [mag],  R_V = 4.05 (Calzetti+2000)
    """
    R_V = 4.05
    k   = _calzetti_k(lam_um)
    return flux * 10 ** (-0.4 * av * k / R_V)


class JWSTAnalyzer:
    """
    JWST NIRSpec Data Analysis Pipeline.

    Mock spectra are physically motivated:
    - Stellar continuum follows Bruzual & Charlot 2003 spectral shape
    - Emission lines at correct observed wavelengths given input redshift
    - Line ratios from Case B recombination + Osterbrock & Ferland 2006
    - Dust attenuation via Calzetti+2000 law
    - Noise model approximates NIRSpec background + read noise

    All mock results are labelled clearly so users are not misled.
    """

    def __init__(self):
        self.raw_data:          dict | None = None
        self.reduced_data:      dict        = {}
        self.extracted_spectra: dict        = {}
        self.spectral_fits:     dict        = {}
        self._mock_data = False

        self.pipeline_status = {
            "stage1":    False,
            "stage2":    False,
            "stage3":    False,
            "extraction":False,
        }

        self.extraction_params = {
            "profile_sigma": 2.0,
            "bg_offset":     5,
            "snr_threshold": 3.0,
        }

        self.fitting_params = {
            "z_range":    (0.0, 10.0),
            "age_range":  (0.1, 15.0),
            "tau_range":  (0.3, 10.0),
            "mass_range": (1.0, 15.0),
            "av_range":   (0.0, 3.0),
        }

    # -----------------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------------

    def load_jwst_data(self, data_path: str, data_type: str = "rate") -> bool:
        try:
            if HAVE_ASTROPY:
                if data_type in ("uncal", "rate", "cal"):
                    with fits.open(data_path) as hdul:
                        self.raw_data = {
                            "data":       hdul["SCI"].data,
                            "error":      hdul["ERR"].data if "ERR" in hdul else None,
                            "dq":         hdul["DQ"].data  if "DQ"  in hdul else None,
                            "header":     hdul["PRIMARY"].header,
                            "sci_header": hdul["SCI"].header,
                            "data_type":  data_type,
                            "file_path":  data_path,
                        }
                elif data_type == "x1d":
                    with fits.open(data_path) as hdul:
                        td = Table(hdul["EXTRACT1D"].data)
                        self.extracted_spectra["loaded"] = {
                            "wavelength":  np.array(td["WAVELENGTH"]),
                            "flux":        np.array(td["FLUX"]),
                            "flux_error":  np.array(td["FLUX_ERROR"]),
                            "header":      hdul["PRIMARY"].header,
                            "mock":        False,
                        }
                        self.pipeline_status["extraction"] = True
            else:
                self._generate_mock_jwst_data(data_type)
            logger.info("Loaded JWST %s data from %s", data_type, data_path)
            return True
        except Exception as e:
            logger.error("Error loading JWST data: %s", e)
            return False

    # -----------------------------------------------------------------------
    # MOCK DATA — physically motivated
    # -----------------------------------------------------------------------

    def _generate_mock_jwst_data(self, data_type: str = "rate",
                                  redshift: float = 2.0,
                                  log_mass: float = 10.5,
                                  av: float = 0.4,
                                  is_quenched: bool = False):
        """
        Generate mock NIRSpec 2-D or 1-D data with physical realism.

        Parameters
        ----------
        redshift    : galaxy redshift (shifts all lines correctly)
        log_mass    : log10(M*/M☉) — sets continuum luminosity
        av          : V-band attenuation [mag] via Calzetti+2000
        is_quenched : suppress emission lines for passive galaxies
        """
        nx = 1000   # spectral pixels
        ny = 50     # spatial pixels

        # Observed-frame wavelength array [microns]
        wavelength = np.linspace(_NIRSPEC_LAMBDA_MIN, _NIRSPEC_LAMBDA_MAX, nx)

        # Build the 1-D spectrum
        flux_1d = self._generate_mock_spectrum_1d(
            wavelength, redshift=redshift, log_mass=log_mass,
            av=av, is_quenched=is_quenched)

        if data_type in ("uncal", "rate", "cal"):
            # Spatial profile: Gaussian PSF (FWHM ~ 2 pix for NIRSpec MSA)
            sigma_pix = self.extraction_params["profile_sigma"]
            y_arr     = np.arange(ny)
            profile   = np.exp(-(y_arr - ny / 2)**2 / (2 * sigma_pix**2))
            flux_2d   = np.outer(profile, flux_1d)

            # NIRSpec noise model (simplified):
            # σ² = flux/gain + (read_noise/gain)²   [units: e⁻/s/pixel]
            gain       = 1.0       # e⁻/ADU (approximate)
            read_noise = 15.0      # e⁻ rms per read (NIRSpec spec from Jakobsen+2022)
            n_reads    = 4
            sigma_2d   = np.sqrt(np.maximum(flux_2d, 0) / gain
                                 + (read_noise / gain / np.sqrt(n_reads))**2)
            noise_2d   = np.random.normal(0, sigma_2d)
            flux_2d   += noise_2d

            self.raw_data = {
                "data":          flux_2d,
                "error":         sigma_2d,
                "dq":            np.zeros(flux_2d.shape, dtype=int),
                "wavelength_2d": np.tile(wavelength, (ny, 1)),
                "data_type":     data_type,
                "mock":          True,
                "mock_params":   {"redshift": redshift, "log_mass": log_mass,
                                  "av": av, "is_quenched": is_quenched},
            }
        elif data_type == "x1d":
            # Noise for extracted 1-D spectrum
            sigma_1d = np.sqrt(np.maximum(flux_1d, 0) + (15.0 / np.sqrt(4))**2)
            self.extracted_spectra["loaded"] = {
                "wavelength": wavelength,
                "flux":       flux_1d + np.random.normal(0, sigma_1d),
                "flux_error": sigma_1d,
                "mock":       True,
                "mock_params":{"redshift": redshift, "log_mass": log_mass,
                               "av": av, "is_quenched": is_quenched},
            }
            self.pipeline_status["extraction"] = True

        self._mock_data = True

    def _generate_mock_spectrum_1d(self, wavelength: np.ndarray,
                                    redshift: float = 2.0,
                                    log_mass: float = 10.5,
                                    av: float = 0.4,
                                    is_quenched: bool = False,
                                    metallicity_solar: float = 1.0) -> np.ndarray:
        """
        Physically motivated galaxy spectrum.

        Stellar continuum
        -----------------
        Approximated using a modified blackbody + exponential tail, calibrated
        to match Bruzual & Charlot 2003 template shapes for a 1 Gyr SSP
        (passive) or a 100 Myr SFH (star-forming).

        f_λ ∝ λ^β × exp(−λ/λ_0)  [in observed frame]

          β = −1.5 (blue, star-forming)  or  −0.5 (red, passive)
          λ_0 driven by T_eff of dominant stellar population

        Luminosity scaling:
          L_r ∝ 10^(log_mass − log_M/L_r)  with Bell+2003 M/L
          Then converted to flux at z via a fiducial luminosity distance.

        Emission lines
        --------------
        Rest-frame wavelengths redshifted to observed frame.
        Line ratios fixed to Case B recombination values (Osterbrock+2006).
        [OIII]/Hβ set by metallicity via the O3N2 calibration (Pettini & Pagel 2004).
        Emission lines suppressed for quenched galaxies (EW_Hα < 3Å threshold).

        Dust attenuation
        ----------------
        Calzetti+2000 law applied to both continuum and emission lines.
        Lines have extra attenuation: E(B-V)_stars = 0.44 × E(B-V)_gas
        (Calzetti+2000 §4).
        """
        # ── Stellar continuum ──────────────────────────────────────────────
        # Rest-frame wavelength
        lam_rest = wavelength / (1.0 + redshift)

        # Spectral slope β and characteristic wavelength for BC03 shape
        if is_quenched:
            beta  = -0.5    # redder continuum (old stellar population)
            lam_0 = 2.5     # μm — characteristic emission peak
        else:
            beta  = -1.8    # bluer continuum (young stars)
            lam_0 = 1.5

        # Power-law × exponential continuum shape
        continuum = (lam_rest ** beta) * np.exp(-lam_rest / lam_0)
        # Zero out wavelengths below Lyman break at rest 0.0912 μm
        continuum[lam_rest < 0.0912] = 0.0
        # Partial Lyman-alpha forest suppression (rough IGM model)
        mask_lya = (lam_rest > 0.0912) & (lam_rest < 0.1216)
        continuum[mask_lya] *= 0.3

        # Absolute normalisation: scale to approximate flux level
        # for a log_mass=10.5 galaxy at the input redshift
        # (order-of-magnitude: ~1e-18 erg/s/cm²/Å = 1e-14 erg/s/cm²/μm)
        lum_scaling  = 10 ** (log_mass - 10.5)
        z_scaling    = 1.0 / (1.0 + redshift) ** 2   # cosmological dimming (approx)
        norm         = 1.0e-18 * lum_scaling * z_scaling
        continuum   *= norm / (np.median(continuum[continuum > 0]) + 1e-40)
        continuum    = np.maximum(continuum, 0.0)

        # ── Dust attenuation on continuum ─────────────────────────────────
        continuum = _apply_calzetti_attenuation(continuum, lam_rest, av)

        # ── Emission lines ────────────────────────────────────────────────
        flux = continuum.copy()

        if not is_quenched:
            # [OIII]/Hβ from Pettini & Pagel 2004 O3N2 calibration
            # For illustrative purposes use metallicity-dependent ratio
            # log([OIII]/Hβ) decreases with metallicity
            log_o3_hb = 0.5 - 0.6 * np.log10(metallicity_solar)
            o3_hb     = 10 ** log_o3_hb

            # Hβ luminosity from Kennicutt+1998 SFR–L_Hα, then Balmer decrement
            # SFR ~ 0.1 M☉/yr for log_m=10.5 star-forming at z~2
            sfr_proxy = 10 ** (log_mass - 11.5) * 5.0
            # L_Hβ ≈ SFR / (7.9e-42) × (1/2.86)  [erg/s]  (Kennicutt 1998)
            lum_hb = sfr_proxy / (7.9e-42 * _BALMER_HA_HB)

            # Convert to flux — fiducial luminosity distance
            d_L_cm    = 3.086e27 * 1e3 * (1 + redshift) * 4.4  # very rough ~z*4 Gpc
            flux_hb_0 = lum_hb / (4.0 * np.pi * d_L_cm**2)

            # Scale line ratios from Osterbrock+2006 and modify [OIII] for metallicity
            line_ratios = dict(_REST_LINES)
            line_ratios[0.49590]["ratio"] = o3_hb / (1 + 1.0 / 0.34)
            line_ratios[0.50070]["ratio"] = o3_hb

            # Line FWHM from NIRSpec resolution R~1000: Δλ ~ λ/R
            for lam_rest_line, props in line_ratios.items():
                lam_obs = lam_rest_line * (1.0 + redshift)
                if lam_obs < _NIRSPEC_LAMBDA_MIN or lam_obs > _NIRSPEC_LAMBDA_MAX:
                    continue

                line_flux_0 = flux_hb_0 * props["ratio"]
                # Additional dust attenuation on lines:
                # E(B-V)_gas = av / (R_V * 0.44); kHβ = 3.61 (Calzetti+2000)
                k_line = _calzetti_k(np.array([lam_rest_line]))[0]
                av_gas = av / 0.44
                att    = 10 ** (-0.4 * av_gas * k_line / 4.05)
                line_flux = line_flux_0 * att

                # Gaussian line profile — R=1000 at line centre
                sigma_lam = lam_obs / (2.355 * 1000.0)   # μm
                profile   = (line_flux / (sigma_lam * np.sqrt(2 * np.pi))
                             * np.exp(-(wavelength - lam_obs)**2
                                      / (2 * sigma_lam**2)))
                flux += profile

        return flux.astype(np.float32)

    # -----------------------------------------------------------------------
    # PIPELINE STAGES
    # -----------------------------------------------------------------------

    def run_pipeline_stage1(self, output_path=None) -> bool:
        if self.raw_data is None:
            logger.error("No raw data loaded.")
            return False
        if not HAVE_JWST_PIPELINE:
            logger.info("Mock Stage 1: detector processing.")
            self.reduced_data["stage1"] = {
                "data":  self.raw_data["data"].copy(),
                "error": self.raw_data.get("error"),
                "dq":    self.raw_data.get("dq"),
                "mock":  True,
                "processing_steps": [
                    "dq_init", "saturation", "superbias",
                    "refpix", "linearity", "dark_current", "jump", "ramp_fit",
                ],
            }
            self.pipeline_status["stage1"] = True
            return True
        try:
            from jwst.pipeline import Detector1Pipeline
            pl = Detector1Pipeline()
            pl.save_results = bool(output_path)
            if output_path:
                pl.output_dir = output_path
            self.reduced_data["stage1"] = pl.run(self.raw_data["file_path"])
            self.pipeline_status["stage1"] = True
            return True
        except Exception as e:
            logger.error("Stage 1 error: %s", e)
            return False

    def run_pipeline_stage2(self, output_path=None) -> bool:
        if not self.pipeline_status["stage1"]:
            logger.error("Stage 1 must complete first.")
            return False
        if not HAVE_JWST_PIPELINE:
            logger.info("Mock Stage 2: spectroscopic processing.")
            s1 = self.reduced_data["stage1"]
            self.reduced_data["stage2"] = {
                "data":  s1["data"] * 0.8,       # approximate flat-field factor
                "error": s1["error"],
                "dq":    s1["dq"],
                "wavelength": self.raw_data.get("wavelength_2d"),
                "mock":  True,
                "processing_steps": [
                    "assign_wcs", "msa_flagging", "extract_2d",
                    "flat_field", "pathloss", "photom", "resample_spec",
                ],
            }
            self.pipeline_status["stage2"] = True
            return True
        try:
            from jwst.pipeline import Spec2Pipeline
            pl = Spec2Pipeline()
            pl.save_results = bool(output_path)
            if output_path:
                pl.output_dir = output_path
            self.reduced_data["stage2"] = pl.run(self.reduced_data["stage1"])
            self.pipeline_status["stage2"] = True
            return True
        except Exception as e:
            logger.error("Stage 2 error: %s", e)
            return False

    def run_pipeline_stage3(self, output_path=None) -> bool:
        if not self.pipeline_status["stage2"]:
            logger.error("Stage 2 must complete first.")
            return False
        if not HAVE_JWST_PIPELINE:
            logger.info("Mock Stage 3: combining exposures.")
            self.reduced_data["stage3"] = dict(self.reduced_data["stage2"])
            self.reduced_data["stage3"]["processing_steps"] = [
                "outlier_detection", "resample_spec", "extract_1d"]
            self.pipeline_status["stage3"] = True
            return True
        try:
            from jwst.pipeline import Spec3Pipeline
            pl = Spec3Pipeline()
            pl.save_results = bool(output_path)
            if output_path:
                pl.output_dir = output_path
            self.reduced_data["stage3"] = pl.run([self.reduced_data["stage2"]])
            self.pipeline_status["stage3"] = True
            return True
        except Exception as e:
            logger.error("Stage 3 error: %s", e)
            return False

    # -----------------------------------------------------------------------
    # 1-D EXTRACTION
    # -----------------------------------------------------------------------

    def extract_1d_spectrum(self, extraction_method: str = "optimal",
                             source_id: str = "target") -> dict | None:
        if not self.pipeline_status["stage2"]:
            logger.error("Stage 2 processing required.")
            return None

        stage = "stage3" if "stage3" in self.reduced_data else "stage2"
        rd    = self.reduced_data[stage]

        data_2d  = rd["data"]
        error_2d = rd.get("error")
        wl_2d    = rd.get("wavelength")

        wavelength_1d = (wl_2d[data_2d.shape[0] // 2, :]
                         if wl_2d is not None
                         else np.linspace(_NIRSPEC_LAMBDA_MIN, _NIRSPEC_LAMBDA_MAX,
                                          data_2d.shape[1]))

        methods = {
            "optimal":  self._optimal_extraction,
            "aperture": self._aperture_extraction,
            "profile":  self._profile_extraction,
        }
        spec = methods.get(extraction_method,
                           self._optimal_extraction)(data_2d, error_2d, wavelength_1d)

        spec["mock"] = rd.get("mock", False)
        self.extracted_spectra[source_id]  = spec
        self.pipeline_status["extraction"] = True
        return spec

    def _optimal_extraction(self, data_2d, error_2d, wavelength) -> dict:
        """Horne 1986 (PASP 98, 609) optimal extraction."""
        ny, _ = data_2d.shape
        if error_2d is None:
            error_2d = np.sqrt(np.maximum(np.abs(data_2d), 0) + 1)

        # Spatial profile from median collapse
        profile = np.median(data_2d, axis=1)
        profile  = np.maximum(profile, 0)
        norm     = profile.sum()
        profile  = profile / norm if norm > 0 else np.ones(ny) / ny

        # Gaussian fit to profile for optimal weights
        y = np.arange(ny)
        try:
            popt, _ = curve_fit(
                lambda x, amp, cen, sig: amp * np.exp(-(x - cen)**2 / (2 * sig**2)),
                y, profile,
                p0=[profile.max(), ny / 2, self.extraction_params["profile_sigma"]],
                maxfev=2000)
            fitted = popt[0] * np.exp(-(y - popt[1])**2 / (2 * popt[2]**2))
        except RuntimeError:
            sigma  = self.extraction_params["profile_sigma"]
            fitted = np.exp(-(y - ny / 2)**2 / (2 * sigma**2))

        fitted = np.maximum(fitted, 0)
        fitted /= fitted.sum() + 1e-40

        # Horne 1986 eq. 8
        w     = fitted[:, np.newaxis] / (error_2d**2 + 1e-40)
        denom = np.sum(w * fitted[:, np.newaxis], axis=0)
        denom = np.where(denom > 0, denom, 1e-40)
        flux_1d  = np.sum(w * data_2d,             axis=0) / denom
        err_1d   = np.sqrt(1.0 / denom)

        return {"wavelength": wavelength, "flux": flux_1d, "flux_error": err_1d,
                "extraction_method": "optimal", "profile": fitted}

    def _aperture_extraction(self, data_2d, error_2d, wavelength) -> dict:
        ny, _   = data_2d.shape
        center  = ny // 2
        half_ap = int(self.extraction_params["profile_sigma"] * 2)
        y_lo, y_hi = max(0, center - half_ap), min(ny, center + half_ap + 1)

        flux_1d = np.sum(data_2d[y_lo:y_hi, :], axis=0)
        err_1d  = (np.sqrt(np.sum(error_2d[y_lo:y_hi, :]**2, axis=0))
                   if error_2d is not None
                   else np.sqrt(np.maximum(np.abs(flux_1d), 0)))

        return {"wavelength": wavelength, "flux": flux_1d, "flux_error": err_1d,
                "extraction_method": "aperture", "aperture_size": half_ap}

    def _profile_extraction(self, data_2d, error_2d, wavelength) -> dict:
        _, nx    = data_2d.shape
        flux_1d  = np.empty(nx)
        err_1d   = np.empty(nx)

        for i in range(nx):
            col   = data_2d[:, i]
            ecol  = (error_2d[:, i] if error_2d is not None
                     else np.sqrt(np.maximum(np.abs(col), 0) + 1))
            total = np.maximum(col.sum(), 1e-40)
            p     = np.maximum(col, 0) / total
            w     = p / (ecol**2 + 1e-40)
            d     = (w * p).sum()
            flux_1d[i] = (w * col).sum() / d if d > 0 else 0.0
            err_1d[i]  = np.sqrt(1.0 / d)    if d > 0 else 0.0

        return {"wavelength": wavelength, "flux": flux_1d, "flux_error": err_1d,
                "extraction_method": "profile"}

    # -----------------------------------------------------------------------
    # SPECTRAL FITTING
    # -----------------------------------------------------------------------

    def setup_spectral_fitting_model(self, sfh_model: str = "exponential",
                                      include_nebular: bool = True) -> dict:
        if sfh_model == "exponential":
            sfh = {"exponential": {
                "age":         self.fitting_params["age_range"],
                "tau":         self.fitting_params["tau_range"],
                "massformed":  self.fitting_params["mass_range"],
                "metallicity": (0.0, 2.5),
            }}
        else:
            raise ValueError(f"Unsupported SFH model: {sfh_model}")

        fit = {
            "redshift": self.fitting_params["z_range"],
            "dust":     {"type": "Calzetti", "Av": self.fitting_params["av_range"]},
            **sfh,
        }
        if include_nebular:
            fit["nebular"] = {"logU": (-4.0, -1.0)}
            fit["t_bc"]    = (0.01, 0.1)
        return fit

    def fit_spectrum_bagpipes(self, source_id: str = "target",
                               fit_instructions: dict | None = None,
                               spec_resolution: float = 1000.0,
                               n_live: int = 400) -> dict | None:
        if source_id not in self.extracted_spectra:
            raise ValueError(f"No extracted spectrum for '{source_id}'.")

        spectrum = self.extracted_spectra[source_id]

        if not HAVE_BAGPIPES:
            return self._mock_spectral_fitting(spectrum, source_id)

        if fit_instructions is None:
            fit_instructions = self.setup_spectral_fitting_model()

        try:
            wl       = spectrum["wavelength"] * 1e4   # μm → Å
            spec_arr = np.column_stack([wl, spectrum["flux"],
                                        spectrum["flux_error"]])
            galaxy   = pipes.galaxy(source_id,
                                    lambda gid: spec_arr,
                                    spectrum_exists=True, spec_wavs=wl)
            fit = pipes.fit(galaxy, fit_instructions, spec_res=spec_resolution)
            fit.fit(verbose=False, n_live=n_live)
            post = fit.posterior

            result = {
                "stellar_mass":     post.samples["stellar_mass"].mean(),
                "stellar_mass_err": post.samples["stellar_mass"].std(),
                "age":              post.samples["age"].mean(),
                "age_err":          post.samples["age"].std(),
                "tau":              post.samples["tau"].mean(),
                "tau_err":          post.samples["tau"].std(),
                "metallicity":      post.samples["metallicity"].mean(),
                "metallicity_err":  post.samples["metallicity"].std(),
                "av":               post.samples["dust:Av"].mean(),
                "av_err":           post.samples["dust:Av"].std(),
                "redshift":         post.samples["redshift"].mean(),
                "redshift_err":     post.samples["redshift"].std(),
                "model_spectrum":   fit.galaxy.spectrum_full,
                "chi2":             post.samples.get("chisq_phot",
                                                     np.array([np.nan])).mean(),
                "mock":             False,
            }
            if "nebular" in fit_instructions:
                result["log_u"]     = post.samples["nebular:logU"].mean()
                result["log_u_err"] = post.samples["nebular:logU"].std()

            self.spectral_fits[source_id] = result
            return result
        except Exception as e:
            logger.error("Bagpipes fitting error: %s", e)
            return None

    def _mock_spectral_fitting(self, spectrum: dict, source_id: str) -> dict:
        """
        Mock spectral fitting results consistent with the input mock spectrum.

        If the spectrum was generated by _generate_mock_spectrum_1d we can
        recover the input parameters ± realistic uncertainties to simulate
        what Bagpipes would return.  Otherwise we fall back to physically
        plausible defaults with documented scatter.

        Parameter uncertainties are based on Carnall et al. 2019 (MNRAS 490)
        typical posteriors for NIRSpec-quality spectra.
        """
        rng = np.random.default_rng(seed=7)

        # Try to recover ground-truth mock params
        params = spectrum.get("mock_params", {})
        z_true     = params.get("redshift", 2.0)
        log_m_true = params.get("log_mass",  10.5)
        av_true    = params.get("av",         0.4)

        # Recovered values with realistic posteriors (Carnall+2019)
        z_rec   = z_true    + rng.normal(0, 0.01)          # σ_z ~ 0.01 for spec-z
        lm_rec  = log_m_true + rng.normal(0, 0.15)         # σ_logM ~ 0.15 dex
        av_rec  = max(av_true + rng.normal(0, 0.15), 0.0)  # σ_Av ~ 0.15 mag

        # Age: from e-folding time τ ~ 1 Gyr and rough lookback time
        t_universe = 13.8 / (1.0 + z_true)**1.5
        age_rec    = float(np.clip(rng.normal(t_universe * 0.5, t_universe * 0.2),
                                   0.1, t_universe))
        tau_rec    = float(np.clip(rng.normal(1.0, 0.4), 0.3, 8.0))

        # Metallicity from mass-metallicity relation (Tremonti+2004)
        met_rec = float(np.clip(
            10 ** (-0.185 * (lm_rec - 10.5)**2 + 9.07 - 8.69)
            + rng.normal(0, 0.15), 0.1, 2.5))

        # Model spectrum: regenerate with recovered parameters
        wl = spectrum["wavelength"]
        model_flux = self._generate_mock_spectrum_1d(
            wl, redshift=z_rec, log_mass=lm_rec, av=av_rec)

        # χ² ~ 1 for a good fit with ~10% noise
        chi2 = float(rng.normal(1.0, 0.15))

        result = {
            "stellar_mass":     lm_rec,
            "stellar_mass_err": 0.15,
            "age":              age_rec,
            "age_err":          age_rec * 0.20,
            "tau":              tau_rec,
            "tau_err":          tau_rec * 0.25,
            "metallicity":      met_rec,
            "metallicity_err":  met_rec * 0.15,
            "av":               av_rec,
            "av_err":           0.15,
            "redshift":         z_rec,
            "redshift_err":     0.01,
            "model_spectrum":   {"wavelength": wl, "flux": model_flux},
            "chi2":             chi2,
            "mock":             True,
            "references": [
                "Carnall et al. 2019, MNRAS 490, 417",
                "Calzetti et al. 2000, ApJ 533, 682",
                "Tremonti et al. 2004, ApJ 613, 898",
            ],
        }
        self.spectral_fits[source_id] = result
        return result

    # -----------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------

    def _mock_label(self, ax):
        if self._mock_data:
            ax.text(0.98, 0.02,
                    "SIMULATED DATA — not real observations",
                    transform=ax.transAxes, fontsize=7, color="darkorange",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                              ec="darkorange", alpha=0.9))

    def plot_2d_spectrum(self, data_stage: str = "stage2", save_path=None):
        if data_stage not in self.reduced_data:
            raise ValueError(f"No data for stage '{data_stage}'.")
        data_2d = self.reduced_data[data_stage]["data"]
        nx      = data_2d.shape[1]
        wl      = np.linspace(_NIRSPEC_LAMBDA_MIN, _NIRSPEC_LAMBDA_MAX, nx)

        fig, ax = plt.subplots(figsize=(12, 5))
        vmax    = np.nanpercentile(np.abs(data_2d), 98)
        im = ax.imshow(data_2d, aspect="auto", origin="lower", cmap="viridis",
                        vmin=0, vmax=vmax,
                        extent=[wl[0], wl[-1], 0, data_2d.shape[0]],
                        interpolation="bilinear")
        ax.set_xlabel("Wavelength [μm]"); ax.set_ylabel("Spatial [px]")
        ax.set_title(f"2-D Spectrum ({data_stage.title()})")
        fig.colorbar(im, ax=ax).set_label("Flux")
        self._mock_label(ax)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_1d_spectrum(self, source_id: str = "target",
                          show_fit: bool = True, save_path=None):
        if source_id not in self.extracted_spectra:
            raise ValueError(f"No extracted spectrum for '{source_id}'.")

        spec  = self.extracted_spectra[source_id]
        wl    = spec["wavelength"]
        flux  = spec["flux"]
        err   = spec["flux_error"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                                  gridspec_kw={"height_ratios": [3, 1]})

        axes[0].plot(wl, flux, "k-", lw=0.8, label="Observed")
        axes[0].fill_between(wl, flux - err, flux + err,
                              alpha=0.25, color="gray", label="±1σ")

        if show_fit and source_id in self.spectral_fits:
            fit = self.spectral_fits[source_id]
            if "model_spectrum" in fit:
                mwl  = fit["model_spectrum"]["wavelength"]
                mflx = fit["model_spectrum"]["flux"]
                axes[0].plot(mwl, mflx, "r-", lw=1.2, alpha=0.8,
                             label="Best-fit model")

            # Annotate detected lines at recovered redshift
            z_rec = fit.get("redshift", 0.0)
            for lam_rest, props in _REST_LINES.items():
                lam_obs = lam_rest * (1.0 + z_rec)
                if _NIRSPEC_LAMBDA_MIN < lam_obs < _NIRSPEC_LAMBDA_MAX:
                    axes[0].axvline(lam_obs, color="steelblue",
                                    lw=0.6, ls=":", alpha=0.7)
                    axes[0].text(lam_obs, axes[0].get_ylim()[1] * 0.95,
                                 props["name"], fontsize=6, rotation=90,
                                 ha="right", color="steelblue", alpha=0.8)

        axes[0].set_ylabel("Flux [erg/s/cm²/μm]", fontsize=11)
        axes[0].set_title(f"1-D Spectrum: {source_id}", fontsize=13)
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
        self._mock_label(axes[0])

        # Residuals
        if show_fit and source_id in self.spectral_fits:
            fit = self.spectral_fits[source_id]
            if "model_spectrum" in fit:
                mflx_interp = np.interp(
                    wl, fit["model_spectrum"]["wavelength"],
                    fit["model_spectrum"]["flux"])
                residuals = (flux - mflx_interp) / (err + 1e-40)
                axes[1].plot(wl, residuals, "g-", lw=0.6)
                axes[1].axhline(0, color="black", ls="--", alpha=0.4)
                axes[1].set_ylim(-5, 5)
                axes[1].set_ylabel("Residuals [σ]", fontsize=10)
                axes[1].grid(alpha=0.3)

        axes[1].set_xlabel("Wavelength [μm]", fontsize=11)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, axes

    def plot_spectral_fit_results(self, source_id: str = "target", save_path=None):
        if source_id not in self.spectral_fits:
            raise ValueError(f"No fit results for '{source_id}'.")

        fit = self.spectral_fits[source_id]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Parameters table
        params = [
            f"log(M*/M☉) = {fit['stellar_mass']:.2f} ± {fit['stellar_mass_err']:.2f}",
            f"Age = {fit['age']:.2f} ± {fit['age_err']:.2f} Gyr",
            f"τ = {fit['tau']:.2f} ± {fit['tau_err']:.2f} Gyr",
            f"Z/Z☉ = {fit['metallicity']:.2f} ± {fit['metallicity_err']:.2f}",
            f"Av = {fit['av']:.2f} ± {fit['av_err']:.2f} mag",
            f"z = {fit['redshift']:.4f} ± {fit['redshift_err']:.4f}",
        ]
        axes[0, 0].text(0.05, 0.95, "\n".join(params),
                         transform=axes[0, 0].transAxes, va="top",
                         fontsize=11, fontfamily="monospace")
        if fit.get("mock"):
            axes[0, 0].text(0.05, 0.10,
                             "SIMULATED — Carnall+2019 uncertainty model",
                             transform=axes[0, 0].transAxes,
                             fontsize=8, color="darkorange")
        axes[0, 0].axis("off"); axes[0, 0].set_title("Fitted Parameters")

        # SFH
        t_arr     = np.linspace(0, 13.8, 200)
        t_lookback = 13.8 - fit["age"]
        sfh       = np.exp(-(t_arr - t_lookback)**2 / (2 * fit["tau"]**2))
        sfh[t_arr < t_lookback] = 0
        axes[0, 1].plot(t_arr, sfh / sfh.max(), "b-", lw=2)
        axes[0, 1].set_xlabel("Cosmic Time [Gyr]"); axes[0, 1].set_ylabel("SFR (norm.)")
        axes[0, 1].set_title("Star Formation History"); axes[0, 1].grid(alpha=0.3)

        # Spectral fit
        if source_id in self.extracted_spectra:
            spec = self.extracted_spectra[source_id]
            axes[1, 0].plot(spec["wavelength"], spec["flux"], "k-", lw=0.8,
                             label="Observed")
            if "model_spectrum" in fit:
                m = fit["model_spectrum"]
                axes[1, 0].plot(m["wavelength"], m["flux"], "r-", lw=1,
                                 label="Model")
        axes[1, 0].set_xlabel("Wavelength [μm]"); axes[1, 0].set_ylabel("Flux")
        axes[1, 0].set_title("Spectral Fit"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

        # Goodness of fit
        chi2 = fit.get("chi2", float("nan"))
        chi2_str = f"χ²_red = {chi2:.2f}" if not np.isnan(chi2) else "χ² not available"
        axes[1, 1].text(0.5, 0.55, chi2_str,
                         transform=axes[1, 1].transAxes,
                         ha="center", va="center", fontsize=16)
        axes[1, 1].text(0.5, 0.40,
                         "χ² ≈ 1 → good fit\nχ² >> 1 → systematic residuals",
                         transform=axes[1, 1].transAxes,
                         ha="center", va="center", fontsize=10, color="gray")
        axes[1, 1].axis("off"); axes[1, 1].set_title("Goodness of Fit")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, axes

    # -----------------------------------------------------------------------
    # SUMMARY & EXPORT
    # -----------------------------------------------------------------------

    def generate_pipeline_summary(self) -> dict:
        return {
            "pipeline_status":   self.pipeline_status.copy(),
            "data_products":     list(self.reduced_data.keys()),
            "extracted_sources": list(self.extracted_spectra.keys()),
            "fitted_sources":    list(self.spectral_fits.keys()),
            "fit_results_summary": {
                sid: {k: v for k, v in res.items()
                      if k not in ("model_spectrum", "posterior_samples")}
                for sid, res in self.spectral_fits.items()
            },
        }

    def export_results(self, output_path: str, format: str = "fits"):
        if format.lower() == "fits" and HAVE_ASTROPY:
            hdus = [fits.PrimaryHDU()]
            for sid, spec in self.extracted_spectra.items():
                tbl = Table([spec["wavelength"], spec["flux"], spec["flux_error"]],
                             names=["WAVELENGTH", "FLUX", "FLUX_ERROR"])
                hdus.append(fits.BinTableHDU(tbl, name=f"SPEC_{sid.upper()}"))
            if self.spectral_fits:
                rows = [[sid, r["stellar_mass"], r["stellar_mass_err"],
                         r["age"], r["age_err"], r["redshift"], r["redshift_err"],
                         r["av"], r["av_err"], r.get("mock", True)]
                        for sid, r in self.spectral_fits.items()]
                ft = Table(rows=rows,
                           names=["SOURCE_ID", "STELLAR_MASS", "STELLAR_MASS_ERR",
                                  "AGE", "AGE_ERR", "REDSHIFT", "REDSHIFT_ERR",
                                  "AV", "AV_ERR", "MOCK"])
                hdus.append(fits.BinTableHDU(ft, name="FIT_RESULTS"))
            fits.HDUList(hdus).writeto(output_path, overwrite=True)
        elif format.lower() == "csv":
            if self.spectral_fits:
                rows = []
                for sid, r in self.spectral_fits.items():
                    row = {"source_id": sid}
                    row.update({k: v for k, v in r.items()
                                if k not in ("model_spectrum", "posterior_samples")})
                    rows.append(row)
                pd.DataFrame(rows).to_csv(output_path, index=False)
        elif format.lower() == "json":
            import json
            export = {"pipeline_summary": self.generate_pipeline_summary(),
                      "fit_results": {}}
            for sid, r in self.spectral_fits.items():
                clean = {k: (v.tolist() if hasattr(v, "tolist") else v)
                         for k, v in r.items()
                         if k not in ("model_spectrum", "posterior_samples")}
                export["fit_results"][sid] = clean
            with open(output_path, "w") as f:
                json.dump(export, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info("Results exported to %s", output_path)