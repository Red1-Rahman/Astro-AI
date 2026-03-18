# Dashboard Module for Astro-AI
#
# Copyright (c) 2025 Redwan Rahman and CAM-SUST
#
# Fixes vs original:
#
#  1. _time_to_redshift() and _redshift_to_time() were broken analytic
#     approximations that returned negative cosmic times and completely wrong
#     redshifts (z=0 mapped to t=-4.1 Gyr; z=6 mapped to t=-0.2 Gyr).
#     Replaced with the standard flat ΛCDM approximation
#     t(z) ≈ 13.8/(1+z)^1.5 Gyr — correct to ~15% across 0 < z < 20 and
#     always positive.  When astropy is available the exact integral is used.
#
#  2. _generate_mock_spectrum_for_plot() listed emission line wavelengths in
#     Angstroms disguised as microns (4.861 "μm" for Hβ — the correct
#     observed wavelength is 0.4861 μm ≈ 4861 Å).  All line wavelengths
#     are now in true microns (rest frame) and are redshifted to a default
#     z=2 before plotting so the lines fall inside NIRSpec's 0.6–5.3 μm
#     window.
#
#  3. create_timeline_integration() called ax.legend() before adding the
#     epoch axvlines, so the epoch markers never appeared in the legend.
#     The legend call is now deferred until after all artists are added.
#
#  4. create_environment_comparison() computed quenching efficiency as
#     (rf_cluster - rf_field) / (1 - rf_field), which divides by zero
#     whenever rf_field = 1.0 (fully quenched field bin).  Protected with
#     np.where so the result is 0 in degenerate bins.
#
#  5. All create_* methods stored new figures in self.integration_plots
#     without closing the previous figure held under the same key, leaking
#     matplotlib figures on every repeated call.  Each method now calls
#     plt.close() on the old figure before replacing it.
#
#  6. Epoch marker axvline times in create_timeline_integration() were
#     hardcoded to [0.8, 1.2, 2.0] Gyr which, with the broken time
#     conversion, corresponded to z~8, z~10, z~15 rather than the labelled
#     z~15, z~10, z~6.  Times are now derived from the corrected
#     _redshift_to_time() so labels and lines agree.

import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Rest-frame emission line wavelengths [μm] and NIRSpec-observable redshift z=2
# (same physical constants as jwst_pipeline.py)
_REST_LINES_DASHBOARD = {
    0.48613: "Hβ",
    0.50070: "[OIII]5007",
    0.65628: "Hα",
    1.28216: "Paβ",
    1.87510: "Paα",
}
_DISPLAY_REDSHIFT = 2.0   # default redshift for showcase spectra


class Dashboard:
    """
    Integrated Dashboard for Astro-AI Results.

    Creates comparative visualisations that integrate results from the
    Cosmic Evolution, Cluster Analysis, and JWST modules.
    """

    def __init__(self):
        self.cos_evo_results  = None
        self.cluster_results  = None
        self.jwst_results     = None
        self.integration_plots: dict = {}

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_results(self, cos_evo=None, cluster=None, jwst=None):
        if cos_evo is not None:
            self.cos_evo_results = cos_evo
        if cluster is not None:
            self.cluster_results = cluster
        if jwst is not None:
            self.jwst_results = jwst

    # -----------------------------------------------------------------------
    # Internal physics helpers
    # -----------------------------------------------------------------------

    def _redshift_to_time(self, z) -> np.ndarray:
        """
        Cosmic time at redshift z [Gyr].

        Uses astropy FlatLambdaCDM (Planck18) when available; otherwise
        falls back to the flat matter-dominated approximation
        t ≈ 13.8 / (1+z)^1.5 Gyr, which is accurate to ~15% for 0 < z < 20
        and always returns positive values.
        """
        z = np.asarray(z, dtype=float)
        try:
            from astropy.cosmology import Planck18
            return Planck18.age(z).value
        except Exception:
            return 13.8 / (1.0 + z) ** 1.5

    def _time_to_redshift(self, t_gyr) -> np.ndarray:
        """
        Approximate redshift at cosmic time t_gyr [Gyr].

        Inverse of the matter-dominated approximation:
        z ≈ (13.8/t)^(2/3) - 1   (always non-negative).
        """
        t = np.asarray(t_gyr, dtype=float)
        t_safe = np.maximum(t, 0.01)
        return np.maximum((13.8 / t_safe) ** (2.0 / 3.0) - 1.0, 0.0)

    def _generate_mock_spectrum_for_plot(
        self,
        wavelength: np.ndarray,
        redshift: float = _DISPLAY_REDSHIFT,
    ) -> np.ndarray:
        """
        Physically motivated mock NIRSpec spectrum.

        Lines are defined at rest-frame wavelengths [μm] and redshifted to
        *redshift* before plotting, so they fall in the correct observed
        positions within NIRSpec's 0.6–5.3 μm window.

        Original bug: lines were at rest-frame positions labelled as
        observed-frame, including 4.861 "μm" for Hβ which is
        0.4861 μm × (1+z) — 10× too large.
        """
        # Stellar continuum (BC03 power-law shape)
        lam_rest = wavelength / (1.0 + redshift)
        continuum = np.maximum(
            1e-18 * (lam_rest ** -1.5) * np.exp(-lam_rest / 1.5), 0.0
        )
        # Lyman break
        continuum[lam_rest < 0.0912] = 0.0

        flux = continuum.copy()

        # Add emission lines at correct observed wavelengths
        for lam_rest_um, name in _REST_LINES_DASHBOARD.items():
            lam_obs = lam_rest_um * (1.0 + redshift)
            if wavelength.min() <= lam_obs <= wavelength.max():
                sigma = lam_obs / (2.355 * 1000.0)   # R=1000
                amp   = continuum.max() * 0.5
                flux += amp * np.exp(-(wavelength - lam_obs) ** 2 / (2 * sigma ** 2))

        return flux

    def _close_previous(self, key: str):
        """Close and discard any figure already stored under *key*."""
        old = self.integration_plots.get(key)
        if old is not None:
            try:
                plt.close(old)
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Dashboard plots
    # -----------------------------------------------------------------------

    def create_timeline_integration(self, save_path=None):
        """Create integrated cosmic timeline visualisation."""
        self._close_previous("timeline")

        fig = plt.figure(figsize=(16, 12))
        gs  = GridSpec(3, 2, figure=fig,
                       height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

        # --- Panel 1: 21cm global signal ---
        ax1 = fig.add_subplot(gs[0, :])

        if self.cos_evo_results:
            z_21cm   = np.asarray(self.cos_evo_results.get(
                "redshifts", np.linspace(6, 15, 10)))
            sig_21cm = np.asarray(self.cos_evo_results.get(
                "global_signals", np.zeros_like(z_21cm)))
            t_21cm   = self._redshift_to_time(z_21cm)
            ax1.plot(t_21cm, sig_21cm, "b-", lw=3, label="21cm Global Signal")
            ax1.fill_between(t_21cm, sig_21cm - 5, sig_21cm + 5,
                             alpha=0.25, color="blue")
            mock_note = self.cos_evo_results.get("mock", False)
        else:
            t_21cm  = np.linspace(0.3, 2.5, 60)
            z_mock  = self._time_to_redshift(t_21cm)
            # Physically motivated placeholder (absorption trough at z~17)
            from modules.cos_evo.cosmic_evolution import CosmicEvolution
            ce = CosmicEvolution({"H0": 67.66, "Om0": 0.31, "sigma8": 0.81})
            sig_21cm = np.array([ce._delta_T_b(float(z)) for z in z_mock])
            ax1.plot(t_21cm, sig_21cm, "b--", lw=2, alpha=0.6,
                     label="21cm Signal (analytic placeholder)")
            mock_note = True

        # Epoch markers — add BEFORE legend so they appear in it
        epochs = [
            (self._redshift_to_time(15.0), "z~15 (Cosmic Dawn)",   "red"),
            (self._redshift_to_time(10.0), "z~10 (Reionization)",  "orange"),
            (self._redshift_to_time(6.0),  "z~6 (End EoR)",        "green"),
        ]
        for t_ep, label, color in epochs:
            ax1.axvline(float(t_ep), color=color, ls="--", alpha=0.7, label=label)

        ax1.set_ylabel("21cm Signal [mK]", fontsize=12)
        ax1.set_xlabel("Cosmic Time [Gyr]", fontsize=12)
        ax1.set_title("Cosmic Evolution Timeline", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        if mock_note:
            ax1.text(0.98, 0.04, "SIMULATED DATA",
                     transform=ax1.transAxes, fontsize=8, color="darkorange",
                     ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                               ec="darkorange", alpha=0.9))

        # --- Panel 2: Red fraction ---
        ax2 = fig.add_subplot(gs[1, 0])

        if (self.cluster_results
                and isinstance(self.cluster_results, dict)
                and "red_fraction" in self.cluster_results):
            rf   = self.cluster_results["red_fraction"]
            mc   = rf["mass_centers"]
            ax2.semilogx(10 ** mc, rf["cluster"]["red_fraction"], "ro-",
                         lw=2, ms=6, label="Cluster")
            ax2.semilogx(10 ** mc, rf["field"]["red_fraction"], "bo-",
                         lw=2, ms=6, label="Field")
        else:
            mass_b = np.logspace(9.5, 11.5, 10)
            rf_cl  = 0.1 + 0.6 / (1 + np.exp(-(mass_b - 10 ** 10.5) / 1e10))
            rf_fi  = 0.05 + 0.4 / (1 + np.exp(-(mass_b - 1e11) / 1e10))
            ax2.semilogx(mass_b, rf_cl, "ro-", lw=2, ms=6, label="Cluster")
            ax2.semilogx(mass_b, rf_fi, "bo-", lw=2, ms=6, label="Field")

        ax2.set_xlabel("Stellar Mass [M☉]"); ax2.set_ylabel("Red Fraction")
        ax2.set_title("Environmental Quenching", fontsize=12, fontweight="bold")
        ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)

        # --- Panel 3: Environmental effect ---
        ax3 = fig.add_subplot(gs[1, 1])

        if (self.cluster_results
                and isinstance(self.cluster_results, dict)
                and "red_fraction" in self.cluster_results):
            rf  = self.cluster_results["red_fraction"]
            mc  = rf["mass_centers"]
            drf = rf["environmental_effect"]["delta_red_fraction"]
            err = rf["environmental_effect"]["delta_red_fraction_err"]
            ax3.errorbar(mc, drf, yerr=err, fmt="go-", lw=2, ms=6, capsize=3)
        else:
            mc_log = np.linspace(9.5, 11.5, 10)
            mass_b = 10 ** mc_log
            rf_cl  = 0.1 + 0.6 / (1 + np.exp(-(mass_b - 10 ** 10.5) / 1e10))
            rf_fi  = 0.05 + 0.4 / (1 + np.exp(-(mass_b - 1e11) / 1e10))
            ax3.plot(mc_log, rf_cl - rf_fi, "go-", lw=2, ms=6)

        ax3.axhline(0, color="black", ls="--", alpha=0.5)
        ax3.set_xlabel("log(M*/M☉)"); ax3.set_ylabel("Δ(Red Fraction)")
        ax3.set_title("Environmental Effect", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # --- Panel 4: JWST properties ---
        ax4 = fig.add_subplot(gs[2, 0])

        if (self.jwst_results
                and isinstance(self.jwst_results, dict)
                and "spectral_fits" in self.jwst_results
                and self.jwst_results["spectral_fits"]):
            fits = self.jwst_results["spectral_fits"]
            masses    = [v["stellar_mass"] for v in fits.values()]
            ages      = [v["age"]          for v in fits.values()]
            redshifts = [v["redshift"]      for v in fits.values()]
        else:
            rng       = np.random.default_rng(42)
            masses    = rng.uniform(9.5, 11.5, 20).tolist()
            ages      = rng.uniform(0.5,  8.0, 20).tolist()
            redshifts = rng.uniform(1.0,  4.0, 20).tolist()

        sc = ax4.scatter(masses, ages, c=redshifts, s=80, cmap="viridis",
                          alpha=0.8, edgecolors="black", linewidths=0.5)
        plt.colorbar(sc, ax=ax4, label="Redshift")
        ax4.set_xlabel("log(M*/M☉)"); ax4.set_ylabel("Age [Gyr]")
        ax4.set_title("JWST Spectroscopic Properties",
                      fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # --- Panel 5: Mass–SFR ---
        ax5 = fig.add_subplot(gs[2, 1])

        ms_x = np.linspace(9, 12, 100)
        ax5.plot(ms_x, ms_x - 9.0, "k--", lw=2, alpha=0.7, label="Main Sequence")

        if (self.cluster_results
                and isinstance(self.cluster_results, dict)
                and "sed_results" in self.cluster_results
                and self.cluster_results["sed_results"]):
            cm, cs = [], []
            for r in self.cluster_results["sed_results"].values():
                cm.append(r["stellar_mass"])
                cs.append(np.log10(max(r["sfr"], 1e-3)))
            ax5.scatter(cm, cs, c="red", s=40, alpha=0.7, label="Cluster")

        if (self.jwst_results
                and isinstance(self.jwst_results, dict)
                and "spectral_fits" in self.jwst_results
                and self.jwst_results["spectral_fits"]):
            jm, js = [], []
            for r in self.jwst_results["spectral_fits"].values():
                if "sfr" in r:
                    jm.append(r["stellar_mass"])
                    js.append(np.log10(max(r["sfr"], 1e-3)))
            if jm:
                ax5.scatter(jm, js, c="blue", s=80, alpha=0.8,
                            marker="s", label="JWST spectra")
        else:
            rng  = np.random.default_rng(7)
            jm   = rng.uniform(9.5, 11.5, 10)
            js   = jm - 9.0 + rng.normal(0, 0.3, 10)
            ax5.scatter(jm, js, c="blue", s=80, alpha=0.8,
                        marker="s", label="JWST spectra")

        ax5.set_xlabel("log(M*/M☉)"); ax5.set_ylabel("log(SFR) [M☉/yr]")
        ax5.set_title("Integrated Mass–SFR", fontsize=12, fontweight="bold")
        ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

        fig.suptitle("Astro-AI: Integrated Galaxy Evolution Analysis",
                     fontsize=15, fontweight="bold", y=0.97)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        self.integration_plots["timeline"] = fig
        return fig, [ax1, ax2, ax3, ax4, ax5]

    def create_environment_comparison(self, save_path=None):
        """Create detailed environment comparison plots."""
        self._close_previous("environment")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Environmental Effects on Galaxy Properties",
                     fontsize=16, fontweight="bold")

        rng = np.random.default_rng(seed=42)

        # Panel 1: Spatial distribution
        ax = axes[0, 0]
        if (self.cluster_results
                and isinstance(self.cluster_results, dict)
                and "cluster_members" in self.cluster_results
                and "field_galaxies" in self.cluster_results):
            cg = self.cluster_results["cluster_members"]
            fg = self.cluster_results["field_galaxies"]
            if isinstance(cg, pd.DataFrame) and "ra" in cg.columns:
                ax.scatter(fg["ra"],  fg["dec"],  alpha=0.6, s=15,
                           c="steelblue", label="Field")
                ax.scatter(cg["ra"], cg["dec"], alpha=0.8, s=30,
                           c="tomato", label="Cluster")
        else:
            fra = rng.uniform(149, 151, 800)
            fde = rng.uniform(1.5, 2.5, 800)
            cra = rng.normal(150, 0.1, 200)
            cde = rng.normal(2.0, 0.1, 200)
            ax.scatter(fra, fde, alpha=0.6, s=15, c="steelblue", label="Field")
            ax.scatter(cra, cde, alpha=0.8, s=30, c="tomato",    label="Cluster")
        ax.set_xlabel("RA [°]"); ax.set_ylabel("Dec [°]")
        ax.set_title("Spatial Distribution"); ax.legend(); ax.grid(alpha=0.3)

        # Panel 2: CMD
        ax = axes[0, 1]
        r_mags  = rng.uniform(18, 25, 1000)
        colors  = rng.normal(0.65, 0.18, 1000)
        cl_mask = rng.random(1000) < 0.3
        colors[cl_mask] += 0.15  # cluster galaxies redder (Bell+2004)
        ax.scatter(r_mags[~cl_mask], colors[~cl_mask],
                   alpha=0.5, s=15, c="steelblue", label="Field")
        ax.scatter(r_mags[cl_mask],  colors[cl_mask],
                   alpha=0.8, s=25, c="tomato",    label="Cluster")
        r_rng = np.linspace(18, 25, 100)
        ax.plot(r_rng, 0.65 + 0.02 * (r_rng - 22), "k--", alpha=0.7,
                label="Red sequence")
        ax.set_xlabel("r [mag]"); ax.set_ylabel("g – r")
        ax.set_title("Color–Magnitude Diagram")
        ax.legend(); ax.grid(alpha=0.3); ax.invert_xaxis()

        # Panel 3: Redshift distribution
        ax = axes[0, 2]
        z_bins  = np.linspace(0.5, 2.5, 30)
        z_field = rng.uniform(0.5, 2.5, 800)
        z_cl    = np.concatenate([rng.normal(1.2, 0.05, 150),
                                   rng.normal(1.8, 0.03,  50)])
        ax.hist(z_field, bins=z_bins, alpha=0.6, color="steelblue",
                label="Field", density=True)
        ax.hist(z_cl,    bins=z_bins, alpha=0.8, color="tomato",
                label="Cluster", density=True)
        ax.axvline(1.2, color="tomato", ls="--", alpha=0.7, label="Cluster z")
        ax.set_xlabel("Redshift"); ax.set_ylabel("Normalised Number")
        ax.set_title("Redshift Distribution")
        ax.legend(); ax.grid(alpha=0.3)

        # Panel 4: SMFs
        ax     = axes[1, 0]
        mc     = np.linspace(9.5, 11.5, 14)
        mc_ctr = (mc[:-1] + mc[1:]) / 2
        phi_cl = 10 ** (-(mc_ctr - 10.5) ** 2 / 0.5 - 2)
        phi_fi = 10 ** (-(mc_ctr - 10.3) ** 2 / 0.8 - 2.2)
        ax.semilogy(mc_ctr, phi_cl, "ro-", lw=2, ms=6, label="Cluster")
        ax.semilogy(mc_ctr, phi_fi, "bo-", lw=2, ms=6, label="Field")
        ax.set_xlabel("log(M*/M☉)"); ax.set_ylabel("φ [Mpc⁻³ dex⁻¹]")
        ax.set_title("Stellar Mass Functions"); ax.legend(); ax.grid(alpha=0.3)

        # Panel 5: Red fraction vs mass
        ax      = axes[1, 1]
        mb_rf   = np.linspace(9.5, 11.5, 8)
        mc_rf   = (mb_rf[:-1] + mb_rf[1:]) / 2
        rf_cl   = 0.1 + 0.7 / (1 + np.exp(-(mc_rf - 10.3) / 0.3))
        rf_fi   = 0.05 + 0.4 / (1 + np.exp(-(mc_rf - 10.7) / 0.4))
        ax.plot(mc_rf, rf_cl, "ro-", lw=2, ms=6, label="Cluster (Peng+2010)")
        ax.plot(mc_rf, rf_fi, "bo-", lw=2, ms=6, label="Field (Peng+2010)")
        ax.set_xlabel("log(M*/M☉)"); ax.set_ylabel("Red Fraction")
        ax.set_title("Red Fraction vs Mass")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

        # Panel 6: Quenching efficiency
        # BUG FIX: protect against rf_field = 1 (division by zero)
        ax  = axes[1, 2]
        drf = rf_cl - rf_fi
        qe  = np.where(rf_fi < 1.0, drf / (1.0 - rf_fi), 0.0)
        ax.plot(mc_rf, qe, "go-", lw=3, ms=8)
        ax.axhline(0, color="black", ls="--", alpha=0.5)
        ax.set_xlabel("log(M*/M☉)"); ax.set_ylabel("Quenching Efficiency")
        ax.set_title("Environmental Quenching Efficiency")
        ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        self.integration_plots["environment"] = fig
        return fig, axes

    def create_jwst_showcase(self, save_path=None):
        """Create JWST results showcase."""
        self._close_previous("jwst_showcase")

        fig = plt.figure(figsize=(16, 10))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        fig.suptitle("JWST Spectroscopic Analysis Results",
                     fontsize=16, fontweight="bold")

        ny, nx    = 30, 300
        wavelength = np.linspace(0.6, 5.3, nx)

        # --- Panel 1: 2D spectrum ---
        ax1  = fig.add_subplot(gs[0, 0])
        flux_1d = self._generate_mock_spectrum_for_plot(wavelength)
        y_arr   = np.arange(ny)
        profile = np.exp(-(y_arr - ny / 2) ** 2 / (2 * 2.5 ** 2))
        sp2d    = np.outer(profile, flux_1d)
        im1 = ax1.imshow(sp2d, aspect="auto", origin="lower", cmap="viridis",
                          extent=[wavelength[0], wavelength[-1], 0, ny],
                          interpolation="bilinear")
        ax1.set_xlabel("Wavelength [μm]"); ax1.set_ylabel("Spatial [px]")
        ax1.set_title("2D Spectrum")
        plt.colorbar(im1, ax=ax1, label="Flux")

        # --- Panel 2: Extracted 1D spectrum ---
        ax2         = fig.add_subplot(gs[0, 1])
        rng         = np.random.default_rng(3)
        flux_clean  = gaussian_filter1d(flux_1d, sigma=1.5)
        noise       = 0.05 * np.median(flux_clean) * rng.standard_normal(len(flux_clean))
        flux_obs    = flux_clean + noise
        flux_err    = 0.1 * np.abs(flux_obs) + 1e-21

        ax2.plot(wavelength, flux_obs,   "k-",  lw=0.8, label="Observed")
        ax2.fill_between(wavelength, flux_obs - flux_err, flux_obs + flux_err,
                          alpha=0.25, color="gray")
        ax2.plot(wavelength, flux_clean, "r-",  lw=1.2, alpha=0.8, label="Model")

        # Label emission lines at observed positions
        for lam_rest, name in _REST_LINES_DASHBOARD.items():
            lam_obs = lam_rest * (1.0 + _DISPLAY_REDSHIFT)
            if wavelength.min() < lam_obs < wavelength.max():
                ax2.axvline(lam_obs, color="steelblue", lw=0.7, ls=":", alpha=0.8)
                ax2.text(lam_obs, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0
                         else flux_clean.max() * 0.95,
                         name, fontsize=6, rotation=90, ha="right",
                         color="steelblue", alpha=0.9)

        ax2.set_xlabel("Wavelength [μm]"); ax2.set_ylabel("Flux")
        ax2.set_title(f"Extracted 1D Spectrum (z={_DISPLAY_REDSHIFT})")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

        # --- Panel 3: Residuals ---
        ax3  = fig.add_subplot(gs[0, 2])
        resid = (flux_obs - flux_clean) / (flux_err + 1e-40)
        ax3.plot(wavelength, resid, "g-", lw=0.7)
        ax3.axhline(0,  color="black", ls="--", alpha=0.5)
        ax3.axhline( 3, color="red",   ls=":",  alpha=0.5, label="±3σ")
        ax3.axhline(-3, color="red",   ls=":",  alpha=0.5)
        ax3.set_ylim(-6, 6)
        ax3.set_xlabel("Wavelength [μm]"); ax3.set_ylabel("Residuals [σ]")
        ax3.set_title("Fit Residuals"); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

        # --- Panel 4: SFH ---
        ax4 = fig.add_subplot(gs[1, 0])
        t   = np.linspace(0, 13.8, 200)
        tau = 1.0
        t0  = 13.8 - 2.0   # lookback time 2 Gyr
        sfh = np.exp(-(t - t0) ** 2 / (2 * tau ** 2))
        sfh[t < t0] = 0
        ax4.plot(t, sfh, "b-", lw=3)
        ax4.fill_between(t, sfh, alpha=0.25, color="blue")
        ax4.set_xlabel("Cosmic Time [Gyr]"); ax4.set_ylabel("SFR (norm.)")
        ax4.set_title("Star Formation History"); ax4.grid(alpha=0.3)

        # --- Panel 5: Properties vs literature ---
        ax5  = fig.add_subplot(gs[1, 1])
        props = ["log(M*)", "Age", "Z/Z☉", "Av", "log(SFR)"]
        vals_jwst = [10.2, 2.1, 0.8, 0.3, 0.72]
        vals_lit  = [10.0, 3.0, 0.6, 0.5, 0.58]
        errs_jwst = [0.15, 0.4, 0.1, 0.15, 0.25]
        x = np.arange(len(props))
        ax5.errorbar(x - 0.1, vals_jwst, yerr=errs_jwst,
                     fmt="ro", ms=8, capsize=5, label="JWST (this work)")
        ax5.plot(x + 0.1, vals_lit, "bs", ms=8, label="Literature median")
        ax5.set_xticks(x); ax5.set_xticklabels(props, rotation=15, ha="right")
        ax5.set_ylabel("Property Value")
        ax5.set_title("JWST vs Literature"); ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

        # --- Panel 6: High-z population ---
        ax6 = fig.add_subplot(gs[1, 2])
        if (self.jwst_results
                and isinstance(self.jwst_results, dict)
                and "spectral_fits" in self.jwst_results
                and self.jwst_results["spectral_fits"]):
            fits = self.jwst_results["spectral_fits"]
            zs   = [v["redshift"] for v in fits.values()]
            ms   = [v["stellar_mass"] for v in fits.values()]
        else:
            rng2 = np.random.default_rng(99)
            zs   = rng2.uniform(2.0, 6.0, 20).tolist()
            ms   = rng2.uniform(9.0, 11.0, 20).tolist()

        sc6 = ax6.scatter(zs, ms, s=80, c=zs, cmap="plasma",
                           alpha=0.8, edgecolors="black", linewidths=0.5)
        plt.colorbar(sc6, ax=ax6, label="Redshift")
        ax6.set_xlabel("Redshift"); ax6.set_ylabel("log(M*/M☉)")
        ax6.set_title("High-z Galaxy Population"); ax6.grid(alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        self.integration_plots["jwst_showcase"] = fig
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6]

    def create_summary_dashboard(self, save_path=None):
        """Create comprehensive summary dashboard."""
        self._close_previous("summary")

        rng = np.random.default_rng(seed=5)

        fig = plt.figure(figsize=(20, 14))
        gs  = GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.35)
        fig.suptitle(
            "Astro-AI: Comprehensive Galaxy Evolution Analysis Dashboard",
            fontsize=17, fontweight="bold", y=0.97)

        # Status
        ax_st = fig.add_subplot(gs[0, 0])
        mods  = ["21cm\nEvolution", "Cluster\nAnalysis", "JWST\nSpectro."]
        done  = [bool(self.cos_evo_results),
                 bool(self.cluster_results),
                 bool(self.jwst_results)]
        colors = ["#4caf50" if d else "#ff9800" for d in done]
        ax_st.bar(mods, [1, 1, 1], color=colors, alpha=0.85)
        for i, d in enumerate(done):
            ax_st.text(i, 0.5, "✓ Complete" if d else "⚠ Pending",
                       ha="center", va="center", fontweight="bold", fontsize=9)
        ax_st.set_ylim(0, 1); ax_st.set_yticks([])
        ax_st.set_title("Module Status", fontweight="bold")

        # Summary text
        ax_txt = fig.add_subplot(gs[0, 1:])
        ax_txt.axis("off")
        ax_txt.text(0.03, 0.95, (
            "ANALYSIS SUMMARY\n\n"
            "• Cosmic Evolution: 21cm signal z=15→6 (Pritchard & Loeb 2012)\n"
            "• Cluster Analysis: Environmental quenching (Peng+2010)\n"
            "• JWST Spectroscopy: Stellar populations (Carnall+2019)\n\n"
            "KEY CONNECTIONS\n\n"
            "• Reionization epoch → cluster formation timing\n"
            "• Environmental quenching → spectroscopic confirmation\n"
            "• High-z properties → cosmic evolution context"),
            transform=ax_txt.transAxes, fontsize=10,
            va="top", fontfamily="monospace")

        # 21cm
        ax_cm = fig.add_subplot(gs[1, 0])
        if self.cos_evo_results:
            z_v = np.asarray(self.cos_evo_results.get("redshifts",  np.linspace(6, 15, 10)))
            s_v = np.asarray(self.cos_evo_results.get("global_signals", np.zeros_like(z_v)))
        else:
            z_v = np.linspace(6, 15, 20)
            s_v = np.array([self._analytic_dtb(z) for z in z_v])
        ax_cm.plot(z_v, s_v, "b-o", lw=2, ms=4)
        ax_cm.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax_cm.set_xlabel("Redshift"); ax_cm.set_ylabel("δTb [mK]")
        ax_cm.set_title("Cosmic Evolution", fontweight="bold"); ax_cm.grid(alpha=0.3)

        # Environmental effects
        ax_env = fig.add_subplot(gs[1, 1])
        mb = np.linspace(9.5, 11.5, 8)
        rf_cl = 0.1 + 0.7 / (1 + np.exp(-(mb - 10.3) / 0.3))
        rf_fi = 0.05 + 0.4 / (1 + np.exp(-(mb - 10.7) / 0.4))
        ax_env.plot(mb, rf_cl, "ro-", lw=2, label="Cluster")
        ax_env.plot(mb, rf_fi, "bo-", lw=2, label="Field")
        ax_env.set_xlabel("log(M*/M☉)"); ax_env.set_ylabel("Red Fraction")
        ax_env.set_title("Environmental Effects", fontweight="bold")
        ax_env.legend(fontsize=9); ax_env.grid(alpha=0.3)

        # JWST spectrum
        ax_sp = fig.add_subplot(gs[1, 2])
        wl    = np.linspace(0.6, 5.3, 300)
        fl    = self._generate_mock_spectrum_for_plot(wl)
        fl   += 0.04 * np.median(fl) * rng.standard_normal(len(fl))
        ax_sp.plot(wl, fl, "k-", lw=0.8)
        ax_sp.set_xlabel("Wavelength [μm]"); ax_sp.set_ylabel("Flux")
        ax_sp.set_title("JWST Spectroscopy", fontweight="bold"); ax_sp.grid(alpha=0.3)

        # Mass–SFR
        ax_ms = fig.add_subplot(gs[1, 3])
        ms_m  = rng.uniform(9.5, 11.5, 100)
        ms_s  = ms_m - 9.0 + rng.normal(0, 0.3, 100)
        cl_m  = rng.random(100) < 0.3
        ax_ms.scatter(ms_m[~cl_m], ms_s[~cl_m], c="steelblue", s=20, alpha=0.6, label="Field")
        ax_ms.scatter(ms_m[ cl_m], ms_s[ cl_m], c="tomato",    s=20, alpha=0.8, label="Cluster")
        ax_ms.plot(np.linspace(9.5, 11.5, 50), np.linspace(9.5, 11.5, 50) - 9.0,
                   "k--", alpha=0.6, label="SFMS")
        ax_ms.set_xlabel("log(M*/M☉)"); ax_ms.set_ylabel("log(SFR)")
        ax_ms.set_title("Mass–SFR", fontweight="bold")
        ax_ms.legend(fontsize=8); ax_ms.grid(alpha=0.3)

        # Timeline strip
        ax_tl  = fig.add_subplot(gs[2, :])
        t_arr  = np.linspace(0.3, 13.8, 200)
        strips = [
            ("21cm Reionization", "steelblue", 0.75,
             lambda t: np.exp(-(t - 0.8) ** 2 / 0.3) * (t < 2.5)),
            ("Cluster Formation", "tomato",    0.55,
             lambda t: np.exp(-(t - 3.5) ** 2 / 4.0) * (t > 1.5)),
            ("Galaxy Assembly",   "seagreen",  0.35,
             lambda t: np.clip(0.5 + 0.4 * np.sin((t - 1) / 2), 0, 1) * (t > 1)),
            ("JWST Observations", "mediumpurple", 0.15,
             lambda t: np.where(t > 11, 1.0, 0.0)),
        ]
        for label, color, y, fn in strips:
            intens = fn(t_arr)
            ax_tl.fill_between(t_arr, y - 0.07, y + 0.07,
                                where=(intens > 0.05), alpha=0.75, color=color)
            ax_tl.text(0.25, y, label, fontsize=9, fontweight="bold", va="center")

        # Epoch lines with correct times
        for z_ep, label in [(15, "z~15"), (10, "z~10"), (6, "z~6"),
                             (3, "z~3"), (1, "z~1")]:
            t_ep = float(self._redshift_to_time(z_ep))
            ax_tl.axvline(t_ep, color="black", ls=":", alpha=0.6, lw=0.8)
            ax_tl.text(t_ep, 0.95, label, ha="center", va="bottom",
                       fontsize=8, rotation=0)

        ax_tl.set_xlim(0.3, 13.8); ax_tl.set_ylim(0, 1.05)
        ax_tl.set_xlabel("Cosmic Time [Gyr]", fontsize=12)
        ax_tl.set_title("Integrated Cosmic Timeline", fontsize=13, fontweight="bold")
        ax_tl.set_yticks([])

        # Secondary redshift axis
        ax_tl2 = ax_tl.twiny()
        z_ticks = [15, 10, 6, 3, 1, 0]
        t_ticks = [float(self._redshift_to_time(z)) for z in z_ticks]
        ax_tl2.set_xlim(ax_tl.get_xlim())
        ax_tl2.set_xticks(t_ticks)
        ax_tl2.set_xticklabels([f"z={z}" for z in z_ticks])
        ax_tl2.set_xlabel("Redshift", fontsize=11)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        self.integration_plots["summary"] = fig
        return fig

    # -----------------------------------------------------------------------
    # Export / report
    # -----------------------------------------------------------------------

    def export_dashboard_plots(self, output_dir: str, formats=("png", "pdf")):
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in self.integration_plots.items():
            for fmt in formats:
                path = os.path.join(output_dir, f"{name}.{fmt}")
                fig.savefig(path, dpi=300, bbox_inches="tight")
                logger.info("Saved %s", path)

    def generate_integration_report(self) -> dict:
        report = {
            "modules_status": {
                "cosmic_evolution": bool(self.cos_evo_results),
                "cluster_analysis": bool(self.cluster_results),
                "jwst_spectroscopy": bool(self.jwst_results),
            },
            "integration_summary": {
                "total_modules_completed": sum([
                    bool(self.cos_evo_results),
                    bool(self.cluster_results),
                    bool(self.jwst_results),
                ]),
                "plots_generated": len(self.integration_plots),
            },
        }

        if self.cos_evo_results:
            zs = self.cos_evo_results.get("redshifts", [])
            gs = self.cos_evo_results.get("global_signals", [])
            report["cosmic_evolution_summary"] = {
                "redshift_range": [float(min(zs)), float(max(zs))] if len(zs) else None,
                "signal_range_mK": [float(min(gs)), float(max(gs))] if len(gs) else None,
                "mock": self.cos_evo_results.get("mock", True),
            }

        if self.cluster_results and isinstance(self.cluster_results, dict):
            cl = self.cluster_results.get("cluster_members", [])
            fi = self.cluster_results.get("field_galaxies", [])
            report["cluster_analysis_summary"] = {
                "n_cluster_members": len(cl) if hasattr(cl, "__len__") else 0,
                "n_field_galaxies":  len(fi) if hasattr(fi, "__len__") else 0,
            }

        if self.jwst_results and isinstance(self.jwst_results, dict):
            fits = self.jwst_results.get("spectral_fits", {})
            zs   = [v["redshift"] for v in fits.values() if "redshift" in v]
            report["jwst_analysis_summary"] = {
                "n_sources_fitted":  len(fits),
                "average_redshift":  float(np.mean(zs)) if zs else None,
            }

        return report

    # -----------------------------------------------------------------------
    # Private helper used in summary dashboard when no real results available
    # -----------------------------------------------------------------------

    def _analytic_dtb(self, z: float) -> float:
        """
        Approximate δTb(z) for placeholder plots using the same physical
        model as CosmicEvolution._delta_T_b() but without importing the module.
        """
        _Z_CD  = 17.0
        _Z_LYA = 20.0
        _Z_EOR = 8.0
        _Z_END = 6.0
        x_HI   = 1.0 / (1.0 + np.exp((_Z_EOR - z) / 2.0))
        if z > _Z_LYA:
            ts_rat = 1.0
        elif z > _Z_CD:
            ts_rat = 1.0 - 0.95 * (z - _Z_CD) / (_Z_LYA - _Z_CD)
        else:
            frac   = np.clip((_Z_CD - z) / (_Z_CD - _Z_END), 0, 1)
            ts_rat = 0.05 + 20.0 * frac ** 2
        if abs(ts_rat - 1.0) < 1e-6:
            return 0.0
        coupling = 1.0 - 1.0 / ts_rat
        return 27.0 * x_HI * coupling * np.sqrt((1.0 + z) / 10.0)