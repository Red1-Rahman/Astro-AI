# Cosmic Evolution Module for Astro-AI
#
# Copyright (c) 2025 Redwan Rahman and CAM-SUST
#
# Physically grounded mock signal based on:
#   - Pritchard & Loeb 2012 (Rev. Mod. Phys. 84, 1):
#       global 21cm signal parameterisation
#   - Cohen et al. 2017 (ApJ 847, 64):
#       absorption trough amplitude and redshift
#   - Furlanetto et al. 2006 (Phys. Rep. 433, 181):
#       power-spectrum shape and amplitude
#   - Mesinger et al. 2011 (MNRAS 411, 955):
#       21cmFAST methodology

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from numpy import pi
import warnings
import logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import py21cmfast as p21c
    from py21cmfast import plotting, cache_tools
    HAVE_21CMFAST = True
except ImportError:
    HAVE_21CMFAST = False
    logger.warning("py21cmfast not available — using physically motivated analytic model.")

try:
    import tools21cm as t2c
    HAVE_TOOLS21CM = True
except ImportError:
    HAVE_TOOLS21CM = False


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_T21_FACTOR = 27.0          # mK — prefactor in dTb expression (Furlanetto+2006 eq.1)
_Z_REION_END   = 6.0        # End of reionization (Fan+2006)
_Z_REION_MID   = 8.0        # Midpoint of reionization (Planck 2020)
_Z_REION_START = 12.0       # Start of reionization (conservative)
_Z_CD_PEAK     = 17.0       # Cosmic Dawn absorption trough (Cohen+2017)
_Z_LYMAN_COUP  = 20.0       # Wouthuysen-Field coupling onset
_DT_ABSORPTION = -120.0     # mK  absorption trough amplitude (Cohen+2017 median)
_DT_EMISSION   =   20.0     # mK  weak emission signal during early reionization
_SIGMA8_FIDUCIAL = 0.8118   # Planck 2018


class CosmicEvolution:
    """
    Cosmic Evolution Analysis using 21cmFAST simulations.

    When py21cmfast is unavailable the module falls back to a physically
    motivated analytic model of the global 21cm signal and power spectrum,
    parameterised from Pritchard & Loeb (2012) and Furlanetto et al. (2006).
    All mock quantities carry explicit 'SIMULATED' labels in the UI so that
    users are never misled about data provenance.
    """

    def __init__(self, cosmology_params: dict):
        """
        Parameters
        ----------
        cosmology_params : dict
            H0      – Hubble constant [km/s/Mpc]
            Om0     – matter density parameter
            sigma8  – amplitude of matter fluctuations
            z_range – [z_min, z_max]
        """
        self.cosmo_params = cosmology_params
        self.results: dict = {}
        self.using_mock = not HAVE_21CMFAST

        h = cosmology_params.get("H0", 67.66) / 100.0
        self.default_params = {
            "HII_DIM": 50,
            "BOX_LEN": 50,
            "SIGMA_8": cosmology_params.get("sigma8", _SIGMA8_FIDUCIAL),
            "hlittle": h,
            "OMm":     cosmology_params.get("Om0", 0.31),
        }

    # -----------------------------------------------------------------------
    # ANALYTIC PHYSICS HELPERS
    # -----------------------------------------------------------------------

    def _x_HI(self, z: float) -> float:
        """
        Neutral hydrogen fraction x_HI(z).

        Smooth sigmoid transition between fully neutral (z >> z_reion_start)
        and fully ionized (z < z_reion_end), calibrated to midpoint z~8
        (Planck 2020 optical depth constraint).
        """
        # Sigmoid centred on reionization midpoint with width ~2
        return 1.0 / (1.0 + np.exp(((_Z_REION_MID - z) / 2.0)))

    def _T_spin_over_T_cmb(self, z: float) -> float:
        """
        Spin temperature coupling ratio T_S / T_CMB.

        Three physical regimes following Pritchard & Loeb (2012) §3:
          z > z_lyman  : no Ly-α coupling, T_S → T_CMB  → δTb → 0
          z_cd < z < z_lyman : Ly-α coupling drives T_S below T_CMB (absorption)
          z < z_cd     : X-ray heating raises T_S above T_CMB (emission/reionization)
        """
        if z > _Z_LYMAN_COUP:
            return 1.0          # no coupling yet
        elif z > _Z_CD_PEAK:
            # Ly-α coupling growing, spin temperature cooling toward T_gas < T_CMB
            frac = (z - _Z_CD_PEAK) / (_Z_LYMAN_COUP - _Z_CD_PEAK)
            # T_S/T_CMB interpolates from 1 (no coupling) to ~0.05 (deep absorption)
            return 1.0 - 0.95 * frac
        else:
            # X-ray heating: T_S rises back through T_CMB and above
            frac = (_Z_CD_PEAK - z) / (_Z_CD_PEAK - _Z_REION_START)
            frac = np.clip(frac, 0.0, 1.0)
            # Goes from ~0.05 at z_cd_peak to >> 1 (saturated) by z_reion_start
            return 0.05 + 20.0 * frac**2

    def _delta_T_b(self, z: float) -> float:
        """
        Global (sky-averaged) 21cm brightness temperature offset δTb [mK].

        Based on Furlanetto et al. 2006 eq. 1 and Pritchard & Loeb 2012 eq. 3,
        evaluated at the mean density (δ=0) with spin temperature approximation
        from _T_spin_over_T_cmb().

        δTb ≈ 27 x_HI (1+δ) (1 - T_CMB/T_S)
                × sqrt((1+z)/10 · 0.15/Ω_m h²) · Ω_b h²/0.023   [mK]
        """
        Om0    = self.default_params["OMm"]
        h      = self.default_params["hlittle"]
        Ob_h2  = 0.0224   # Planck 2018 best-fit

        x_HI   = self._x_HI(z)
        ts_rat = self._T_spin_over_T_cmb(z)         # T_S / T_CMB

        if abs(ts_rat - 1.0) < 1e-6:
            return 0.0                               # no signal when T_S = T_CMB

        coupling = 1.0 - 1.0 / ts_rat               # (1 - T_CMB/T_S)

        cosmo_factor = (
            np.sqrt((1.0 + z) / 10.0 * 0.15 / (Om0 * h**2))
            * Ob_h2 / 0.023
        )

        return _T21_FACTOR * x_HI * coupling * cosmo_factor

    def _mock_power_spectrum(self, z: float, k_arr: np.ndarray) -> np.ndarray:
        """
        Dimensionless 21cm power spectrum Δ²(k) = k³P(k)/(2π²) [mK²].

        Analytic approximation calibrated to 21cmFAST outputs
        (Mesinger et al. 2011; Greig & Mesinger 2015):

          Δ²(k) ≈ A(z) · (k/k_pivot)^n_eff · exp(-k/k_damp)

        where A(z) encodes the redshift-dependent amplitude driven by
        the ionization and spin-temperature fluctuations.
        """
        k_pivot = 0.1     # h/Mpc — characteristic scale of 21cm fluctuations
        k_damp  = 2.0     # h/Mpc — damping scale (finite mean free path)
        n_eff   = 0.5     # effective spectral index on large scales

        # Amplitude peaks near midpoint of reionization and at Cosmic Dawn
        dtb = abs(self._delta_T_b(z))
        # Fluctuation amplitude ~ 10-30% of global signal squared
        A = 0.15 * dtb**2
        # Add ionization-driven bump near reionization midpoint
        A += 50.0 * np.exp(-((z - _Z_REION_MID)**2) / 4.0)

        ps = A * (k_arr / k_pivot)**n_eff * np.exp(-k_arr / k_damp)
        return np.maximum(ps, 0.0)

    def _mock_brightness_temperature_cube(self, box_size: int, z: float,
                                          seed: int = 42) -> np.ndarray:
        """
        Generate a 3-D brightness temperature cube [mK] consistent with
        the analytic global signal and power spectrum.

        Method: draw a Gaussian random field in Fourier space with the
        correct power spectrum, then shift the mean to δTb(z).
        This preserves the correct 2-point statistics while being fully
        analytic (no py21cmfast required).
        """
        rng = np.random.default_rng(seed + int(z * 100))
        box_len_mpc = float(box_size)

        # Fourier-space grid
        freq = fftpack.fftfreq(box_size, d=box_len_mpc / box_size)
        kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
        k_mag = 2.0 * pi * np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag[0, 0, 0] = 1.0   # avoid division by zero at DC

        # Power spectrum on the grid
        ps_grid = self._mock_power_spectrum(z, k_mag)
        # Convert dimensionless Δ² to P(k): P = Δ² · 2π²/k³
        pk_grid = ps_grid * (2.0 * pi**2) / k_mag**3
        pk_grid[0, 0, 0] = 0.0  # zero mean

        # Draw Gaussian random field
        amp   = np.sqrt(pk_grid / 2.0)
        noise = (rng.standard_normal((box_size, box_size, box_size))
                 + 1j * rng.standard_normal((box_size, box_size, box_size)))
        field_k = amp * noise

        # Transform to real space
        field_r = np.real(fftpack.ifftn(field_k)) * box_size**3

        # Normalise to unit variance then scale to physical δTb fluctuations
        std = field_r.std()
        if std > 0:
            field_r /= std

        # RMS fluctuations ~ 20% of mean signal amplitude (order-of-magnitude)
        mean_dtb = self._delta_T_b(z)
        rms_fluct = 0.20 * abs(mean_dtb) if abs(mean_dtb) > 1.0 else 5.0
        cube = mean_dtb + rms_fluct * field_r

        return cube.astype(np.float32)

    # -----------------------------------------------------------------------
    # PUBLIC API — mirrors the py21cmfast path exactly
    # -----------------------------------------------------------------------

    def brightness_temperature(self, box_size: int = 50, redshift: float = 10.0,
                                hubble=None, matter=None, random_seed: int = 54321):
        """Return a brightness temperature object (real or mock)."""
        if HAVE_21CMFAST:
            return self._real_brightness_temperature(
                box_size, redshift, hubble, matter, random_seed)
        return self._MockBT(
            self._mock_brightness_temperature_cube(box_size, redshift, random_seed))

    class _MockBT:
        """Thin wrapper so mock and real paths share the same attribute."""
        def __init__(self, cube: np.ndarray):
            self.brightness_temp = cube

    def _real_brightness_temperature(self, box_size, redshift, hubble, matter, seed):
        cosmo = p21c.CosmoParams(
            SIGMA_8 = self.default_params["SIGMA_8"],
            hlittle = hubble if hubble is not None else self.default_params["hlittle"],
            OMm     = matter if matter is not None else self.default_params["OMm"],
        )
        ic  = p21c.initial_conditions(
            user_params={"HII_DIM": box_size, "BOX_LEN": box_size},
            cosmo_params=cosmo, random_seed=seed)
        pf  = p21c.perturb_field(redshift=redshift, init_boxes=ic)
        ib  = p21c.ionize_box(perturbed_field=pf)
        return p21c.brightness_temperature(ionized_box=ib, perturbed_field=pf)

    def compute_power_spectrum_1d(self, cube: np.ndarray,
                                   kbins: int = 15, box_length: float = 50.0):
        """Compute spherically averaged 1-D power spectrum from 3-D cube."""
        h       = self.default_params["hlittle"]
        box_dims = [float(box_length)] * 3

        ft = fftpack.fftshift(fftpack.fftn(cube.astype("float64")))
        ps = np.abs(ft)**2

        boxvol    = box_length**3
        pixelsize = boxvol / cube.size
        ps       *= pixelsize**2 / boxvol

        ps_1d, ks, _ = self._radial_average(ps, kbins=kbins, box_dims=box_dims)
        return ks, ps_1d * ks**3 / (2.0 * pi**2)

    def _radial_average(self, arr, box_dims, kbins=10):
        _, k = self._get_k(arr, box_dims)
        kbins_arr = self._get_kbins(kbins, box_dims, k)
        dk        = (kbins_arr[1:] - kbins_arr[:-1]) / 2.0
        outdata   = np.histogram(k.flatten(), bins=kbins_arr,
                                 weights=arr.flatten())[0]
        n_modes   = np.histogram(k.flatten(), bins=kbins_arr)[0].astype(float)
        n_modes[n_modes == 0] = 1
        outdata  /= n_modes
        return outdata, kbins_arr[:-1] + dk, n_modes

    def _get_k(self, arr, box_dims):
        nx, ny, nz = arr.shape
        x, y, z    = np.indices(arr.shape, dtype="int32")
        cx, cy, cz = nx / 2, ny / 2, nz / 2
        kx = 2.0 * pi * (x - cx) / box_dims[0]
        ky = 2.0 * pi * (y - cy) / box_dims[1]
        kz_ = 2.0 * pi * (z - cz) / box_dims[2]
        k   = np.sqrt(kx**2 + ky**2 + kz_**2)
        return [kx, ky, kz_], k

    def _get_kbins(self, kbins, box_dims, k):
        if isinstance(kbins, int):
            kmin   = 2.0 * pi / min(box_dims)
            kbins  = 10 ** np.linspace(np.log10(kmin), np.log10(k.max()), kbins + 1)
        return kbins

    # -----------------------------------------------------------------------
    # SIMULATION RUNNER
    # -----------------------------------------------------------------------

    def run_simulation(self, box_size: int = 50, resolution: int = 50,
                       z_start: float = 15.0, z_end: float = 6.0,
                       z_step: float = 1.0) -> dict:
        """
        Run 21cm simulation across redshift range.

        Uses py21cmFAST when available, otherwise falls back to the
        physically motivated analytic model.  The 'mock' key in the
        returned dict is True whenever real simulations were not used —
        the UI must surface this clearly to the user.
        """
        z_array = np.arange(z_end, z_start + z_step, z_step)[::-1]
        logger.info("Running simulation for z = %s (mock=%s)", z_array, self.using_mock)

        brightness_temps, ps_k_list, ps_list, global_signals = [], [], [], []

        for z in z_array:
            bt = self.brightness_temperature(box_size=resolution, redshift=float(z))
            cube = bt.brightness_temp

            brightness_temps.append(cube)
            k, ps = self.compute_power_spectrum_1d(cube, box_length=float(box_size))
            ps_k_list.append(k)
            ps_list.append(ps)
            global_signals.append(float(np.mean(cube)))

        self.results = {
            "redshifts":             z_array,
            "brightness_temperatures": brightness_temps,
            "power_spectra_k":       ps_k_list,
            "power_spectra_ps":      ps_list,
            "global_signals":        global_signals,
            "mock":                  self.using_mock,
            "simulation_params": {
                "box_size":   box_size,
                "resolution": resolution,
                "cosmology":  self.cosmo_params,
            },
        }
        return self.results

    # -----------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------

    def _mock_label(self, ax):
        """Add a clear SIMULATED watermark when real data are unavailable."""
        if self.results.get("mock", False):
            ax.text(0.98, 0.02, "SIMULATED DATA — not real observations",
                    transform=ax.transAxes, fontsize=8, color="darkorange",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                              ec="darkorange", alpha=0.9))

    def plot_global_evolution(self, save_path=None):
        """Plot global 21cm signal vs redshift with physical epoch annotations."""
        if not self.results:
            raise ValueError("Run simulation first.")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.results["redshifts"], self.results["global_signals"],
                "b-o", lw=2, ms=5, label="Global δTb")

        # Physical epoch markers
        ax.axvline(_Z_REION_END,   color="green",  ls="--", alpha=0.7,
                   label=f"End of reionization (z≈{_Z_REION_END})")
        ax.axvline(_Z_REION_MID,   color="orange", ls="--", alpha=0.7,
                   label=f"Reionization midpoint (z≈{_Z_REION_MID})")
        ax.axvline(_Z_CD_PEAK,     color="red",    ls=":",  alpha=0.7,
                   label=f"Cosmic Dawn peak (z≈{_Z_CD_PEAK})")
        ax.axhline(0, color="black", ls="-", lw=0.5, alpha=0.4)

        ax.set_xlabel("Redshift z",              fontsize=13)
        ax.set_ylabel("δTb [mK]",               fontsize=13)
        ax.set_title("Global 21cm Signal Evolution", fontsize=15)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        self._mock_label(ax)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_power_spectra_evolution(self, save_path=None):
        """Plot 21cm power spectrum Δ²(k) at each simulated redshift."""
        if not self.results:
            raise ValueError("Run simulation first.")

        fig, ax = plt.subplots(figsize=(12, 7))
        colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(self.results["redshifts"])))

        for i, z in enumerate(self.results["redshifts"]):
            k  = self.results["power_spectra_k"][i]
            ps = self.results["power_spectra_ps"][i]
            mask = (ps > 0) & (k > 0)
            if mask.sum() > 1:
                ax.loglog(k[mask], ps[mask], color=colors[i], lw=2, label=f"z = {z:.1f}")

        ax.set_xlabel("k [Mpc⁻¹]",              fontsize=13)
        ax.set_ylabel("Δ²(k) = k³P(k)/2π² [mK²]", fontsize=13)
        ax.set_title("21cm Power Spectrum Evolution", fontsize=15)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
        ax.legend(ncol=2, fontsize=9)
        self._mock_label(ax)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_brightness_temperature_slices(self, z_indices=None, save_path=None):
        """Plot 2-D slices of the brightness temperature cube."""
        if not self.results:
            raise ValueError("Run simulation first.")

        n_z = len(self.results["redshifts"])
        if z_indices is None:
            z_indices = [0, n_z // 2, n_z - 1]

        fig, axes = plt.subplots(1, len(z_indices),
                                  figsize=(5 * len(z_indices), 4))
        if len(z_indices) == 1:
            axes = [axes]

        for i, zi in enumerate(z_indices):
            cube  = self.results["brightness_temperatures"][zi]
            z_val = self.results["redshifts"][zi]
            sl    = cube[:, :, cube.shape[2] // 2]

            vmax = max(abs(sl.min()), abs(sl.max()))
            im   = axes[i].imshow(sl, origin="lower", cmap="RdBu_r",
                                   vmin=-vmax, vmax=vmax,
                                   interpolation="bilinear")
            axes[i].set_title(f"z = {z_val:.1f}", fontsize=13)
            axes[i].set_xlabel("x [pixels]")
            if i == 0:
                axes[i].set_ylabel("y [pixels]")
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label("δTb [mK]", fontsize=9)

        if self.results.get("mock", False):
            fig.text(0.5, 0.01,
                     "SIMULATED DATA — not real observations",
                     ha="center", fontsize=8, color="darkorange",
                     bbox=dict(boxstyle="round", fc="lightyellow",
                               ec="darkorange", alpha=0.9))

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, axes

    # -----------------------------------------------------------------------
    # SUMMARY REPORT
    # -----------------------------------------------------------------------

    def generate_summary_report(self) -> dict:
        if not self.results:
            raise ValueError("Run simulation first.")

        signals   = np.array(self.results["global_signals"])
        redshifts = np.array(self.results["redshifts"])

        i_min = int(np.argmin(signals))
        i_max = int(np.argmax(signals))

        return {
            "data_provenance": {
                "mode": "analytic mock (py21cmfast not installed)"
                         if self.results.get("mock") else "py21cmFAST simulation",
                "references": [
                    "Pritchard & Loeb 2012, Rev. Mod. Phys. 84, 1",
                    "Furlanetto et al. 2006, Phys. Rep. 433, 181",
                    "Mesinger et al. 2011, MNRAS 411, 955",
                ],
            },
            "simulation_summary": {
                "redshift_range":  [float(redshifts.min()), float(redshifts.max())],
                "n_redshift_steps": len(redshifts),
                "box_size_mpc":    self.results["simulation_params"]["box_size"],
                "resolution":      self.results["simulation_params"]["resolution"],
            },
            "signal_characteristics": {
                "absorption_min_mK":       float(signals[i_min]),
                "absorption_min_redshift": float(redshifts[i_min]),
                "emission_max_mK":         float(signals[i_max]),
                "emission_max_redshift":   float(redshifts[i_max]),
                "signal_range_mK":         float(signals.max() - signals.min()),
            },
            "physical_interpretation": {
                "reionization_signature":
                    bool(np.any(signals < -20)),
                "cosmic_dawn_absorption":
                    bool(np.any(signals < -50)),
                "x_ray_heating_visible":
                    bool(np.any(signals > 0)),
                "evolution_trend":
                    "heating" if signals[-1] > signals[0] else "cooling",
            },
        }

    def export_results(self, output_path: str):
        if not self.results:
            raise ValueError("Run simulation first.")
        if output_path.endswith(".npz"):
            save_dict = {
                "redshifts":      self.results["redshifts"],
                "global_signals": np.array(self.results["global_signals"]),
                "mock":           np.array([self.results.get("mock", False)]),
            }
            for i, (k, ps) in enumerate(zip(self.results["power_spectra_k"],
                                             self.results["power_spectra_ps"])):
                save_dict[f"ps_k_{i}"]  = k
                save_dict[f"ps_ps_{i}"] = ps
            np.savez(output_path, **save_dict)
        else:
            raise ValueError("Only .npz export supported without h5py.")
        logger.info("Results exported to %s", output_path)