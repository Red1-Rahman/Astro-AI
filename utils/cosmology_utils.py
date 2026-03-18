"""
Cosmology Utilities Module for Astro-AI
Provides cosmological calculations and conversions using Astropy.

Fixes vs original:
  1. redshift_velocity() and velocity_to_redshift() used the non-relativistic
     approximation v = z·c.  This is only valid for z ≪ 1.  At z=0.5 the
     error is 30%; at z=1 it is 67%; at z=2 it exceeds 150%.  Since this
     platform analyses galaxies out to z~10, the non-relativistic formula
     produces physically wrong velocities.  Both methods now use the special
     relativistic Doppler formula:
       z = sqrt((1+β)/(1-β)) − 1  ↔  β = ((1+z)²−1)/((1+z)²+1)
     with a clear docstring note and the old formula available as a kwarg
     for cases where callers genuinely need the cosmological redshift
     approximation (peculiar velocity work).

  2. age_to_redshift() searched z ∈ [0, 20] by default but brentq() raises
     ValueError when the age_gyr is older than the universe at z=0 (i.e.
     > 13.8 Gyr) or younger than the universe at z=20 (~0.18 Gyr for
     Planck18).  The function now validates the input range, returns np.nan
     with a logger.warning rather than crashing, and documents the limits.

  3. astropy is a hard dependency used throughout — if it is missing the
     module raised ImportError only at method call time with a confusing
     traceback.  A top-level import guard now provides a clear error message
     at import time.

  4. angular_scale() docstring said "kpc per arcsec" but the formula gave
     Mpc · (π/648000) which is kpc/arcsec only after the *1000 factor.
     Formula is unchanged but the intermediate steps are clarified.
"""

import logging
from typing import Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from astropy.cosmology import FlatLambdaCDM, Planck18
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    _HAVE_ASTROPY = True
except ImportError:
    _HAVE_ASTROPY = False
    raise ImportError(
        "astropy is required for CosmologyUtils. "
        "Install it with: pip install astropy"
    )

# Speed of light — single source of truth
_C_KMS = 299792.458   # km/s  (IAU 2012 definition)


class CosmologyUtils:
    """
    Utility class for cosmological calculations.

    Uses Astropy's FlatLambdaCDM with Planck 2018 parameters as default.
    All distance methods return values stripped of astropy units so results
    are plain Python floats / numpy arrays.
    """

    def __init__(
        self,
        H0:  float = 67.66,
        Om0: float = 0.3111,
        Ob0: float = 0.0490,
    ):
        """
        Parameters
        ----------
        H0  : Hubble constant [km/s/Mpc]  (Planck18 default)
        Om0 : matter density parameter     (Planck18 default)
        Ob0 : baryon density parameter     (Planck18 default)
        """
        self.cosmo    = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)
        self.planck18 = Planck18

    # -----------------------------------------------------------------------
    # Time / distance
    # -----------------------------------------------------------------------

    def redshift_to_age(
        self, z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Age of the universe at redshift *z* [Gyr]."""
        return self.cosmo.age(z).value

    def age_to_redshift(
        self,
        age_gyr: float,
        z_range: Tuple[float, float] = (0.0, 20.0),
    ) -> float:
        """
        Redshift corresponding to a given age of the universe [Gyr].

        Uses Brent's method to invert cosmo.age(z).

        Parameters
        ----------
        age_gyr : float
            Target age [Gyr].  Must lie within the age range implied by
            z_range.  Values outside this range return np.nan with a warning.
        z_range : tuple
            (z_min, z_max) search bracket.

        Returns
        -------
        float
            Corresponding redshift, or np.nan if out of range.
        """
        from scipy.optimize import brentq

        age_at_zmin = self.cosmo.age(z_range[0]).value
        age_at_zmax = self.cosmo.age(z_range[1]).value

        if not (age_at_zmax <= age_gyr <= age_at_zmin):
            logger.warning(
                "age_to_redshift: age_gyr=%.3f Gyr is outside the range "
                "[%.3f, %.3f] Gyr implied by z_range=%s. Returning nan.",
                age_gyr, age_at_zmax, age_at_zmin, z_range,
            )
            return float("nan")

        try:
            return brentq(
                lambda z: self.cosmo.age(z).value - age_gyr,
                z_range[0],
                z_range[1],
                xtol=1e-6,
            )
        except ValueError as e:
            logger.warning("age_to_redshift brentq failed: %s", e)
            return float("nan")

    def luminosity_distance(
        self, z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Luminosity distance D_L [Mpc]."""
        return self.cosmo.luminosity_distance(z).value

    def angular_diameter_distance(
        self, z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Angular diameter distance D_A [Mpc]."""
        return self.cosmo.angular_diameter_distance(z).value

    def comoving_distance(
        self, z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Comoving distance D_C [Mpc]."""
        return self.cosmo.comoving_distance(z).value

    def comoving_volume(
        self, z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Comoving volume out to redshift *z* [Gpc³]."""
        return self.cosmo.comoving_volume(z).value / 1e9

    def lookback_time(
        self, z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Lookback time to redshift *z* [Gyr]."""
        return self.cosmo.lookback_time(z).value

    def critical_density(
        self, z: Union[float, np.ndarray] = 0
    ) -> Union[float, np.ndarray]:
        """Critical density ρ_c(z) [g cm⁻³]."""
        return self.cosmo.critical_density(z).to(u.g / u.cm**3).value

    def hubble_parameter(
        self, z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Hubble parameter H(z) [km/s/Mpc]."""
        return self.cosmo.H(z).value

    # -----------------------------------------------------------------------
    # Angular scale
    # -----------------------------------------------------------------------

    def angular_scale(self, z: float) -> float:
        """
        Physical scale corresponding to 1 arcsecond at redshift *z*.

        Returns
        -------
        float
            Angular scale [kpc/arcsec].
        """
        d_A_mpc = self.angular_diameter_distance(z)   # Mpc
        d_A_kpc = d_A_mpc * 1e3                        # kpc
        arcsec_per_rad = 206265.0
        return d_A_kpc / arcsec_per_rad                # kpc/arcsec

    # -----------------------------------------------------------------------
    # Magnitude / flux conversions
    # -----------------------------------------------------------------------

    def absolute_to_apparent_magnitude(self, M: float, z: float) -> float:
        """
        Distance modulus conversion M → m.

        m = M + 5·log10(D_L / 10 pc)
        """
        d_L_pc = self.luminosity_distance(z) * 1e6    # Mpc → pc
        return M + 5.0 * np.log10(d_L_pc / 10.0)

    def apparent_to_absolute_magnitude(self, m: float, z: float) -> float:
        """
        Distance modulus conversion m → M.

        M = m − 5·log10(D_L / 10 pc)
        """
        d_L_pc = self.luminosity_distance(z) * 1e6
        return m - 5.0 * np.log10(d_L_pc / 10.0)

    def flux_to_luminosity(self, flux: float, z: float) -> float:
        """
        Convert flux to luminosity.

        Parameters
        ----------
        flux : float
            Observed flux [erg/s/cm²].

        Returns
        -------
        float
            Luminosity [erg/s].
        """
        d_L_cm = self.luminosity_distance(z) * 3.085677581e24   # Mpc → cm
        return flux * 4.0 * np.pi * d_L_cm**2

    def luminosity_to_flux(self, luminosity: float, z: float) -> float:
        """
        Convert luminosity to flux.

        Parameters
        ----------
        luminosity : float
            Intrinsic luminosity [erg/s].

        Returns
        -------
        float
            Observed flux [erg/s/cm²].
        """
        d_L_cm = self.luminosity_distance(z) * 3.085677581e24
        return luminosity / (4.0 * np.pi * d_L_cm**2)

    # -----------------------------------------------------------------------
    # Velocity / redshift conversions  (BUG FIX — relativistic formula)
    # -----------------------------------------------------------------------

    def redshift_velocity(
        self, z: float, relativistic: bool = True
    ) -> float:
        """
        Convert redshift to recession velocity.

        Parameters
        ----------
        z : float
            Redshift.
        relativistic : bool
            If True (default), use the special-relativistic Doppler formula:

              β = ((1+z)² − 1) / ((1+z)² + 1)
              v = β · c

            This is correct for any z.  The non-relativistic approximation
            v = z·c introduces 30% error at z=0.5, 67% at z=1, and >150%
            at z=2.  Set relativistic=False only when you are working with
            small peculiar velocities (z ≲ 0.01) and need the simple formula
            for a specific reason.

        Returns
        -------
        float
            Recession velocity [km/s].
        """
        if relativistic:
            z2 = (1.0 + z) ** 2
            beta = (z2 - 1.0) / (z2 + 1.0)
            return beta * _C_KMS
        else:
            return z * _C_KMS

    def velocity_to_redshift(
        self, v: float, relativistic: bool = True
    ) -> float:
        """
        Convert recession velocity to redshift.

        Parameters
        ----------
        v : float
            Recession velocity [km/s].
        relativistic : bool
            If True (default), invert the special-relativistic Doppler
            formula.  If False, uses z = v/c (only valid for v ≪ c).

        Returns
        -------
        float
            Corresponding redshift.
        """
        if relativistic:
            beta = v / _C_KMS
            if abs(beta) >= 1.0:
                raise ValueError(
                    f"Velocity |v|={abs(v):.1f} km/s ≥ c — unphysical."
                )
            return np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0
        else:
            return v / _C_KMS

    # -----------------------------------------------------------------------
    # Scale factor
    # -----------------------------------------------------------------------

    def scale_factor(
        self, z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Scale factor a = 1/(1+z)."""
        return 1.0 / (1.0 + z)

    def redshift_from_scale_factor(
        self, a: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Redshift from scale factor z = 1/a − 1."""
        return 1.0 / a - 1.0

    # -----------------------------------------------------------------------
    # Parameter accessors
    # -----------------------------------------------------------------------

    def get_cosmology_params(self) -> dict:
        """Return the current cosmological parameters as a plain dict."""
        return {
            "H0":    float(self.cosmo.H0.value),
            "Om0":   float(self.cosmo.Om0),
            "Ob0":   float(self.cosmo.Ob0),
            "Ode0":  float(self.cosmo.Ode0),
            "Tcmb0": float(self.cosmo.Tcmb0.value),
            "h":     float(self.cosmo.h),
        }

    @staticmethod
    def planck18_params() -> dict:
        """Return Planck 2018 cosmological parameters as a plain dict."""
        return {
            "H0":    float(Planck18.H0.value),
            "Om0":   float(Planck18.Om0),
            "Ob0":   float(Planck18.Ob0),
            "Ode0":  float(Planck18.Ode0),
            "Tcmb0": float(Planck18.Tcmb0.value),
            "h":     float(Planck18.h),
        }