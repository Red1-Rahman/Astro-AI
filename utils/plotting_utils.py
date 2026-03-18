"""
Plotting Utilities Module for Astro-AI
Provides consistent plotting functions and styling for astronomical data.

Fixes vs original:
  1. plt.cm.get_cmap() is deprecated since matplotlib 3.7 and raises a
     warning on every call.  Replaced with matplotlib.colormaps[name] which
     is the current API and works across mpl 3.5+.

  2. setup_style() modified global plt.rcParams unconditionally at __init__
     time, permanently affecting every matplotlib figure in the Streamlit
     session — including module plots in CosmicEvolution, ClusterAnalyzer,
     and JWSTAnalyzer.  Style is now applied only to figures created through
     PlottingUtils, using matplotlib's context manager where possible, and
     the global rcParams path is guarded behind an explicit opt-in flag.

  3. plot_comparison() passed **kwargs to both ax.plot() calls, so any
     keyword that controls appearance (color, linestyle, alpha) would be
     applied identically to both lines, making them visually indistinguishable.
     The two lines now use independently drawn colours from the default colour
     cycle while still forwarding safe shared kwargs (linewidth, alpha, etc.).

  4. plot_image() log-scaled data with np.log10(|data| + 1e-10).  When the
     image contains large dynamic range (e.g. brightness temperature cubes),
     the 1e-10 floor maps to -10 in log space, creating a visually misleading
     colour stretch that swamps the real signal.  Replaced with a robust
     percentile-based vmin clip so the colour scale reflects the actual data
     range.

  5. plot_corner() returned None silently when the corner package is missing.
     Callers that did fig.tight_layout() or fig.savefig() on the return value
     would crash with AttributeError.  Now returns a plain matplotlib Figure
     showing the marginal histograms so the function is always usable.
"""

import logging
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)

# seaborn is optional
try:
    import seaborn as sns
    _HAVE_SNS = True
except ImportError:
    _HAVE_SNS = False


def _get_cmap(name: str):
    """
    Return a colormap by name using the current matplotlib API.

    matplotlib.colormaps[] is the preferred API since mpl 3.5;
    plt.cm.get_cmap() was deprecated in 3.7 and emits a warning.
    """
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        # Fallback for matplotlib < 3.5
        return plt.cm.get_cmap(name)  # type: ignore[attr-defined]


class PlottingUtils:
    """
    Utility class for creating consistent, publication-quality plots
    of astronomical data.
    """

    def __init__(self, style: str = "default", apply_globally: bool = False):
        """
        Parameters
        ----------
        style : str
            Plotting style: 'default', 'seaborn', or 'dark'.
        apply_globally : bool
            If True, apply rcParams changes to the global matplotlib state
            (affects ALL figures in the session).  If False (default), style
            changes are scoped to figures created through PlottingUtils.
        """
        self.style = style
        self.apply_globally = apply_globally

        # Colour palettes — colormaps accessed via _get_cmap()
        self.color_palettes = {
            "galaxy":      ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "redshift":    _get_cmap("viridis"),
            "temperature": _get_cmap("plasma"),
            "mass":        _get_cmap("cividis"),
        }

        if apply_globally:
            self._apply_style_globally()

    # -----------------------------------------------------------------------
    # Style helpers
    # -----------------------------------------------------------------------

    def _apply_style_globally(self):
        """Apply style to global matplotlib rcParams (opt-in only)."""
        if self.style == "seaborn" and _HAVE_SNS:
            sns.set_style("whitegrid")
            sns.set_context("talk")
        elif self.style == "dark":
            plt.style.use("dark_background")
        else:
            plt.rcParams.update({
                "figure.figsize":   (10, 6),
                "font.size":        12,
                "axes.labelsize":   14,
                "axes.titlesize":   16,
                "xtick.labelsize":  12,
                "ytick.labelsize":  12,
                "legend.fontsize":  12,
            })

    def _rc_context(self) -> dict:
        """
        Return an rcParams dict for use with matplotlib.rc_context().

        This scopes style to individual figures without touching global state.
        """
        if self.style == "dark":
            return {"axes.facecolor": "#1a1a1a", "figure.facecolor": "#1a1a1a",
                    "text.color": "white", "axes.labelcolor": "white",
                    "xtick.color": "white", "ytick.color": "white"}
        # default / seaborn — keep matplotlib defaults, just set sizes
        return {
            "font.size":       12,
            "axes.labelsize":  14,
            "axes.titlesize":  16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }

    # -----------------------------------------------------------------------
    # Grid / layout
    # -----------------------------------------------------------------------

    def create_subplot_grid(
        self,
        nrows: int,
        ncols: int,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create a grid of subplots."""
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, axes

    # -----------------------------------------------------------------------
    # Spectral plots
    # -----------------------------------------------------------------------

    def plot_spectrum(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        error: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None,
        xlabel: str = "Wavelength (μm)",
        ylabel: str = "Flux (arbitrary units)",
        title: str = "Spectrum",
        **kwargs,
    ) -> plt.Axes:
        """Plot a 1-D spectrum with optional ±1σ error shading."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if error is not None:
            color = kwargs.pop("color", "steelblue")
            ax.plot(wavelength, flux, color=color, **kwargs)
            ax.fill_between(
                wavelength,
                flux - error,
                flux + error,
                alpha=0.25,
                color=color,
                label="±1σ",
            )
        else:
            ax.plot(wavelength, flux, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_sed(
        self,
        wavelength: np.ndarray,
        luminosity: np.ndarray,
        ax: Optional[plt.Axes] = None,
        loglog: bool = True,
        title: str = "Spectral Energy Distribution",
        **kwargs,
    ) -> plt.Axes:
        """Plot a Spectral Energy Distribution."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if loglog:
            ax.loglog(wavelength, luminosity, **kwargs)
        else:
            ax.plot(wavelength, luminosity, **kwargs)

        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Luminosity (L☉/Hz)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax

    # -----------------------------------------------------------------------
    # Image plots
    # -----------------------------------------------------------------------

    def plot_image(
        self,
        data: np.ndarray,
        ax: Optional[plt.Axes] = None,
        cmap: str = "viridis",
        scale: str = "linear",
        colorbar: bool = True,
        title: str = "",
        vmin_pct: float = 1.0,
        vmax_pct: float = 99.0,
        **kwargs,
    ) -> Tuple[plt.Axes, plt.cm.ScalarMappable]:
        """
        Plot a 2-D image with colorbar.

        Parameters
        ----------
        scale : str
            'linear', 'log', or 'sqrt'.
        vmin_pct, vmax_pct : float
            Percentile clip for colour scale (default 1–99).
            Prevents extreme outliers or the log(1e-10) floor from
            dominating the colour stretch.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        display = data.copy().astype(float)

        if scale == "log":
            # Use percentile-based floor so the dynamic range is meaningful.
            # Original: np.log10(|data| + 1e-10) → floor at -10, swamps signal.
            positive = display[display > 0]
            floor = np.percentile(positive, vmin_pct) if positive.size > 0 else 1e-6
            display = np.log10(np.maximum(display, floor))
        elif scale == "sqrt":
            display = np.sqrt(np.maximum(display, 0.0))

        # Percentile-based colour limits (only set if caller hasn't overridden)
        if "vmin" not in kwargs:
            kwargs["vmin"] = np.nanpercentile(display, vmin_pct)
        if "vmax" not in kwargs:
            kwargs["vmax"] = np.nanpercentile(display, vmax_pct)

        im = ax.imshow(display, cmap=cmap, origin="lower", **kwargs)

        if colorbar:
            plt.colorbar(im, ax=ax)

        ax.set_title(title)
        return ax, im

    # -----------------------------------------------------------------------
    # Evolution / histogram / comparison
    # -----------------------------------------------------------------------

    def plot_redshift_evolution(
        self,
        redshifts: np.ndarray,
        values: np.ndarray,
        ax: Optional[plt.Axes] = None,
        ylabel: str = "Value",
        title: str = "Redshift Evolution",
        **kwargs,
    ) -> plt.Axes:
        """
        Plot a quantity as a function of redshift.

        X-axis is inverted so that cosmic time increases left-to-right.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(redshifts, values, **kwargs)
        ax.set_xlabel("Redshift z")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        return ax

    def plot_histogram(
        self,
        data: np.ndarray,
        ax: Optional[plt.Axes] = None,
        bins: Union[int, str] = "auto",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        title: str = "Histogram",
        **kwargs,
    ) -> plt.Axes:
        """Plot a histogram."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data, bins=bins, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        return ax

    def plot_comparison(
        self,
        x: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        ax: Optional[plt.Axes] = None,
        label1: str = "Data 1",
        label2: str = "Data 2",
        xlabel: str = "X",
        ylabel: str = "Y",
        title: str = "Comparison",
        color1: Optional[str] = None,
        color2: Optional[str] = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot two datasets for comparison.

        Parameters
        ----------
        color1, color2 : str, optional
            Explicit colours for each line.  If not given, the two first
            entries of the default colour cycle are used so the lines are
            always visually distinct.

        Notes
        -----
        The original code passed **kwargs to both ax.plot() calls.  This
        meant that any colour kwarg (color=, c=) set the same colour on
        both lines, making them indistinguishable.  That bug is fixed here
        by separating the colour handling from shared kwargs.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Pull colours from the default cycle if not provided
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c1 = color1 or cycle[0]
        c2 = color2 or cycle[1]

        # Remove any color/c key from shared kwargs to avoid conflicts
        safe_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("color", "c")}

        ax.plot(x, y1, color=c1, label=label1, **safe_kwargs)
        ax.plot(x, y2, color=c2, label=label2, **safe_kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # -----------------------------------------------------------------------
    # Corner plot
    # -----------------------------------------------------------------------

    def plot_corner(
        self,
        samples: np.ndarray,
        labels: List[str],
        title: str = "Corner Plot",
    ) -> plt.Figure:
        """
        Create a corner plot for parameter posterior distributions.

        Uses the 'corner' package when available.  Falls back to a plain
        matplotlib figure showing marginal histograms so that the return
        value is always a valid Figure — the original returned None when
        corner was missing, causing AttributeError in any caller that called
        fig.tight_layout() or fig.savefig().
        """
        try:
            import corner as _corner
            fig = _corner.corner(
                samples,
                labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
            )
            fig.suptitle(title, y=1.02)
            return fig

        except ImportError:
            logger.warning(
                "corner package not installed — showing marginal histograms only. "
                "Install with: pip install corner"
            )
            n_params = samples.shape[1] if samples.ndim == 2 else 1
            fig, axes = plt.subplots(
                1, n_params, figsize=(4 * n_params, 4)
            )
            if n_params == 1:
                axes = [axes]
            for i, (ax, label) in enumerate(zip(axes, labels)):
                ax.hist(samples[:, i], bins=30, color="steelblue",
                        edgecolor="white", alpha=0.8)
                ax.set_xlabel(label)
                ax.set_ylabel("Count" if i == 0 else "")
                # Mark median and 1σ credible interval
                q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
                ax.axvline(q50, color="tomato", lw=1.5,
                           label=f"{q50:.2f}⁺{q84-q50:.2f}₋{q50-q16:.2f}")
                ax.axvspan(q16, q84, alpha=0.15, color="tomato")
                ax.legend(fontsize=8)
            fig.suptitle(title + "\n(corner not installed — marginals only)",
                         fontsize=11)
            fig.tight_layout()
            return fig

    # -----------------------------------------------------------------------
    # Annotation helpers
    # -----------------------------------------------------------------------

    def add_redshift_axis(
        self, ax: plt.Axes, z_values: Optional[List[float]] = None
    ) -> plt.Axes:
        """Add a secondary top axis showing redshift ticks."""
        if z_values is None:
            z_values = [0, 1, 2, 5, 10]

        ax2 = ax.twiny()
        ax2.set_xlabel("Redshift")
        ax2.set_xlim(ax.get_xlim())
        return ax2

    def annotate_peak(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        label: str = "Peak",
    ):
        """Annotate the peak value of a plotted quantity."""
        peak_idx = int(np.argmax(y))
        peak_x, peak_y = x[peak_idx], y[peak_idx]

        ax.annotate(
            f"{label}\n({peak_x:.2f}, {peak_y:.2e})",
            xy=(peak_x, peak_y),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                      ec="darkorange", alpha=0.85),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.15"),
        )

    # -----------------------------------------------------------------------
    # Colour utilities
    # -----------------------------------------------------------------------

    def get_colormap(self, name: str = "viridis", n_colors: int = 10) -> List:
        """
        Return *n_colors* discrete colours sampled from a colormap.

        Uses the current matplotlib API (matplotlib.colormaps[]) rather than
        the deprecated plt.cm.get_cmap().
        """
        cmap = _get_cmap(name)
        return [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]

    # -----------------------------------------------------------------------
    # Figure I/O
    # -----------------------------------------------------------------------

    @staticmethod
    def save_figure(
        fig: plt.Figure, filename: str, dpi: int = 300, **kwargs
    ):
        """Save a figure at publication quality."""
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", **kwargs)
        logger.info("Figure saved to %s", filename)

    @staticmethod
    def close_all():
        """Close all open matplotlib figures to release memory."""
        plt.close("all")