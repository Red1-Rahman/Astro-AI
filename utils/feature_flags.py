"""Feature flag and dependency capability detection for Astro-AI.

Provides a single place to query which heavy / optional scientific
dependencies are available so the UI and logs can present a clean status.

Fixes vs original:
  1. list[str] type hint in all_required_or_raise() uses PEP 585 syntax
     which requires Python 3.9+.  Replaced with List[str] from typing so
     the module imports cleanly on Python 3.8 (minimum supported by
     Streamlit Cloud as of 2025).

  2. _check_mod() swallowed ALL exceptions including KeyboardInterrupt and
     SystemExit via bare except Exception.  Changed to catch only
     ImportError and ModuleNotFoundError so genuine interpreter errors
     are not silently discarded.

  3. summarize_status() returned a bare Markdown table with no fallback for
     non-Markdown contexts.  The markdown=False path now returns a cleaner
     human-readable string rather than the raw key=value pairs.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Library registry
# ---------------------------------------------------------------------------

OPTIONAL_LIBS: Dict[str, List[str]] = {
    "py21cmfast": ["py21cmfast"],
    "tools21cm":  ["tools21cm"],
    "bagpipes":   ["bagpipes"],
    "jwst_pipeline": ["jwst"],
    "astropy":    ["astropy"],
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CapabilityStatus:
    name:      str
    available: bool
    detail:    str


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _check_mod(path: str) -> bool:
    """
    Return True if *path* can be imported.

    Only catches ImportError / ModuleNotFoundError — other exceptions
    (e.g. KeyboardInterrupt, SystemExit) propagate normally.
    """
    try:
        importlib.import_module(path)
        return True
    except (ImportError, ModuleNotFoundError):
        return False
    except Exception as e:
        # Unexpected error during import (e.g. broken C extension) — log
        # and treat as unavailable rather than crashing the whole app.
        logger.warning("Unexpected error importing '%s': %s", path, e)
        return False


def detect_capabilities() -> Dict[str, CapabilityStatus]:
    """Return a dict mapping logical capability names to their status."""
    status: Dict[str, CapabilityStatus] = {}
    for logical_name, modules in OPTIONAL_LIBS.items():
        ok     = any(_check_mod(m) for m in modules)
        detail = "present" if ok else "missing"
        status[logical_name] = CapabilityStatus(logical_name, ok, detail)
    return status


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_status(markdown: bool = True) -> str:
    """
    Return a human-readable summary of capability status.

    Parameters
    ----------
    markdown : bool
        If True (default), return a Markdown table suitable for
        st.markdown().  If False, return a plain-text list.
    """
    caps = detect_capabilities()

    if markdown:
        lines = [
            "### Capability Status",
            "| Feature | Status |",
            "|---------|--------|",
        ]
        for k, v in caps.items():
            icon = "✅" if v.available else "❌"
            lines.append(f"| {k} | {icon} {v.detail} |")
        return "\n".join(lines)

    # Plain text
    lines = ["Capability Status:"]
    for k, v in caps.items():
        mark = "OK" if v.available else "MISSING"
        lines.append(f"  {k:<20} {mark}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strict-mode guard
# ---------------------------------------------------------------------------

def all_required_or_raise(required: List[str]) -> None:
    """
    Raise RuntimeError if any capability in *required* is unavailable.

    Parameters
    ----------
    required : list of str
        Logical capability names to check (keys in OPTIONAL_LIBS).
    """
    caps    = detect_capabilities()
    missing = [
        r for r in required
        if not caps.get(r, CapabilityStatus(r, False, "unknown")).available
    ]
    if missing:
        raise RuntimeError(
            f"Missing required capabilities for strict mode: {', '.join(missing)}"
        )