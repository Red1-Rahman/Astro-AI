"""
Data Handler Module for Astro-AI
Handles data loading, saving, and management for astronomical data.

Fixes vs original:
  1. h5py is an optional dependency — original code imported it at module
     level, crashing the entire app on import if h5py is not installed
     (common on Streamlit Cloud where the package may be absent).
     HDF5 methods now import h5py lazily and raise ImportError with a clear
     message when it is missing.

  2. save_hdf5() silently dropped any value that was not an ndarray/list/
     tuple/scalar — dicts, DataFrames, and strings longer than a scalar
     were discarded without warning.  The method now logs a warning for
     skipped keys so the caller knows data was not persisted.

  3. save_json() used json.dump() with no fallback for non-serialisable
     types (numpy scalars, numpy arrays, pandas objects).  On astronomical
     data this raises TypeError silently swallowed by callers.  A custom
     NumpyEncoder is now used so numpy scalars and small arrays serialise
     correctly.

  4. load_pickle() had no protection against loading untrusted files from
     arbitrary paths.  A path-traversal guard is added so filenames with
     ".." components are rejected.

  5. clean_data() modified a copy but the docstring said it replaced values
     in-place.  Behaviour is unchanged (returns a copy) but the docstring
     is corrected and the method handles both finite and non-finite edge
     cases more robustly.
"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON encoder that handles numpy types
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that converts numpy scalars and small arrays to Python
    native types so that astronomical result dicts serialise without error.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            if obj.size <= 1000:
                return obj.tolist()
            return f"<array shape={obj.shape}>"
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="list")
            if isinstance(obj, pd.Series):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


# ---------------------------------------------------------------------------
# DataHandler
# ---------------------------------------------------------------------------

class DataHandler:
    """
    Handles data I/O operations for various astronomical data formats
    including FITS, HDF5, CSV, JSON, and pickle.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Parameters
        ----------
        data_dir : str
            Base directory for data storage.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Path helpers
    # -----------------------------------------------------------------------

    def _safe_path(self, filename: str) -> Path:
        """
        Resolve *filename* relative to data_dir and reject path traversal.

        Raises
        ------
        ValueError
            If the resolved path escapes data_dir (e.g. "../../etc/passwd").
        """
        resolved = (self.data_dir / filename).resolve()
        if not str(resolved).startswith(str(self.data_dir.resolve())):
            raise ValueError(
                f"Path traversal detected: '{filename}' resolves outside data_dir."
            )
        return resolved

    # -----------------------------------------------------------------------
    # HDF5
    # -----------------------------------------------------------------------

    def save_hdf5(
        self, data: Dict[str, Any], filename: str, overwrite: bool = True
    ) -> Path:
        """
        Save data to HDF5 format.

        Parameters
        ----------
        data : dict
            Keys become dataset names (ndarray/list/tuple) or file attributes
            (int/float/str).  Other types are skipped with a warning.
        filename : str
            Output filename (relative to data_dir).
        overwrite : bool
            If False, raise FileExistsError when the file already exists.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 support. "
                "Install it with: pip install h5py"
            )

        filepath = self._safe_path(filename)

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filepath} already exists.")

        with h5py.File(filepath, "w") as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    try:
                        f.create_dataset(key, data=np.asarray(value))
                    except Exception as e:
                        logger.warning(
                            "HDF5 save: could not store '%s' as dataset: %s", key, e
                        )
                elif isinstance(value, (int, float, bool, str)):
                    f.attrs[key] = value
                elif isinstance(value, np.integer):
                    f.attrs[key] = int(value)
                elif isinstance(value, np.floating):
                    f.attrs[key] = float(value)
                else:
                    logger.warning(
                        "HDF5 save: skipping key '%s' — unsupported type %s. "
                        "Convert to ndarray or scalar before saving.",
                        key, type(value).__name__,
                    )

        return filepath

    def load_hdf5(self, filename: str) -> Dict[str, Any]:
        """
        Load data from HDF5 format.

        Parameters
        ----------
        filename : str
            Input filename (relative to data_dir).

        Returns
        -------
        dict
            Datasets loaded as numpy arrays; attributes as Python scalars.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 support. "
                "Install it with: pip install h5py"
            )

        filepath = self._safe_path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found.")

        data: Dict[str, Any] = {}
        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                data[key] = f[key][:]
            for key in f.attrs.keys():
                data[key] = f.attrs[key]

        return data

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------

    def save_csv(
        self, data: Union[pd.DataFrame, Dict], filename: str
    ) -> Path:
        """
        Save data to CSV format.

        Parameters
        ----------
        data : pd.DataFrame or dict
            Data to save.  Dicts are converted to DataFrame first.
        filename : str
            Output filename.

        Returns
        -------
        Path
        """
        filepath = self._safe_path(filename)

        if isinstance(data, dict):
            data = pd.DataFrame(data)

        data.to_csv(filepath, index=False)
        return filepath

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV format.

        Parameters
        ----------
        filename : str
            Input filename.

        Returns
        -------
        pd.DataFrame
        """
        filepath = self._safe_path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found.")

        return pd.read_csv(filepath)

    # -----------------------------------------------------------------------
    # Pickle
    # -----------------------------------------------------------------------

    def save_pickle(self, data: Any, filename: str) -> Path:
        """
        Save data using pickle.

        Parameters
        ----------
        data : Any
            Python object to serialise.
        filename : str
            Output filename.

        Returns
        -------
        Path
        """
        filepath = self._safe_path(filename)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        return filepath

    def load_pickle(self, filename: str) -> Any:
        """
        Load data from a pickle file.

        Parameters
        ----------
        filename : str
            Input filename — must reside inside data_dir (path traversal
            is rejected).

        Returns
        -------
        Any
        """
        # Path-traversal guard applied inside _safe_path
        filepath = self._safe_path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found.")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    # -----------------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------------

    def save_json(self, data: Dict, filename: str) -> Path:
        """
        Save data to JSON format.

        Numpy scalars, small arrays, and pandas objects are automatically
        converted to JSON-compatible types via _NumpyEncoder.

        Parameters
        ----------
        data : dict
            Dictionary to save.
        filename : str
            Output filename.

        Returns
        -------
        Path
        """
        filepath = self._safe_path(filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

        return filepath

    def load_json(self, filename: str) -> Dict:
        """
        Load data from JSON format.

        Parameters
        ----------
        filename : str
            Input filename.

        Returns
        -------
        dict
        """
        filepath = self._safe_path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found.")

        with open(filepath, "r") as f:
            return json.load(f)

    # -----------------------------------------------------------------------
    # Directory utilities
    # -----------------------------------------------------------------------

    def list_files(self, pattern: str = "*") -> list:
        """
        List files in data_dir matching a glob pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern (e.g. "*.csv", "*.h5").

        Returns
        -------
        list[Path]
        """
        return sorted(self.data_dir.glob(pattern))

    def delete_file(self, filename: str) -> bool:
        """
        Delete a file from data_dir.

        Parameters
        ----------
        filename : str
            File to delete.

        Returns
        -------
        bool
            True if the file was deleted, False if it did not exist.
        """
        filepath = self._safe_path(filename)

        if filepath.exists():
            filepath.unlink()
            return True
        return False

    # -----------------------------------------------------------------------
    # Array validation / cleaning
    # -----------------------------------------------------------------------

    @staticmethod
    def validate_data(
        data: np.ndarray, expected_shape: Optional[tuple] = None
    ) -> bool:
        """
        Validate a numpy array.

        Parameters
        ----------
        data : np.ndarray
            Array to validate.
        expected_shape : tuple, optional
            If given, the array must match this shape exactly.

        Returns
        -------
        bool
            True if valid (shape matches and no NaN/inf are present as hard
            failures; NaN/inf trigger a warning but still return True so that
            callers can decide whether to clean or reject).
        """
        if not isinstance(data, np.ndarray):
            return False

        if np.any(np.isnan(data)):
            warnings.warn(
                f"Data contains {np.isnan(data).sum()} NaN value(s).",
                stacklevel=2,
            )

        if np.any(np.isinf(data)):
            warnings.warn(
                f"Data contains {np.isinf(data).sum()} infinite value(s).",
                stacklevel=2,
            )

        if expected_shape is not None and data.shape != expected_shape:
            return False

        return True

    @staticmethod
    def clean_data(
        data: np.ndarray, fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Return a copy of *data* with NaN and ±inf replaced by *fill_value*.

        The original array is never modified.

        Parameters
        ----------
        data : np.ndarray
            Input array.
        fill_value : float
            Replacement value for non-finite elements.

        Returns
        -------
        np.ndarray
            Cleaned copy.
        """
        cleaned = np.array(data, dtype=float, copy=True)
        cleaned[~np.isfinite(cleaned)] = fill_value
        return cleaned