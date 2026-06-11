#!/usr/bin/env python3
"""
Setup script for bagpipes to work in Streamlit Cloud environment.
Configures bagpipes to use a writable data directory.
"""

import os
import tempfile
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_bagpipes_environment():
    """
    Configure bagpipes to use a writable directory for data files.
    This prevents PermissionError in read-only Streamlit Cloud environment.
    """
    try:
        # Create a writable directory for bagpipes data
        bagpipes_data_dir = os.path.join(tempfile.gettempdir(), "bagpipes_data")
        os.makedirs(bagpipes_data_dir, exist_ok=True)

        # Create subdirectories that bagpipes expects
        grids_dir = os.path.join(bagpipes_data_dir, "grids")
        filters_dir = os.path.join(bagpipes_data_dir, "filters")
        os.makedirs(grids_dir, exist_ok=True)
        os.makedirs(filters_dir, exist_ok=True)

        # Set environment variables for bagpipes data directory
        os.environ["BAGPIPES_FILTERS"] = bagpipes_data_dir
        os.environ["BAGPIPES_DATA"] = bagpipes_data_dir

        # Try to patch bagpipes config before import
        try:
            # Monkey patch the bagpipes config to use our directory
            import types

            # Create a mock config module to override bagpipes.config
            config_module = types.ModuleType("bagpipes.config")
            config_module.BAGPIPES_DIR = bagpipes_data_dir
            config_module.filters_dir = filters_dir
            config_module.grid_dir = grids_dir

            # Set commonly used paths
            config_module.bagpipes_dir = bagpipes_data_dir

            # Add to sys.modules before bagpipes is imported
            sys.modules["bagpipes.config"] = config_module

            logger.info(f"🔧 Patched bagpipes config to use: {bagpipes_data_dir}")

        except Exception as patch_error:
            logger.warning(f"⚠️ Config patching failed: {patch_error}")

        logger.info(f"✅ Bagpipes data directory configured: {bagpipes_data_dir}")
        logger.info(f"✅ Bagpipes grids directory: {grids_dir}")
        logger.info(f"✅ Bagpipes filters directory: {filters_dir}")

        return bagpipes_data_dir

    except Exception as e:
        logger.error(f"❌ Failed to setup bagpipes: {e}")
        return None


if __name__ == "__main__":
    setup_bagpipes_environment()
