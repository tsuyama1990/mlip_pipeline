"""
This module provides utility functions for loading data files.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_sssp_data(sssp_data_path: Path) -> Dict[str, Any]:
    """
    Loads the SSSP pseudopotential data from a JSON file.

    This function is a dedicated data loader for the Standard Solid State
    Pseudopotentials (SSSP) library data, which is used to determine
    heuristic DFT parameters like plane-wave cutoffs.

    Args:
        sssp_data_path: The path to the JSON file containing the SSSP data.

    Returns:
        A dictionary containing the SSSP data, keyed by chemical element.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON file.
    """
    try:
        with sssp_data_path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"SSSP data file not found at: {sssp_data_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding SSSP data file: {sssp_data_path}")
        raise
