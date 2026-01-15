# ruff: noqa: D101, T201
"""Utilities for the main workflow."""

import json
import logging
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io.jsonio import decode, encode
from dask.distributed import Client, LocalCluster

from mlip_autopipec.config_schemas import SystemConfig


def setup_dask_client(config: SystemConfig) -> Client:
    """Set up and return a Dask client."""
    if config.dask.scheduler_address:
        logging.info(
            "Connecting to Dask scheduler at: %s", config.dask.scheduler_address
        )
        return Client(config.dask.scheduler_address)  # type: ignore[no-untyped-call]
    logging.info("No Dask scheduler specified, starting a local cluster.")
    cluster = LocalCluster()  # type: ignore[no-untyped-call]
    return Client(cluster)  # type: ignore[no-untyped-call]


def atoms_to_json(atoms: Atoms) -> dict[str, Any]:
    """Serialize an ASE Atoms object to a JSON-compatible dictionary."""
    return json.loads(encode(atoms))  # type: ignore[no-any-return]


def json_to_atoms(data: dict[str, Any]) -> Atoms:
    """Deserialize a JSON dictionary back to an ASE Atoms object."""
    return decode(json.dumps(data))  # type: ignore[no-untyped-call, no-any-return]


class CheckpointManager:
    """Manages saving and loading of the workflow state."""

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path

    def save(self, state: dict[str, Any]) -> None:
        """Save the current workflow state to a checkpoint file."""
        logging.info(f"Saving checkpoint to: {self.path}")
        with open(self.path, "w") as f:
            json.dump(state, f, indent=4)

    def load(self) -> dict[str, Any] | None:
        """Load the workflow state from a checkpoint file if it exists."""
        if not self.path.exists():
            return None
        logging.info(f"Loading checkpoint from: {self.path}")
        with open(self.path) as f:
            return json.load(f)  # type: ignore[no-any-return]
