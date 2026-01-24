"""
Module for extracting local atomic environments (candidates) from larger structures.
Used in active learning to isolate uncertain regions.
"""

import logging

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from mlip_autopipec.config.schemas.common import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extracts a cluster of atoms around a central atom (presumably high uncertainty).
    It handles periodic boundary conditions by unwrapping neighbors.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """
        Initialize the extractor.

        Args:
            config: Configuration for embedding extraction (cutoff, etc.)
        """
        self.config = config

    def extract(self, large_atoms: Atoms, center_idx: int) -> Atoms:
        """
        Extract a local environment around the atom at center_idx.

        Args:
            large_atoms: The parent structure (must have PBC and cell).
            center_idx: The index of the central atom.

        Returns:
            A new Atoms object representing the cluster, centered at (0,0,0) with vacuum padding.
        """
        self._validate_input(large_atoms, center_idx)

        # 1. Get neighbors
        indices, offsets = self._get_neighbors(large_atoms, center_idx)

        # 2. Build cluster
        return self._build_cluster(large_atoms, center_idx, indices, offsets)

    def _validate_input(self, large_atoms: Atoms, center_idx: int) -> None:
        """Validate input structure and index."""
        if not isinstance(large_atoms, Atoms):
            msg = "Input must be an ase.Atoms object."
            raise TypeError(msg)
        if len(large_atoms) == 0:
            msg = "Input structure is empty."
            raise ValueError(msg)
        if center_idx < 0 or center_idx >= len(large_atoms):
            msg = f"Center index {center_idx} out of bounds (0-{len(large_atoms) - 1})."
            raise IndexError(msg)

    def _get_neighbors(self, large_atoms: Atoms, center_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Find neighbors within cutoff.
        Returns indices and cell offsets.
        """
        # We use ase.neighborlist.neighbor_list
        # "i" is first atom indices, "j" is second atom indices.
        # We want neighbors of center_idx.

        # Mapping config to cutoff
        # We want to extract everything within core_radius + buffer_width
        cutoff = self.config.core_radius + self.config.buffer_width

        # We can optimize by only computing for the specific index if we use a NeighborList object,
        # but neighbor_list primitive is often fast enough for single frame or we filter.
        # For efficiency with many atoms, building a full list might be slow.
        # Let's use strict filtering.

        # Optimized approach: Use primitive neighbor list but filter
        # "S" -> shift vectors (number of cell vectors added)
        i_arr, j_arr, s_arr = neighbor_list("ijS", large_atoms, cutoff)
        mask = i_arr == center_idx
        neighbors_indices = j_arr[mask]
        neighbor_offsets = s_arr[mask]

        return neighbors_indices, neighbor_offsets

    def _build_cluster(
        self, large_atoms: Atoms, center_idx: int, indices: np.ndarray, offsets: np.ndarray
    ) -> Atoms:
        """
        Construct the cluster Atoms object.
        """
        cell = large_atoms.get_cell()
        center_pos = large_atoms.positions[center_idx]

        positions = []
        symbols = []
        # Add center atom first
        positions.append(np.array([0.0, 0.0, 0.0]))
        symbols.append(large_atoms.get_chemical_symbols()[center_idx])

        for i, offset in zip(indices, offsets, strict=True):
            # Unwrap: position + offset @ cell
            # offset is integers (n1, n2, n3). cell is 3x3.
            # dot(offset, cell) gives the Cartesian shift vector.
            pos = large_atoms.positions[i]
            shifted_pos = pos + np.dot(offset, cell)
            # Relative to center
            rel_pos = shifted_pos - center_pos
            positions.append(rel_pos)
            symbols.append(large_atoms.get_chemical_symbols()[i])

        # Create new Atoms
        cluster = Atoms(
            symbols=symbols,
            positions=positions,
            pbc=False,  # Clusters are non-periodic
        )

        # Add vacuum padding
        cluster.center(vacuum=self.config.buffer_width + 5.0)

        return cluster
