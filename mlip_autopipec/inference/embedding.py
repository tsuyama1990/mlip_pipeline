import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList

from mlip_autopipec.config.schemas.inference import EmbeddingConfig
from mlip_autopipec.data_models.inference_models import ExtractedStructure


class EmbeddingExtractor:
    """
    Extracts a local cluster around a focal atom and embeds it in a periodic box.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the extractor.

        Args:
            config: Configuration defining cutoff radii and box size.
        """
        self.config = config

    def _validate_inputs(self, large_atoms: Atoms, center_idx: int) -> None:
        """Validate input structure and index."""
        if not isinstance(large_atoms, Atoms):
            raise TypeError("Input must be an ase.Atoms object.")
        if len(large_atoms) == 0:
            raise ValueError("Input structure is empty.")
        if center_idx < 0 or center_idx >= len(large_atoms):
            raise IndexError(f"Center index {center_idx} out of bounds (0-{len(large_atoms) - 1}).")

    def _get_neighbors(self, large_atoms: Atoms, center_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Find neighbors within cutoff + buffer."""
        cutoff = self.config.core_radius + self.config.buffer_width
        nl = NeighborList(
            [cutoff / 2.0] * len(large_atoms), self_interaction=True, bothways=True, skin=0.0
        )
        nl.update(large_atoms)
        indices, offsets = nl.get_neighbors(center_idx)
        return indices, offsets

    def _build_cluster(
        self, large_atoms: Atoms, center_idx: int, indices: np.ndarray, offsets: np.ndarray
    ) -> Atoms:
        """Construct the cluster Atoms object centered in the new box."""
        center_pos = large_atoms.positions[center_idx]
        cell = large_atoms.get_cell()

        cluster_positions = []
        cluster_symbols = []
        cluster_indices = []

        for i, offset in zip(indices, offsets):
            pos = large_atoms.positions[i]
            # Unwrap: position + offset @ cell
            shifted_pos = pos + np.dot(offset, cell)
            # Relative to center
            rel_pos = shifted_pos - center_pos

            cluster_positions.append(rel_pos)
            cluster_symbols.append(large_atoms.symbols[i])
            cluster_indices.append(i)

        L = self.config.box_size
        box_center = np.array([L / 2.0, L / 2.0, L / 2.0])
        final_positions = np.array(cluster_positions) + box_center

        cluster = Atoms(
            symbols=cluster_symbols, positions=final_positions, cell=[L, L, L], pbc=True
        )
        cluster.new_array("original_index", np.array(cluster_indices))
        return cluster

    def extract(self, large_atoms: Atoms, center_idx: int) -> ExtractedStructure:
        """
        Extracts a local cluster around the atom at center_idx.

        Args:
            large_atoms: The source structure (usually large supercell).
            center_idx: The index of the focal atom.

        Returns:
            ExtractedStructure containing the cluster in a small periodic box.
        """
        try:
            self._validate_inputs(large_atoms, center_idx)

            indices, offsets = self._get_neighbors(large_atoms, center_idx)
            cluster = self._build_cluster(large_atoms, center_idx, indices, offsets)

            origin_uuid = "unknown"
            if large_atoms.info and "uuid" in large_atoms.info:
                origin_uuid = str(large_atoms.info["uuid"])

            return ExtractedStructure(
                atoms=cluster,
                origin_uuid=origin_uuid,
                origin_index=center_idx,
                mask_radius=self.config.core_radius,
            )

        except (IndexError, TypeError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(f"Extraction failed for index {center_idx}: {e!s}") from e
