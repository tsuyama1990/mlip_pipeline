import logging
from typing import Any

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.common import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extracts embeddings/descriptors from atoms.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config

    def extract(self, atoms: Atoms, center_idx: int | None = None) -> Atoms:
        """
        Extracts a cluster centered at center_idx or just returns the atoms.

        Note: The implementation here is a placeholder. Real implementation
        would cut a cluster.

        Args:
            atoms: The structure.
            center_idx: Index of the central atom.

        Returns:
            Atoms: The extracted cluster (embedded).
        """
        if not isinstance(atoms, Atoms):
            msg = "Expected Atoms object"
            raise TypeError(msg)

        # Placeholder: Return the original atoms as "embedded" for now
        # In reality, this would cut a cluster.
        return atoms
