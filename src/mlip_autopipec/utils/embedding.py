import logging
from typing import Any

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extracts embeddings/descriptors from atoms.
    """

    def __init__(self, config: Any = None):
        self.config = config

    def extract(self, atoms: Atoms) -> np.ndarray:
        if not isinstance(atoms, Atoms):
            msg = "Expected Atoms object"
            raise TypeError(msg)
        return np.zeros((len(atoms), 10))
