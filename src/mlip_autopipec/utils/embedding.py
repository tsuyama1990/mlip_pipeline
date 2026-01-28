import logging

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extracts embeddings/descriptors from atoms.
    """

    def __init__(self):
        pass

    def extract(self, atoms: Atoms) -> np.ndarray:
        if not isinstance(atoms, Atoms):
            raise TypeError("Expected Atoms object")
        return np.zeros((len(atoms), 10))
