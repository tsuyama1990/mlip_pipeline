from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ase import Atoms
import torch
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig, RejectionInfo
import os

class MaceClient:
    """
    Wrapper around MACE-MP model for inference.
    """

    def __init__(self, config: SurrogateConfig):
        self.config = config
        self.model = None
        self.device = config.device

    def _load_model(self):
        """Lazy loading of the MACE model."""
        if self.model is None:
            try:
                # Basic security check for model_path to prevent path traversal if it's a file path
                if "/" in self.config.model_path or "\\" in self.config.model_path:
                    # It's a path, check if it exists and is not malicious (basic check)
                    # For strict security, we might want to restrict to specific directories.
                    # But mace accepts paths.
                    if ".." in self.config.model_path:
                         raise ValueError("Path traversal detected in model_path")

                from mace.calculators import mace_mp
                self.model = mace_mp(model=self.config.model_path, device=self.device, default_dtype="float32")

            except (ImportError, Exception) as e:
                # Log error in a real app
                print(f"Warning: Failed to load MACE model: {e}")
                self.model = None

    def predict_forces(self, atoms_list: List[Atoms]) -> List[np.ndarray]:
        """
        Predicts forces for a list of atoms.
        Returns a list of force arrays (N_atoms, 3).
        """
        self._load_model()
        if self.model is None:
             raise RuntimeError("MACE model not loaded.")

        forces_list = []
        # Batching could be improved here using mace's batch capabilities
        # but for now we iterate (or use mace's internal batching if we passed a list,
        # but mace_mp returns a calculator, which is usually single structure).
        # To do batching with mace_mp calculator, we might need to use the underlying model directly.
        # However, for robustness, we stick to ASE interface for now.

        for atoms in atoms_list:
            # We must be careful not to mutate the original atoms' calculator permanently if not desired
            # But usually it's fine.
            atoms.calc = self.model
            try:
                forces = atoms.get_forces()
                forces_list.append(forces)
            except Exception as e:
                # If calculation fails for one structure, we should probably fail it.
                # But we can't return a list of different length easily.
                # Let's raise.
                raise RuntimeError(f"Force calculation failed for structure: {e}")

        return forces_list

    def filter_unphysical(self, atoms_list: List[Atoms]) -> Tuple[List[Atoms], List[RejectionInfo]]:
        """
        Filters out structures with forces exceeding the threshold.
        Returns (kept_atoms, rejected_info).
        """
        threshold = self.config.force_threshold
        kept = []
        rejected = []

        try:
            forces_list = self.predict_forces(atoms_list)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

        for i, (atoms, forces) in enumerate(zip(atoms_list, forces_list)):
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norms)

            if max_force > threshold:
                rejected.append(RejectionInfo(
                    index=i,
                    max_force=float(max_force),
                    reason="force_threshold_exceeded"
                ))
            else:
                kept.append(atoms)

        return kept, rejected
