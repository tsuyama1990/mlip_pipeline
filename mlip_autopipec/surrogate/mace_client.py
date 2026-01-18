from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ase import Atoms
import torch
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig

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
            # In a real implementation, we would load the model here.
            # from mace.calculators import MACECalculator
            # self.model = MACECalculator(model_paths=self.config.model_path, device=self.device)
            # But for now, since we mock it in tests and might not have the model file,
            # we will assume it's handled or mocked.
            # If we are running in an environment with mace installed, we try to load.
            try:
                # Attempt to import to check if available, but actual loading requires model file.
                # The SPEC says "It loads the model (lazily...)".
                # For this exercise, we'll assume the mace-torch package provides the loading mechanism.
                # But we don't have a model file "medium". MACE usually downloads from cache.
                # To be safe for the environment without internet or huge downloads,
                # we should probably handle the case where it fails.

                # However, for the purpose of "Implementation", I should write the code that WOULD load it.
                from mace.calculators import mace_mp
                # mace_mp is a function returning the calculator
                # It accepts model="medium", device=...
                self.model = mace_mp(model=self.config.model_path, device=self.device, default_dtype="float32")

            except (ImportError, Exception) as e:
                # Fallback or re-raise if strictly required.
                # For now, print warning and maybe allow mock injection.
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

        # Batch prediction
        # MACE calculator can handle list of atoms?
        # mace_mp returns a calculator that can be attached to Atoms.
        # But for batching, we usually need to use the underlying model or loop.
        # ase Calculator is per atoms.

        # If we use the ASE interface:
        forces_list = []
        for atoms in atoms_list:
            atoms.calc = self.model
            forces_list.append(atoms.get_forces())

        return forces_list

    def filter_unphysical(self, atoms_list: List[Atoms]) -> Tuple[List[Atoms], List[Dict[str, Any]]]:
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
            # If prediction fails, we might want to fail safe or raise.
            raise RuntimeError(f"Prediction failed: {e}")

        for i, (atoms, forces) in enumerate(zip(atoms_list, forces_list)):
            # Check max force norm
            # forces shape (N, 3)
            # norm shape (N,)
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norms)

            if max_force > threshold:
                rejected.append({
                    "index": i,
                    "max_force": float(max_force),
                    "reason": "force_threshold_exceeded"
                })
            else:
                kept.append(atoms)

        return kept, rejected
