import logging
import os

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import RejectionInfo, SurrogateConfig

logger = logging.getLogger(__name__)


class MaceClient:
    """
    Wrapper around MACE-MP model for inference.

    This class handles the loading of the MACE model and execution of force predictions.
    It provides functionality to filter structures based on physical plausibility (force thresholds).
    """

    def __init__(self, config: SurrogateConfig):
        """
        Initializes the MaceClient.

        Args:
            config: A SurrogateConfig object containing model parameters and thresholds.
        """
        self.config = config
        self.model = None
        self.device = config.device

    def _load_model(self) -> None:
        """
        Lazy loads the MACE model.

        Raises:
            ValueError: If a potential path traversal attack is detected in model_path.
        """
        if self.model is None:
            model_path = self.config.model_path

            # Security Check: Path Traversal
            # If model_path looks like a file path (contains separator), validate it.
            if os.path.sep in model_path or (os.path.altsep and os.path.altsep in model_path):
                # Resolve absolute path
                abs_path = os.path.abspath(model_path)
                # Check if it tries to go up levels inappropriately or access sensitive dirs.
                if ".." in model_path:
                     raise ValueError("Path traversal attempt detected in model_path.")

            try:
                from mace.calculators import mace_mp
                # default_dtype="float32" is used to save memory/speed as per standard MACE usage
                self.model = mace_mp(model=model_path, device=self.device, default_dtype="float32")

            except (ImportError, Exception) as e:
                if isinstance(e, ValueError) and "Path traversal" in str(e):
                    raise e

                logger.warning(f"Failed to load MACE model: {e}")
                self.model = None

    def predict_forces(self, atoms_list: list[Atoms]) -> list[np.ndarray]:
        """
        Predicts forces for a batch of atomic structures.

        Args:
            atoms_list: A list of ase.Atoms objects.

        Returns:
            A list of numpy arrays, where each array has shape (N_atoms, 3) representing forces.

        Raises:
            RuntimeError: If the model is not loaded or prediction fails.
        """
        self._load_model()
        if self.model is None:
             raise RuntimeError("MACE model could not be loaded.")

        forces_list: list[np.ndarray] = []

        # Note: In a production environment with MACE, we would use batch processing
        # features of the model if available for performance.
        # Here we iterate to ensure ASE interface compatibility.
        for atoms in atoms_list:
            # Temporarily attach calculator
            original_calc = atoms.calc
            atoms.calc = self.model
            try:
                # get_forces returns np.ndarray
                forces = atoms.get_forces()
                forces_list.append(forces)
            except Exception as e:
                # Log full exception
                logger.exception(f"Force calculation failed for structure {atoms}")
                raise RuntimeError(f"Force calculation failed for structure {atoms}: {e}") from e
            finally:
                # Restore original calculator
                atoms.calc = original_calc

        return forces_list

    def filter_unphysical(self, atoms_list: list[Atoms]) -> tuple[list[Atoms], list[RejectionInfo]]:
        """
        Filters out structures that exhibit unphysical forces (exceeding the configured threshold).

        Args:
            atoms_list: A list of ase.Atoms objects to screen.

        Returns:
            A tuple containing:
                - kept_atoms: List of Atoms objects that passed the filter.
                - rejected_info: List of RejectionInfo objects detailing why structures were dropped.

        Raises:
            RuntimeError: If force prediction fails.
        """
        threshold = self.config.force_threshold
        kept: list[Atoms] = []
        rejected: list[RejectionInfo] = []

        try:
            forces_list = self.predict_forces(atoms_list)
        except Exception as e:
            # Re-raise with context
            logger.exception("Pre-screening prediction failed")
            raise RuntimeError(f"Pre-screening prediction failed: {e}") from e

        for i, (atoms, forces) in enumerate(zip(atoms_list, forces_list)):
            # Robustness check: Ensure forces array is valid
            if forces is None:
                max_force = float('inf')
            elif forces.size == 0:
                max_force = 0.0
            else:
                force_norms = np.linalg.norm(forces, axis=1)
                max_force = float(np.max(force_norms))

            if max_force > threshold:
                rejected.append(RejectionInfo(
                    index=i,
                    max_force=max_force,
                    reason="force_threshold_exceeded"
                ))
            else:
                kept.append(atoms)

        return kept, rejected
