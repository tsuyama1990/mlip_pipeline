import logging

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.surrogate.mace_wrapper import MaceWrapper
from mlip_autopipec.surrogate.model_interface import ModelInterface
from mlip_autopipec.surrogate.sampling import FarthestPointSampling

logger = logging.getLogger(__name__)

class SurrogatePipeline:
    def __init__(self, db_manager: DatabaseManager, config: SurrogateConfig, model: ModelInterface | None = None):
        self.db_manager = db_manager
        self.config = config
        self.model = model

        # Initialize model if needed
        if self.model is None:
            self.model = MaceWrapper(model_type=self.config.model_type)

    def run(self) -> None:
        """
        Executes the surrogate selection pipeline.
        1. Fetch pending candidates.
        2. Filter unphysical structures.
        3. Select diverse subset.
        4. Update database.
        """
        logger.info("Starting Surrogate Pipeline")
        try:
            entries = self._fetch_pending_entries()
            if not entries:
                return

            ids = [e[0] for e in entries]
            atoms_list = [e[1] for e in entries]

            self._ensure_model_loaded()

            valid_indices, rejected_ids = self._prescreen_candidates(ids, atoms_list)

            if not valid_indices:
                logger.warning("No valid structures remained after filtering.")
                return

            self._select_and_update(valid_indices, ids, atoms_list)

        except Exception:
            logger.error("Surrogate Pipeline failed.", exc_info=True)
            raise

    def _fetch_pending_entries(self) -> list[tuple[int, Atoms]]:
        # Consume generator to list
        entries = list(self.db_manager.get_entries(selection="status=pending"))
        if not entries:
            logger.info("No pending structures found.")
            return []
        logger.info(f"Found {len(entries)} pending structures.")
        return entries

    def _ensure_model_loaded(self) -> None:
        if self.model:
            self.model.load_model(self.config.model_path, self.config.device)

    def _prescreen_candidates(self, ids: list[int], atoms_list: list[Atoms]) -> tuple[list[int], list[int]]:
        logger.info("Computing energy and forces...")

        if not atoms_list:
            return [], []

        # Assume model is loaded
        if not self.model:
             raise RuntimeError("Model not initialized")

        energies, forces_list = self.model.compute_energy_forces(atoms_list)

        # Validate shapes
        if len(energies) != len(atoms_list):
            raise RuntimeError(f"Energy array length {len(energies)} mismatches atoms list {len(atoms_list)}")
        if len(forces_list) != len(atoms_list):
             raise RuntimeError(f"Forces array length {len(forces_list)} mismatches atoms list {len(atoms_list)}")

        valid_indices = []
        rejected_ids = []

        for idx, forces in enumerate(forces_list):
            # Validate forces shape for individual atom
            n_atoms = len(atoms_list[idx])
            if forces.shape != (n_atoms, 3):
                 # Log error but maybe skip structure? Or fail?
                 # If model returns wrong shape, it's critical.
                 raise RuntimeError(f"Forces shape {forces.shape} invalid for {n_atoms} atoms at index {idx}")

            max_force = np.max(np.linalg.norm(forces, axis=1))
            energy = float(energies[idx])

            meta = {
                "mace_energy": energy,
                "mace_max_force": float(max_force)
            }

            current_id = ids[idx]
            self.db_manager.update_metadata(current_id, meta)

            if max_force > self.config.force_threshold:
                logger.debug(f"Rejecting structure {current_id}: Max force {max_force:.2f} > {self.config.force_threshold}")
                rejected_ids.append(current_id)
                self.db_manager.update_status(current_id, "rejected")
            else:
                valid_indices.append(idx)

        logger.info(f"Rejected {len(rejected_ids)} structures due to high forces.")
        return valid_indices, rejected_ids

    def _select_and_update(self, valid_indices: list[int], ids: list[int], atoms_list: list[Atoms]) -> None:
        valid_atoms = [atoms_list[i] for i in valid_indices]
        valid_ids = [ids[i] for i in valid_indices]

        n_samples = min(self.config.n_samples, len(valid_atoms))
        selected_valid_indices = set()

        if n_samples < len(valid_atoms):
            logger.info(f"Selecting {n_samples} structures from {len(valid_atoms)} valid candidates.")
            logger.info("Computing descriptors...")

            if not self.model:
                 raise RuntimeError("Model not initialized")

            descriptors = self.model.compute_descriptors(valid_atoms)
            # Validate descriptors
            if len(descriptors) != len(valid_atoms):
                 raise RuntimeError("Descriptor count mismatch")

            fps = FarthestPointSampling(n_samples=n_samples)
            selected_indices_local = fps.select(descriptors)
            selected_valid_indices = set(selected_indices_local)
        else:
            logger.info(f"Selecting all {len(valid_atoms)} valid candidates (requested {self.config.n_samples}).")
            selected_valid_indices = set(range(len(valid_atoms)))

        selected_count = 0
        held_count = 0

        for i in range(len(valid_atoms)):
            real_id = valid_ids[i]
            if i in selected_valid_indices:
                self.db_manager.update_status(real_id, "selected")
                selected_count += 1
            else:
                self.db_manager.update_status(real_id, "held")
                held_count += 1

        logger.info(f"Pipeline complete. Selected: {selected_count}, Held: {held_count}")
