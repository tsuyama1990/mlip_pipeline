import logging

import numpy as np

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.surrogate.mace_wrapper import MaceWrapper
from mlip_autopipec.surrogate.model_interface import ModelInterface
from mlip_autopipec.surrogate.sampling import FarthestPointSampling

logger = logging.getLogger(__name__)

class SurrogatePipeline:
    def __init__(self, db_manager: DatabaseManager, config: SurrogateConfig, model: ModelInterface | None = None):
        self.db_manager = db_manager
        self.config = config
        self.model = model

    def run(self) -> None:
        """
        Executes the surrogate selection pipeline.
        1. Fetch pending candidates.
        2. Filter unphysical structures.
        3. Select diverse subset.
        4. Update database.
        """
        logger.info("Starting Surrogate Pipeline")

        # 1. Fetch
        # We need IDs to update status later
        entries = self.db_manager.get_entries(selection="status=pending")
        if not entries:
            logger.info("No pending structures found.")
            return

        logger.info(f"Found {len(entries)} pending structures.")

        ids = [e[0] for e in entries]
        atoms_list = [e[1] for e in entries]

        # Initialize model if needed
        if self.model is None:
            self.model = MaceWrapper(model_type=self.config.model_type)

        self.model.load_model(self.config.model_path, self.config.device)

        # 2. Pre-screen & Filter
        logger.info("Computing energy and forces...")
        energies, forces_list = self.model.compute_energy_forces(atoms_list)

        valid_indices = []
        rejected_ids = []

        for idx, forces in enumerate(forces_list):
            max_force = np.max(np.linalg.norm(forces, axis=1))
            energy = float(energies[idx])

            # Prepare metadata to save
            meta = {
                "mace_energy": energy,
                "mace_max_force": float(max_force)
            }

            current_id = ids[idx]

            if max_force > self.config.force_threshold:
                logger.debug(f"Rejecting structure {current_id}: Max force {max_force:.2f} > {self.config.force_threshold}")
                rejected_ids.append(current_id)
                # Update status and metadata
                self.db_manager.update_status(current_id, "rejected")
                self.db_manager.update_metadata(current_id, meta)
            else:
                valid_indices.append(idx)
                # Store metadata for valid ones too
                self.db_manager.update_metadata(current_id, meta)

        logger.info(f"Rejected {len(rejected_ids)} structures due to high forces.")

        if not valid_indices:
            logger.warning("No valid structures remained after filtering.")
            return

        # Prepare valid subset
        valid_atoms = [atoms_list[i] for i in valid_indices]
        valid_ids = [ids[i] for i in valid_indices]

        # 3. Sampling
        n_samples = min(self.config.n_samples, len(valid_atoms))

        selected_valid_indices: set[int] = set()

        if n_samples < len(valid_atoms):
            logger.info(f"Selecting {n_samples} structures from {len(valid_atoms)} valid candidates.")
            logger.info("Computing descriptors...")
            descriptors = self.model.compute_descriptors(valid_atoms)

            fps = FarthestPointSampling(n_samples=n_samples)
            selected_indices_local = fps.select(descriptors)
            selected_valid_indices = set(selected_indices_local)
        else:
            # Select all if we have fewer candidates than requested samples
            logger.info(f"Selecting all {len(valid_atoms)} valid candidates (requested {self.config.n_samples}).")
            selected_valid_indices = set(range(len(valid_atoms)))

        # 4. Update
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

        logger.info(f"Pipeline complete. Selected: {selected_count}, Held: {held_count}, Rejected: {len(rejected_ids)}")
