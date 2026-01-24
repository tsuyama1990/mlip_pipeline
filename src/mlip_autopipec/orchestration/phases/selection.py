import logging

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from mlip_autopipec.orchestration.phases.base import BasePhase
from mlip_autopipec.orchestration.strategies import GammaSelectionStrategy
from mlip_autopipec.training.pacemaker import PacemakerWrapper

logger = logging.getLogger(__name__)

class SelectionPhase(BasePhase):
    def execute(self) -> None:
        """
        Execute Phase: Selection.
        Selects from 'screening' candidates and promotes them to 'pending'.
        """
        logger.info("Phase: Selection")
        try:
            # 1. Load candidates pending screening
            screening_entries = list(self.db.select_entries("status=screening"))
            if not screening_entries:
                logger.info("No candidates in screening.")
                return

            candidates = [atoms for _, atoms in screening_entries]
            ids = [i for i, _ in screening_entries]

            logger.info(f"Screening {len(candidates)} candidates.")

            # 2. Initialize Strategy
            potential_path = self.manager.state.latest_potential_path or (self.manager.work_dir / "current.yace")

            if not self.config.training_config:
                 logger.warning("No Training Config for Selection Strategy.")
                 # Fallback: select all if no training config (can't run pacemaker)
                 selected_indices = range(len(candidates))
            else:
                 pacemaker = PacemakerWrapper(self.config.training_config, self.manager.work_dir)
                 strategy = GammaSelectionStrategy(pacemaker, EmbeddingConfig()) # Using default embedding config

                 # Since GammaSelectionStrategy uses PacemakerWrapper.select_active_set which works on file,
                 # and returns indices relative to the input list.
                 # I can rely on list order preservation.

                 indices = pacemaker.select_active_set(candidates, potential_path)
                 selected_indices = set(indices)

            # 3. Update Status
            selected_count = 0
            for i, db_id in enumerate(ids):
                if i in selected_indices:
                    self.db.update_status(db_id, "pending")
                    selected_count += 1
                else:
                    self.db.update_status(db_id, "rejected")

            logger.info(f"Selection complete. Selected: {selected_count}, Rejected: {len(candidates) - selected_count}")

        except Exception:
            logger.exception("Selection phase failed")
