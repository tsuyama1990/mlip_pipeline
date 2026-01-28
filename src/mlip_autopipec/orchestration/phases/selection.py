import logging
import tempfile
from pathlib import Path

from ase.io import write

from mlip_autopipec.orchestration.phases.base import BasePhase
from mlip_autopipec.surrogate.candidate_manager import CandidateManager
from mlip_autopipec.training.pacemaker import PacemakerWrapper

logger = logging.getLogger(__name__)


class SelectionPhase(BasePhase):
    def execute(self) -> None:
        """
        Execute Phase: Selection.
        Processes halted structures from Exploration phase OR selects from 'screening' candidates.
        """
        logger.info("Phase: Selection")
        try:
            potential_path = self.manager.state.latest_potential_path
            if not potential_path:
                logger.info("No potential available. Skipping active selection.")
                return

            # A. Process Halted Structures (Cycle 04+)
            halted = self.manager.state.halted_structures
            if halted:
                logger.info(f"Processing {len(halted)} halted structures from Exploration.")

                work_dir = self.manager.work_dir / f"selection_gen_{self.manager.state.cycle_index}"
                work_dir.mkdir(parents=True, exist_ok=True)

                if self.config.training_config:
                    pacemaker = PacemakerWrapper(self.config.training_config, work_dir)
                    manager = CandidateManager(self.db, pacemaker)

                    for dump_path in halted:
                        manager.process_halted(
                            dump_path,
                            potential_path,
                            n_perturbations=10,
                            cycle_index=self.manager.state.cycle_index
                        )

                    # Clear halted structures after processing
                    self.manager.state.halted_structures = []
                    self.manager.save_state()
                else:
                    logger.warning("No Training Config, cannot perform active set selection on halted structures.")

            # B. Legacy / Fallback: Screening entries in DB
            screening_entries = list(self.db.select_entries("status=screening"))
            if screening_entries:
                candidates = [atoms for _, atoms in screening_entries]
                ids = [i for i, _ in screening_entries]

                logger.info(f"Screening {len(candidates)} candidates.")

                if not self.config.training_config:
                    # Fallback: select all if no training config
                    selected_indices = set(range(len(candidates)))
                else:
                    work_dir = self.manager.work_dir / f"selection_gen_{self.manager.state.cycle_index}"
                    work_dir.mkdir(parents=True, exist_ok=True)
                    pacemaker = PacemakerWrapper(self.config.training_config, work_dir)

                    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
                        tmp_path = Path(tmp.name)

                    try:
                        # Write candidates to file
                        # type: ignore[no-untyped-call]
                        write(str(tmp_path), candidates, format="extxyz")

                        indices = pacemaker.select_active_set(tmp_path, potential_path)
                        selected_indices = set(indices)
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()

                # Update Status
                selected_count = 0
                for i, db_id in enumerate(ids):
                    if i in selected_indices:
                        self.db.update_status(db_id, "pending")
                        selected_count += 1
                    else:
                        self.db.update_status(db_id, "rejected")

                logger.info(
                    f"Selection complete. Selected: {selected_count}, Rejected: {len(candidates) - selected_count}"
                )

        except Exception:
            logger.exception("Selection phase failed")
