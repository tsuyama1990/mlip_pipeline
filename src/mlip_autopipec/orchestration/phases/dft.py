import logging

from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.orchestration.phases.base import BasePhase
from mlip_autopipec.orchestration.phases.exploration import chunked

logger = logging.getLogger(__name__)


class DFTPhase(BasePhase):
    def execute(self) -> None:
        """Execute Phase B: DFT Labeling."""
        logger.info("Phase B: DFT Labeling")
        try:
            # TODO: Move batch_size to config
            batch_size = 50
            pending_entries = self.db.select_entries("status=pending")

            total_success = 0
            processed_count = 0

            if self.config.dft_config:
                runner = QERunner(self.config.dft_config)

                for batch in chunked(pending_entries, batch_size):
                    if not batch:
                        continue

                    atoms_list = [at for _, at in batch]
                    ids = [i for i, _ in batch]

                    logger.info(f"Submitting DFT batch of {len(atoms_list)} structures.")

                    futures = self.queue.submit_dft_batch(runner.run, atoms_list)
                    results = self.queue.wait_for_completion(futures)

                    for atoms, db_id, res in zip(atoms_list, ids, results, strict=True):
                        if res:
                            try:
                                self.db.save_dft_result(
                                    atoms,
                                    res,
                                    {
                                        "status": "training",
                                        "generation": self.manager.state.cycle_index,
                                    },
                                )
                                self.db.update_status(db_id, "labeled")
                                total_success += 1
                            except Exception:
                                logger.exception(f"Failed to save DFT result for ID {db_id}")
                        else:
                            logger.warning(f"DFT failed for atom ID {db_id}.")

                    processed_count += len(batch)

            logger.info(
                f"DFT Phase complete. Processed: {processed_count}, Success: {total_success}"
            )

        except Exception:
            logger.exception("DFT phase failed")
