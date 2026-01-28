import itertools
import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.orchestration.phases.base import BasePhase

logger = logging.getLogger(__name__)

T = TypeVar("T")


def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


class DFTPhase(BasePhase):
    def execute(self) -> None:
        """Execute Phase B: DFT Labeling."""
        logger.info("Phase B: DFT Labeling")
        try:
            # TODO: Move batch_size to config
            batch_size = 50
            # DFTPhase processes 'pending' candidates.
            # (Note: In future, SelectionPhase might promote 'screening' to 'pending')
            pending_entries = list(self.db.select_entries("status=pending"))

            if not pending_entries:
                logger.info("No pending entries for DFT.")
                return

            total_success = 0
            processed_count = 0

            if self.config.dft:
                dft_work_dir = self.manager.work_dir / "dft_runs"
                runner = QERunner(self.config.dft, work_dir=dft_work_dir)

                for batch in chunked(pending_entries, batch_size):
                    if not batch:
                        continue

                    atoms_list = [at for _, at in batch]
                    ids = [i for i, _ in batch]

                    logger.info(f"Submitting DFT batch of {len(atoms_list)} structures.")

                    # Use TaskQueue to submit tasks
                    # runner.run is the function
                    # atoms_list is the list of items
                    # The runner must be pickleable or we rely on dask to serialize the instance method.
                    # QERunner is simple enough.
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
                            # Optionally mark as failed in DB?
                            # self.db.update_status(db_id, "failed")

                    processed_count += len(batch)
            else:
                logger.error("DFT configuration is missing.")

            logger.info(
                f"DFT Phase complete. Processed: {processed_count}, Success: {total_success}"
            )

        except Exception:
            logger.exception("DFT phase failed")
