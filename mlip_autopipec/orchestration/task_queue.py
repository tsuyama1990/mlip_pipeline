import logging
from collections.abc import Callable
from typing import Any

from dask.distributed import Client, Future, LocalCluster, wait  # type: ignore

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Manages Dask distributed tasks for the orchestration workflow.

    This implementation uses Dask (distributed) as the task engine,
    as mandated by the system architecture for high-throughput
    parallelism on both local machines and HPC clusters (via dask-jobqueue).
    """

    def __init__(self, scheduler_address: str | None = None, workers: int = 4) -> None:
        """
        Initialize the TaskQueue.

        Args:
            scheduler_address: Address of an existing Dask scheduler.
                               If None, a LocalCluster is started.
            workers: Number of workers to spawn if starting a LocalCluster.
        """
        self.cluster = None
        if scheduler_address:
            logger.info(f"Connecting to existing Dask scheduler at {scheduler_address}")
            self.client = Client(scheduler_address)
        else:
            logger.info(f"Starting LocalCluster with {workers} workers")
            self.cluster = LocalCluster(n_workers=workers)
            self.client = Client(self.cluster)

        logger.info(f"Dask Client initialized: {self.client}")

    def submit_dft_batch(
        self, func: Callable[..., Any], items: list[Any], **kwargs: Any
    ) -> list[Future[Any]]:
        """
        Submit a batch of DFT tasks to the cluster.

        Args:
            func: The function to execute (e.g., QERunner.run).
            items: List of items to process (e.g., list of Atoms).
            kwargs: Additional arguments to pass to the function.

        Returns:
            List of Dask Futures representing the submitted tasks.
        """
        logger.info(f"Submitting {len(items)} tasks to Dask.")
        return self.client.map(func, items, **kwargs)

    def wait_for_completion(
        self, futures: list[Future[Any]], timeout: float | None = None
    ) -> list[Any]:
        """
        Wait for a list of futures to complete and return their results.

        Handles exceptions gracefully by logging failures and returning None
        for failed tasks to prevent pipeline crashes.

        Args:
            futures: List of futures to wait for.
            timeout: Optional timeout in seconds.

        Returns:
            List of results corresponding to the input futures.
            Failed tasks result in None.
        """
        logger.info(f"Waiting for {len(futures)} tasks to complete...")
        wait(futures, timeout=timeout)

        results = []
        for f in futures:
            if f.status == "finished":
                results.append(f.result())
            else:
                logger.warning(f"Task {f} did not finish successfully (status: {f.status}).")
                # Depending on requirement, we might want to capture exception or return None
                # For now, let's try to get exception if failed
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.exception(f"Task failed with error: {e}")
                    results.append(None)

        return results

    def shutdown(self) -> None:
        """
        Shutdown the Dask client and cluster.
        """
        logger.info("Shutting down Dask client...")
        self.client.close()
        if self.cluster:
            self.cluster.close()
