"""
This module provides helper functions for setting up and managing Dask clients,
making it easier to switch between local and distributed computing environments.
"""

import logging
import os

from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def get_dask_client() -> Client:
    """
    Initializes and returns a Dask client.

    This function provides a standardized way to connect to a Dask cluster.
    It checks for the `DASK_SCHEDULER_ADDRESS` environment variable. If the
    variable is set, it connects to the specified Dask scheduler. This is
    typical for an HPC or production environment. If the variable is not set,
    it creates a `dask.distributed.LocalCluster` and connects to it. This is
    useful for local development and testing, as it automatically sets up a
    multi-worker cluster using the local machine's resources without requiring
    any external setup.

    Returns:
        A Dask `Client` object connected to either a remote scheduler or a
        local cluster.
    """
    scheduler_address = os.environ.get("DASK_SCHEDULER_ADDRESS")

    if scheduler_address:
        logger.info(f"Connecting to Dask scheduler at: {scheduler_address}")
        return Client(scheduler_address)
    logger.info("DASK_SCHEDULER_ADDRESS not set. Creating a local Dask cluster.")
    # `LocalCluster` automatically starts a scheduler and workers on the
    # local machine. When the `Client` is created with a `LocalCluster`
    # instance, it manages the lifecycle of the cluster. The cluster
    # will be shut down automatically when the client is closed or when
    # the program exits.
    cluster = LocalCluster()
    return Client(cluster)
