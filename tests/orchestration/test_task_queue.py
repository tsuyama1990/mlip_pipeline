from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.orchestration.task_queue import TaskQueue


@pytest.fixture
def mock_dask_client():
    with patch("mlip_autopipec.orchestration.task_queue.Client") as mock_client:
        with patch("mlip_autopipec.orchestration.task_queue.LocalCluster") as mock_cluster:
            yield mock_client, mock_cluster


def test_task_queue_init_local(mock_dask_client):
    mock_client_cls, mock_cluster_cls = mock_dask_client

    tq = TaskQueue(workers=2)

    mock_cluster_cls.assert_called_once_with(n_workers=2)
    mock_client_cls.assert_called_once_with(mock_cluster_cls.return_value)
    assert tq.cluster is not None


def test_task_queue_init_remote(mock_dask_client):
    mock_client_cls, mock_cluster_cls = mock_dask_client

    tq = TaskQueue(scheduler_address="tcp://remote:8786")

    mock_cluster_cls.assert_not_called()
    mock_client_cls.assert_called_once_with("tcp://remote:8786")
    assert tq.cluster is None


def test_submit_dft_batch(mock_dask_client):
    mock_client_cls, _ = mock_dask_client
    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.map.return_value = ["future1", "future2"]

    tq = TaskQueue()
    func = MagicMock()
    items = [1, 2]

    futures = tq.submit_dft_batch(func, items, some_arg="value")

    mock_client_instance.map.assert_called_once_with(func, items, some_arg="value")
    assert futures == ["future1", "future2"]


@patch("mlip_autopipec.orchestration.task_queue.wait")
def test_wait_for_completion(mock_wait, mock_dask_client):
    mock_client_cls, _ = mock_dask_client
    tq = TaskQueue()

    future_success = MagicMock()
    future_success.status = "finished"
    future_success.result.return_value = "success"

    future_fail = MagicMock()
    future_fail.status = "error"
    # Simulation of exception access. In real dask future.result() raises
    future_fail.result.side_effect = Exception("Task failed")

    futures = [future_success, future_fail]

    results = tq.wait_for_completion(futures)

    mock_wait.assert_called_once_with(futures, timeout=None)
    assert results == ["success", None]  # Failed task returns None as per implementation


def test_shutdown(mock_dask_client):
    mock_client_cls, mock_cluster_cls = mock_dask_client
    tq = TaskQueue()
    tq.shutdown()

    mock_client_cls.return_value.close.assert_called_once()
    mock_cluster_cls.return_value.close.assert_called_once()
