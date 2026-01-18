from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.orchestration.task_queue import TaskQueue


@pytest.fixture
def mock_dask_client():
    with patch('mlip_autopipec.orchestration.task_queue.Client') as mock_client:
        with patch('mlip_autopipec.orchestration.task_queue.LocalCluster') as mock_cluster:
            yield mock_client, mock_cluster

def test_task_queue_init_local(mock_dask_client) -> None:
    mock_client, mock_cluster = mock_dask_client
    tq = TaskQueue(workers=2)

    mock_cluster.assert_called_once_with(n_workers=2)
    mock_client.assert_called_with(mock_cluster.return_value)
    assert tq.cluster is not None

def test_task_queue_init_remote(mock_dask_client) -> None:
    mock_client, mock_cluster = mock_dask_client
    tq = TaskQueue(scheduler_address="tcp://localhost:8786")

    mock_client.assert_called_with("tcp://localhost:8786")
    mock_cluster.assert_not_called()
    assert tq.cluster is None

def test_submit_dft_batch(mock_dask_client) -> None:
    mock_client, _ = mock_dask_client
    tq = TaskQueue()
    mock_client_instance = mock_client.return_value

    func = MagicMock()
    items = [1, 2, 3]
    tq.submit_dft_batch(func, items, extra_arg="test")

    mock_client_instance.map.assert_called_once_with(func, items, extra_arg="test")

def test_wait_for_completion(mock_dask_client) -> None:
    with patch('mlip_autopipec.orchestration.task_queue.wait') as mock_wait:
        mock_client, _ = mock_dask_client
        tq = TaskQueue()

        f1 = MagicMock()
        f1.status = 'finished'
        f1.result.return_value = 'result1'

        f2 = MagicMock()
        f2.status = 'error'
        f2.result.side_effect = Exception("error")

        futures = [f1, f2]

        results = tq.wait_for_completion(futures)

        mock_wait.assert_called_once_with(futures, timeout=None)
        assert len(results) == 2
        assert results[0] == 'result1'
        assert results[1] is None # Error case returns None

def test_shutdown(mock_dask_client) -> None:
    mock_client, mock_cluster = mock_dask_client
    tq = TaskQueue()
    tq.shutdown()

    mock_client.return_value.close.assert_called_once()
    mock_cluster.return_value.close.assert_called_once()
