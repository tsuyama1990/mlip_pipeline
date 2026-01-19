from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from mlip_autopipec.orchestration.task_queue import TaskQueue


@pytest.fixture
def mock_dask(mocker: MockerFixture) -> tuple[MagicMock, MagicMock]:
    mock_client = mocker.patch("mlip_autopipec.orchestration.task_queue.Client")
    mock_cluster = mocker.patch("mlip_autopipec.orchestration.task_queue.LocalCluster")
    return mock_client, mock_cluster


def test_task_queue_init_local(mock_dask: tuple[MagicMock, MagicMock]) -> None:
    mock_client, mock_cluster = mock_dask
    queue = TaskQueue(workers=2)

    mock_cluster.assert_called_once_with(n_workers=2)
    mock_client.assert_called_once_with(mock_cluster.return_value)
    assert queue.client == mock_client.return_value


def test_task_queue_init_remote(mock_dask: tuple[MagicMock, MagicMock]) -> None:
    mock_client, mock_cluster = mock_dask
    address = "tcp://127.0.0.1:8786"
    queue = TaskQueue(scheduler_address=address)

    mock_cluster.assert_not_called()
    mock_client.assert_called_once_with(address)
    assert queue.client == mock_client.return_value


def _dummy_func(x: int) -> int:
    return x

def test_submit_dft_batch(mock_dask: tuple[MagicMock, MagicMock]) -> None:
    mock_client, _ = mock_dask
    queue = TaskQueue()

    inputs = [1, 2, 3]

    queue.submit_dft_batch(_dummy_func, inputs)

    queue.client.map.assert_called_once()
    # Note: client.map args might include kwargs, so explicit check:
    args, _ = queue.client.map.call_args
    assert args[0] == _dummy_func
    assert args[1] == inputs


def test_wait_for_completion(mocker: MockerFixture, mock_dask: tuple[MagicMock, MagicMock]) -> None:
    mock_wait = mocker.patch("mlip_autopipec.orchestration.task_queue.wait")
    queue = TaskQueue()

    # Mock futures
    f1 = MagicMock()
    f1.status = 'finished'
    f1.result.return_value = 'result1'

    f2 = MagicMock()
    f2.status = 'finished'
    f2.result.return_value = 'result2'

    futures = [f1, f2]

    results = queue.wait_for_completion(futures)

    mock_wait.assert_called_once_with(futures, timeout=None)
    assert results == ['result1', 'result2']


def test_shutdown(mock_dask: tuple[MagicMock, MagicMock]) -> None:
    queue = TaskQueue()
    queue.shutdown()
    queue.client.close.assert_called_once()
    # Cluster is None in mock_dask init unless we enforce it,
    # but let's verify logic does not crash
