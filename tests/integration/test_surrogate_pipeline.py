from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import RejectionInfo, SelectionResult, SurrogateConfig
from mlip_autopipec.surrogate.descriptors import DescriptorResult
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline


@pytest.fixture
def mock_mace_client():
    with patch("mlip_autopipec.surrogate.pipeline.MaceClient") as mock:
        yield mock


@pytest.fixture
def mock_descriptor_calc():
    with patch("mlip_autopipec.surrogate.pipeline.DescriptorCalculator") as mock:
        yield mock


@pytest.fixture
def mock_sampler():
    with patch("mlip_autopipec.surrogate.pipeline.FPSSampler") as mock:
        yield mock


def test_surrogate_pipeline_init():
    config = SurrogateConfig()
    pipeline = SurrogatePipeline(config)
    assert pipeline.config == config
    assert pipeline.mace_client is not None
    assert pipeline.descriptor_calc is not None
    assert pipeline.sampler is not None


def test_surrogate_pipeline_run(mock_mace_client, mock_descriptor_calc, mock_sampler):
    config = SurrogateConfig(fps_n_samples=2)
    pipeline = SurrogatePipeline(config)

    # Setup Data
    candidates = [Atoms("H"), Atoms("H"), Atoms("H")]
    kept_candidates = [candidates[0], candidates[2]]  # 2nd one filtered

    # Mock MaceClient
    client_instance = mock_mace_client.return_value
    client_instance.filter_unphysical.return_value = (
        kept_candidates,
        [RejectionInfo(index=1, max_force=100.0, reason="test")],
    )

    # Mock DescriptorCalculator
    calc_instance = mock_descriptor_calc.return_value
    # 2 descriptors for 2 kept atoms
    calc_instance.compute_soap.return_value = DescriptorResult(features=np.zeros((2, 10)))

    # Mock FPSSampler
    sampler_instance = mock_sampler.return_value
    # Select both
    sampler_instance.select_with_scores.return_value = ([0, 1], [0.0, 1.0])

    # Run
    selected, result = pipeline.run(candidates)

    # Verify
    assert len(selected) == 2
    assert selected[0] == candidates[0]
    assert selected[1] == candidates[2]

    # Check contents of SelectionResult
    assert isinstance(result, SelectionResult)
    assert result.selected_indices == [0, 2]
    assert result.scores == [0.0, 1.0]

    # Verify calls
    client_instance.filter_unphysical.assert_called_once()
    calc_instance.compute_soap.assert_called_once()
    sampler_instance.select_with_scores.assert_called_once()


def test_surrogate_pipeline_empty_input():
    config = SurrogateConfig()
    pipeline = SurrogatePipeline(config)
    selected, result = pipeline.run([])
    assert selected == []
    assert result.selected_indices == []


def test_surrogate_pipeline_all_filtered(mock_mace_client):
    config = SurrogateConfig()
    pipeline = SurrogatePipeline(config)

    client_instance = mock_mace_client.return_value
    client_instance.filter_unphysical.return_value = (
        [],
        [RejectionInfo(index=0, max_force=100.0, reason="test")],
    )

    candidates = [Atoms("H")]
    selected, result = pipeline.run(candidates)

    assert selected == []
    assert result.selected_indices == []


def test_surrogate_pipeline_failure(mock_mace_client):
    config = SurrogateConfig()
    pipeline = SurrogatePipeline(config)

    client_instance = mock_mace_client.return_value
    client_instance.filter_unphysical.side_effect = RuntimeError("Prediction failed")

    candidates = [Atoms("H")]
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.run(candidates)
    assert "Surrogate pipeline execution failed" in str(excinfo.value)
