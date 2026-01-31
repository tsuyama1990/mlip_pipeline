from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.domain_models.config import (
    BulkStructureGenConfig,
    Config,
    MDConfig,
    PotentialConfig,
)
from mlip_autopipec.domain_models.validation import (
    ValidationConfig,
    ValidationMetric,
)
from mlip_autopipec.physics.validation.runner import ValidationRunner


@pytest.fixture
def mock_config() -> Config:
    return Config(
        project_name="test",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        md=MDConfig(temperature=300, n_steps=100, ensemble="NVT"),
        structure_gen=BulkStructureGenConfig(
            element="Si", crystal_structure="diamond", lattice_constant=5.43
        ),
        validation=ValidationConfig(),
    )


@patch("mlip_autopipec.physics.validation.runner.StructureGenFactory")
@patch("mlip_autopipec.physics.validation.runner.PhononValidator")
@patch("mlip_autopipec.physics.validation.runner.ElasticityValidator")
@patch("mlip_autopipec.physics.validation.runner.EOSValidator")
def test_runner_validate_success(
    MockEOS: MagicMock,
    MockElasticity: MagicMock,
    MockPhonon: MagicMock,
    MockGenFactory: MagicMock,
    mock_config: Config,
    tmp_path: Path,
) -> None:
    # Setup mocks
    mock_eos_instance = MockEOS.return_value
    mock_eos_instance.validate.return_value = (
        [ValidationMetric(name="EOS", value=1.0, passed=True)],
        {"eos_plot": Path("eos.png")},
    )

    mock_elas_instance = MockElasticity.return_value
    mock_elas_instance.validate.return_value = (
        [ValidationMetric(name="Elastic", value=1.0, passed=True)],
        {},
    )

    mock_phonon_instance = MockPhonon.return_value
    mock_phonon_instance.validate.return_value = (
        [ValidationMetric(name="Phonon", value=1.0, passed=True)],
        {"phonon_plot": Path("phon.png")},
    )

    mock_gen = MockGenFactory.get_generator.return_value
    mock_gen.generate.return_value = MagicMock()  # Structure

    # Run
    runner = ValidationRunner(mock_config, work_dir=tmp_path)
    result = runner.validate(Path("pot.yace"))

    assert result.overall_status == "PASS"
    assert len(result.metrics) == 3
    assert len(result.plots) == 2
    assert "eos_plot" in result.plots
    assert "phonon_plot" in result.plots


@patch("mlip_autopipec.physics.validation.runner.StructureGenFactory")
@patch("mlip_autopipec.physics.validation.runner.PhononValidator")
@patch("mlip_autopipec.physics.validation.runner.ElasticityValidator")
@patch("mlip_autopipec.physics.validation.runner.EOSValidator")
def test_runner_validate_fail(
    MockEOS: MagicMock,
    MockElasticity: MagicMock,
    MockPhonon: MagicMock,
    MockGenFactory: MagicMock,
    mock_config: Config,
    tmp_path: Path,
) -> None:
    # Setup mocks
    mock_eos_instance = MockEOS.return_value
    mock_eos_instance.validate.return_value = (
        [ValidationMetric(name="EOS", value=1.0, passed=True)],
        {},
    )

    mock_elas_instance = MockElasticity.return_value
    mock_elas_instance.validate.return_value = (
        [ValidationMetric(name="Elastic", value=1.0, passed=True)],
        {},
    )

    mock_phonon_instance = MockPhonon.return_value
    # Phonon failed
    mock_phonon_instance.validate.return_value = (
        [ValidationMetric(name="Phonon", value=-1.0, passed=False)],
        {},
    )

    mock_gen = MockGenFactory.get_generator.return_value
    mock_gen.generate.return_value = MagicMock()

    runner = ValidationRunner(mock_config, work_dir=tmp_path)
    result = runner.validate(Path("pot.yace"))

    assert result.overall_status == "FAIL"
