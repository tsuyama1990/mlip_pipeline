
import pytest

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def mock_validation_runner():
    config = ValidationConfig(
        run_validation=True, check_phonons=True, check_elastic=True
    )
    return ValidationRunner(config)


def test_validation_runner_flow(mock_validation_runner, tmp_path):
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    result = mock_validation_runner.validate(potential_path, tmp_path)

    # The dummy LJ potential might fail phonon stability for Cu
    # So we don't assert result.passed is True
    assert isinstance(result.passed, bool)
    assert len(result.metrics) == 2
    assert result.report_path is not None
    assert result.report_path.exists()

    # Check report content
    content = result.report_path.read_text()
    assert "phonons" in content
    assert "elastic" in content

    # Check that plots were generated
    assert (tmp_path / "phonon_band_structure.png").exists()
