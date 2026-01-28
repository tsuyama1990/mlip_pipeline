from unittest.mock import patch

import pytest

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.training.pacemaker import PacemakerWrapper


@pytest.fixture
def mock_config(tmp_path):
    return TrainingConfig(template_path=tmp_path / "template.yaml", potential_name="test_pot")


def test_pacemaker_init(mock_config, tmp_path):
    wrapper = PacemakerWrapper(mock_config, tmp_path)
    assert wrapper.work_dir == tmp_path


@patch("subprocess.run")
@patch("shutil.which")
def test_train_success(mock_which, mock_run, mock_config, tmp_path):
    mock_which.return_value = "/bin/pacemaker"
    wrapper = PacemakerWrapper(mock_config, tmp_path)

    # Mock output file creation
    (tmp_path / "output_potential.yace").touch()

    result = wrapper.train()
    assert result.success
    assert result.potential_path.exists()
