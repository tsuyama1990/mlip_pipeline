from unittest.mock import patch

import pytest

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.training.pacemaker import PacemakerWrapper


@pytest.fixture
def mock_config(tmp_path):
    return TrainingConfig(
        template_path=tmp_path / "template.yaml",
        cutoff=5.0,
        b_basis_size=100,
        kappa=0.5,
        kappa_f=100.0,
        batch_size=32
    )


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
    (tmp_path / "template.yaml").touch()

    result = wrapper.train()
    assert result.success
    assert result.potential_path.exists()
