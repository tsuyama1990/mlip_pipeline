from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig, Structure


def test_valid_structure(valid_structure: Structure) -> None:
    assert valid_structure.energy == -10.5
    assert len(valid_structure.atomic_numbers) == 2

def test_structure_shape_mismatch() -> None:
    with pytest.raises(ValidationError) as exc:
        Structure(
            atomic_numbers=[1, 1],
            positions=[[0.0, 0.0, 0.0]], # Only 1 position
            cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            pbc=[True, True, True]
        )
    assert "positions length 1 does not match atomic_numbers length 2" in str(exc.value)

def test_structure_cell_shape() -> None:
    with pytest.raises(ValidationError) as exc:
        Structure(
            atomic_numbers=[1],
            positions=[[0.0, 0.0, 0.0]],
            cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], # 2x3 cell
            pbc=[True, True, True]
        )
    assert "cell must be 3x3" in str(exc.value)

def test_config_validation(mock_config: GlobalConfig) -> None:
    assert mock_config.max_cycles == 2
    assert mock_config.generator["type"] == "mock"

def test_config_missing_type(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as exc:
        GlobalConfig(
            workdir=tmp_path,
            max_cycles=1,
            generator={}, # Missing type
            oracle={"type": "mock"},
            trainer={"type": "mock"},
            dynamics={"type": "mock"}
        )
    assert "generator config must specify 'type'" in str(exc.value)
