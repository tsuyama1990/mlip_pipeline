from pathlib import Path

import pytest

from mlip_autopipec.domain_models import GlobalConfig, Structure


@pytest.fixture
def valid_structure() -> Structure:
    return Structure(
        atomic_numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=[True, True, True],
        energy=-10.5,
        forces=[[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

@pytest.fixture
def mock_config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(
        workdir=tmp_path / "work",
        max_cycles=2,
        generator={"type": "mock"},
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        dynamics={"type": "mock"}
    )
