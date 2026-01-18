from pathlib import Path

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.inference.inputs import ScriptGenerator


@pytest.fixture
def sample_atoms():
    return Atoms(
        "Al4", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], cell=[4, 4, 4], pbc=True
    )


@pytest.fixture
def config():
    return InferenceConfig(
        temperature=800,
        steps=1000,
        timestep=0.001,
        ensemble="nvt",
        uq_threshold=5.0,
        potential_path=Path("/tmp/model.yace"),
    )


def test_generate_script_nvt(sample_atoms, config):
    generator = ScriptGenerator(config)
    script = generator.generate(sample_atoms, Path("/tmp/work"), Path("structure.data"))

    assert "pair_style      pace" in script
    assert "compute         max_gamma all reduce max c_pace[1]" in script or "compute" in script
    assert "fix             1 all nvt" in script
    assert "temp 800.0" in script
    assert "dump_modify" in script
    assert "thresh c_max_gamma > 5.0" in script or "thresh" in script


def test_generate_script_npt(sample_atoms):
    conf = InferenceConfig(
        temperature=800,
        steps=1000,
        timestep=0.001,
        ensemble="npt",
        pressure=100.0,
        uq_threshold=5.0,
        potential_path=Path("/tmp/model.yace"),
    )
    generator = ScriptGenerator(conf)
    script = generator.generate(sample_atoms, Path("/tmp/work"), Path("structure.data"))

    assert "fix             1 all npt" in script
    assert "iso 100.0 100.0" in script
