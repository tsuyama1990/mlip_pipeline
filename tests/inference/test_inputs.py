import pytest
from pathlib import Path
from mlip_autopipec.inference.inputs import ScriptGenerator
from mlip_autopipec.config.schemas.inference import InferenceConfig

@pytest.fixture
def basic_config(tmp_path: Path) -> InferenceConfig:
    p = tmp_path / "model.yace"
    p.touch()
    return InferenceConfig(
        temperature=1000.0,
        pressure=0.0,
        timestep=0.002,
        steps=5000,
        ensemble="nvt",
        uq_threshold=4.5,
        sampling_interval=50,
        potential_path=p,
        elements=["Al"] # Default in model, but explicit here
    )

def test_generate_script_nvt(basic_config: InferenceConfig, tmp_path: Path) -> None:
    gen = ScriptGenerator(basic_config)
    script = gen.generate(
        atoms_file=Path("data.lammps"),
        potential_path=basic_config.potential_path,
        dump_file=Path("dump.gamma")
    )

    assert "units metal" in script
    assert "atom_style atomic" in script
    # Implementation resolves path, so check for resolved path
    assert f"read_data {Path('data.lammps').resolve()}" in script
    assert f"pair_style pace" in script
    assert f"pair_coeff * * {basic_config.potential_path.resolve()} Al" in script
    assert "fix 1 all nvt temp 1000.0 1000.0 0.2" in script
    assert "timestep 0.002" in script
    assert "compute max_gamma all reduce max c_pace[1]" in script
    assert "dump_modify my_dump thresh c_max_gamma > 4.5" in script
    assert "run 5000" in script

def test_generate_script_multi_element(basic_config: InferenceConfig, tmp_path: Path) -> None:
    basic_config.elements = ["Al", "Cu"]
    gen = ScriptGenerator(basic_config)
    script = gen.generate(
        atoms_file=Path("data.lammps"),
        potential_path=basic_config.potential_path,
        dump_file=Path("dump.gamma")
    )
    assert f"pair_coeff * * {basic_config.potential_path.resolve()} Al Cu" in script

def test_generate_script_npt(basic_config: InferenceConfig, tmp_path: Path) -> None:
    basic_config.ensemble = "npt"
    basic_config.pressure = 100.0

    gen = ScriptGenerator(basic_config)
    script = gen.generate(
        atoms_file=Path("data.lammps"),
        potential_path=basic_config.potential_path,
        dump_file=Path("dump.gamma")
    )

    assert "fix 1 all npt temp 1000.0 1000.0 0.2 iso 100.0 100.0 2.0" in script
