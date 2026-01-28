import pytest

from mlip_autopipec.config.schemas.inference import InferenceConfig, BaselinePotential
from mlip_autopipec.inference.inputs import ScriptGenerator


@pytest.fixture
def basic_config():
    return InferenceConfig(
        steps=1000,
        timestep=1.0,
        temperature=300.0,
        pressure=1.0,
        ensemble="nvt",
        baseline_potential=BaselinePotential.ZBL,
        uncertainty_threshold=5.0,
    )


def test_script_generator_nvt_zbl(basic_config, tmp_path):
    generator = ScriptGenerator(basic_config)
    atoms_file = tmp_path / "data.lammps"
    potential_path = tmp_path / "potential.yace"
    dump_file = tmp_path / "dump.out"
    elements = ["Li", "O"]

    script = generator.generate(atoms_file, potential_path, dump_file, elements)

    assert "pair_style hybrid/overlay pace" in script
    assert f"pair_coeff * * pace {potential_path.resolve()} Li O" in script
    # Li (3), O (8). Indices 1, 2.
    assert "pair_coeff 1 1 zbl 3 3" in script
    assert "pair_coeff 1 2 zbl 3 8" in script
    assert "pair_coeff 2 2 zbl 8 8" in script
    assert "fix 1 all nvt" in script
    assert "fix watchdog all halt" in script
    assert "v_max_gamma > 5.0" in script
    assert "restart 1000" in script


def test_script_generator_npt_no_zbl(basic_config, tmp_path):
    basic_config.ensemble = "npt"
    basic_config.baseline_potential = BaselinePotential.NONE

    generator = ScriptGenerator(basic_config)
    atoms_file = tmp_path / "data.lammps"
    potential_path = tmp_path / "potential.yace"
    dump_file = tmp_path / "dump.out"
    elements = ["Si"]

    script = generator.generate(atoms_file, potential_path, dump_file, elements)

    assert "pair_style pace" in script
    assert "hybrid/overlay" not in script
    assert "fix 1 all npt" in script
    assert "iso 1.0 1.0" in script


def test_script_generator_dump_modify(basic_config, tmp_path):
    generator = ScriptGenerator(basic_config)
    atoms_file = tmp_path / "data.lammps"
    potential_path = tmp_path / "potential.yace"
    dump_file = tmp_path / "dump.out"
    elements = ["Fe"]

    script = generator.generate(atoms_file, potential_path, dump_file, elements)

    # Check that dump command and dump_modify thresh are present
    assert (
        f"dump my_dump all custom {basic_config.sampling_interval} {dump_file.resolve()}" in script
    )
    assert "dump_modify my_dump thresh" in script
    # Thresh uses > 5.0
    assert f"> {basic_config.uncertainty_threshold}" in script


def test_script_generator_multiple_elements_zbl(basic_config, tmp_path):
    # Test complex system
    generator = ScriptGenerator(basic_config)
    atoms_file = tmp_path / "data.lammps"
    potential_path = tmp_path / "potential.yace"
    dump_file = tmp_path / "dump.out"
    elements = ["H", "C", "O"]  # 1, 6, 8

    script = generator.generate(atoms_file, potential_path, dump_file, elements)

    assert "pair_style hybrid/overlay pace" in script
    assert "pair_coeff 1 1 zbl 1 1" in script
    assert "pair_coeff 1 2 zbl 1 6" in script
    assert "pair_coeff 1 3 zbl 1 8" in script
    assert "pair_coeff 2 2 zbl 6 6" in script
    assert "pair_coeff 2 3 zbl 6 8" in script
    assert "pair_coeff 3 3 zbl 8 8" in script


def test_invalid_ensemble(basic_config, tmp_path):
    # Force invalid ensemble manually bypassing pydantic (if possible via dynamic assignment)
    basic_config.ensemble = "invalid"  # type: ignore

    generator = ScriptGenerator(basic_config)
    atoms_file = tmp_path / "data.lammps"
    potential_path = tmp_path / "potential.yace"
    dump_file = tmp_path / "dump.out"
    elements = ["Fe"]

    with pytest.raises(ValueError, match="Unsupported ensemble"):
        generator.generate(atoms_file, potential_path, dump_file, elements)
