
import pytest

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.inference.inputs import ScriptGenerator


@pytest.fixture
def basic_config():
    return InferenceConfig(
        steps=1000,
        timestep=1.0,
        temperature=300.0,
        pressure=1.0,
        ensemble="nvt",
        use_zbl_baseline=True,
        uncertainty_threshold=5.0
    )

def test_script_generator_nvt_zbl(basic_config, tmp_path):
    generator = ScriptGenerator(basic_config)
    atoms_file = tmp_path / "data.lammps"
    potential_path = tmp_path / "potential.yace"
    dump_file = tmp_path / "dump.out"
    elements = ["Li", "O"]

    script = generator.generate(atoms_file, potential_path, dump_file, elements)

    assert "pair_style hybrid/overlay pace zbl" in script
    assert f"pair_coeff * * pace {potential_path.resolve()} Li O" in script
    # We generate explicit pair coefficients for ZBL
    # Li (3), O (8). Indices 1, 2.
    assert "pair_coeff 1 1 zbl 3 3" in script
    assert "pair_coeff 1 2 zbl 3 8" in script
    assert "pair_coeff 2 2 zbl 8 8" in script
    assert "fix 1 all nvt" in script
    assert "fix halter all halt" in script
    assert "v_gamma_val > 5.0" in script
    assert "restart 1000" in script

def test_script_generator_npt_no_zbl(basic_config, tmp_path):
    basic_config.ensemble = "npt"
    basic_config.use_zbl_baseline = False

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
    assert f"dump my_dump all custom {basic_config.sampling_interval} {dump_file.resolve()}" in script
    assert "dump_modify my_dump thresh" in script
    assert str(basic_config.uncertainty_threshold) in script

def test_script_generator_multiple_elements_zbl(basic_config, tmp_path):
    # Test complex system
    generator = ScriptGenerator(basic_config)
    atoms_file = tmp_path / "data.lammps"
    potential_path = tmp_path / "potential.yace"
    dump_file = tmp_path / "dump.out"
    elements = ["H", "C", "O"] # 1, 6, 8

    script = generator.generate(atoms_file, potential_path, dump_file, elements)

    assert "pair_style hybrid/overlay pace zbl" in script
    # H-H (1-1)
    assert "pair_coeff 1 1 zbl 1 1" in script
    # H-C (1-2)
    assert "pair_coeff 1 2 zbl 1 6" in script
    # H-O (1-3)
    assert "pair_coeff 1 3 zbl 1 8" in script
    # C-C (2-2)
    assert "pair_coeff 2 2 zbl 6 6" in script
    # C-O (2-3)
    assert "pair_coeff 2 3 zbl 6 8" in script
    # O-O (3-3)
    assert "pair_coeff 3 3 zbl 8 8" in script

def test_invalid_ensemble(basic_config, tmp_path):
    # Force invalid ensemble manually bypassing pydantic (if possible via dynamic assignment)
    # or just subclassing config if pydantic validates on init
    # Since pydantic validates on init, we can't easily set invalid value if we type check.
    # But generator checks explicitly too?
    # Actually generator relies on config.ensemble which is Literal.
    # If we modify it forcefully:
    basic_config.ensemble = "invalid" # type: ignore

    generator = ScriptGenerator(basic_config)
    atoms_file = tmp_path / "data.lammps"
    potential_path = tmp_path / "potential.yace"
    dump_file = tmp_path / "dump.out"
    elements = ["Fe"]

    with pytest.raises(ValueError, match="Unsupported ensemble"):
        generator.generate(atoms_file, potential_path, dump_file, elements)
