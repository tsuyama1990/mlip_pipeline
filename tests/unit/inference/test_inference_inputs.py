
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
