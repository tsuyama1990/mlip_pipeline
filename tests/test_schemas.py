import pytest
from pydantic import ValidationError

from mlip_autopipec.schemas.user_config import (
    GenerationConfig,
    TargetSystem,
    UserConfig,
)


def test_target_system_valid():
    ts = TargetSystem(
        elements=["Fe", "Ni"],
        composition={"Fe": 0.5, "Ni": 0.5},
        crystal_structure="fcc",
    )
    assert ts.elements == ["Fe", "Ni"]
    assert ts.composition == {"Fe": 0.5, "Ni": 0.5}


def test_target_system_composition_sum_invalid():
    with pytest.raises(ValidationError):
        UserConfig(
            project_name="test",
            target_system=TargetSystem(
                elements=["Fe", "Ni"],
                composition={"Fe": 0.5, "Ni": 0.4},
                crystal_structure="fcc",
            ),
            generation_config=GenerationConfig(generation_type="alloy_sqs"),
        )


def test_target_system_elements_mismatch_invalid():
    with pytest.raises(ValidationError):
        UserConfig(
            project_name="test",
            target_system=TargetSystem(
                elements=["Fe", "Cr"],
                composition={"Fe": 0.5, "Ni": 0.5},
                crystal_structure="fcc",
            ),
            generation_config=GenerationConfig(generation_type="alloy_sqs"),
        )


def test_user_config_valid():
    uc = UserConfig(
        project_name="TestProject",
        target_system=TargetSystem(
            elements=["Fe", "Ni"],
            composition={"Fe": 0.5, "Ni": 0.5},
            crystal_structure="fcc",
        ),
        generation_config=GenerationConfig(generation_type="alloy_sqs"),
    )
    assert uc.project_name == "TestProject"


def test_target_system_empty_elements_invalid():
    with pytest.raises(ValidationError):
        UserConfig(
            project_name="test",
            target_system=TargetSystem(
                elements=[],
                composition={"Fe": 0.5, "Ni": 0.5},
                crystal_structure="fcc",
            ),
            generation_config=GenerationConfig(generation_type="alloy_sqs"),
        )


def test_target_system_empty_composition_invalid():
    with pytest.raises(ValidationError):
        UserConfig(
            project_name="test",
            target_system=TargetSystem(
                elements=["Fe", "Ni"],
                composition={},
                crystal_structure="fcc",
            ),
            generation_config=GenerationConfig(generation_type="alloy_sqs"),
        )


def test_system_config_valid():
    from mlip_autopipec.schemas.system_config import (
        DFTParams,
        GeneratorParams,
        SystemConfig,
    )

    sc = SystemConfig(
        dft_params=DFTParams(
            pseudopotentials={"Fe": "Fe.UPF"},
            cutoff_wfc=40,
            k_points=(4, 4, 4),
            smearing_type="mv",
            degauss=0.01,
            nspin=2,
        ),
        generator_params=GeneratorParams(
            generation_type="alloy_sqs",
            sqs_supercell_size=[2, 2, 2],
            strain_magnitudes=[-0.01, 0.01],
            rattle_std_dev=0.05,
        ),
    )
    assert sc.dft_params.cutoff_wfc == 40


def test_dft_schemas_valid():
    from ase.build import bulk

    from mlip_autopipec.schemas.dft import DFTInput, DFTOutput
    from mlip_autopipec.schemas.system_config import DFTParams

    atoms = bulk("Si", "diamond", a=5.43)
    dft_params = DFTParams(
        pseudopotentials={"Si": "Si.UPF"},
        cutoff_wfc=30,
        k_points=(4, 4, 4),
        smearing_type="fd",
        degauss=0.01,
        nspin=1,
    )
    dft_input = DFTInput(atoms=atoms, dft_params=dft_params)
    assert dft_input.atoms == atoms

    dft_output = DFTOutput(
        total_energy=-100.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert dft_output.total_energy == -100.0
