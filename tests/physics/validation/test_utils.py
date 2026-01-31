from unittest.mock import patch
from mlip_autopipec.physics.validation.utils import get_lammps_calculator
from mlip_autopipec.domain_models.config import PotentialConfig

def test_get_lammps_calculator(tmp_path):
    pot_config = PotentialConfig(
        elements=["Al"],
        cutoff=5.0,
        pair_style="pace"
    )
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    with patch("mlip_autopipec.physics.validation.utils.LAMMPS") as MockLAMMPS:
        get_lammps_calculator(
            potential_path=pot_path,
            potential_config=pot_config,
            lammps_command="lmp_serial",
            working_dir=tmp_path
        )

        MockLAMMPS.assert_called_once()
        _, kwargs = MockLAMMPS.call_args
        assert kwargs["label"] == "val"
        assert kwargs["command"] == "lmp_serial"
        assert "pair_style" in kwargs
        assert kwargs["pair_style"] == "pace"
        assert "pair_coeff" in kwargs
        assert "pace" in kwargs["pair_coeff"][0]

def test_get_lammps_calculator_hybrid(tmp_path):
    pot_config = PotentialConfig(
        elements=["Al"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        zbl_inner_cutoff=0.5,
        zbl_outer_cutoff=1.0
    )
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    with patch("mlip_autopipec.physics.validation.utils.LAMMPS") as MockLAMMPS:
        get_lammps_calculator(
            potential_path=pot_path,
            potential_config=pot_config,
            working_dir=tmp_path
        )
        _, kwargs = MockLAMMPS.call_args
        assert "hybrid/overlay" in kwargs["pair_style"]
        # Check ZBL is in pair_coeff
        # We have 1 element, so 1 pair (1 1)
        # pair_coeff should have length 2 (pace + zbl)
        assert len(kwargs["pair_coeff"]) == 2
        assert "zbl" in kwargs["pair_coeff"][1]


def test_get_lammps_calculator_empty_elements(tmp_path):
    """Test behavior with empty elements list."""
    pot_config = PotentialConfig(
        elements=[],
        cutoff=5.0,
        pair_style="hybrid/overlay"
    )
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    # Should probably raise error or produce weird config
    # Since specorder is empty, LAMMPS calc might complain or just write empty data

    with patch("mlip_autopipec.physics.validation.utils.LAMMPS") as MockLAMMPS:
        get_lammps_calculator(
            potential_path=pot_path,
            potential_config=pot_config,
            working_dir=tmp_path
        )
        _, kwargs = MockLAMMPS.call_args
        # Check that we still got calls
        assert kwargs["specorder"] == []
