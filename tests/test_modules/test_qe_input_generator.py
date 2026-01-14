from unittest.mock import patch

from ase import Atoms

from mlip_autopipec.modules.qe_input_generator import QEInputGenerator
from mlip_autopipec.schemas.dft import DFTInput
from mlip_autopipec.schemas.system_config import DFTParams


def test_qe_input_generator_with_spin_and_smearing() -> None:
    """Test that spin and smearing are correctly written to the input file."""
    atoms = Atoms("Fe", pbc=True, cell=[2.8, 2.8, 2.8])
    dft_params = DFTParams(
        pseudopotentials={"Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"},
        cutoff_wfc=60,
        k_points=(8, 8, 8),
        smearing="methfessel-paxton",
        degauss=0.02,
        nspin=2,
    )
    dft_input = DFTInput(atoms=atoms, dft_params=dft_params)

    from pathlib import Path

    with patch("mlip_autopipec.modules.qe_input_generator.Espresso") as mock_espresso:
        input_generator = QEInputGenerator(dft_input)
        input_generator.write_input(Path("dummy_dir"))

        mock_espresso.assert_called_once()
        args, kwargs = mock_espresso.call_args
        input_data = kwargs["input_data"]
        assert input_data["nspin"] == 2
        assert input_data["smearing"] == "methfessel-paxton"
        assert input_data["degauss"] == 0.02


def test_qe_input_generator_with_mixing_beta() -> None:
    """Test that mixing_beta is correctly written to the input file."""
    atoms = Atoms("H")
    dft_params = DFTParams(
        pseudopotentials={"H": "H.usp"},
        cutoff_wfc=40,
        k_points=(1, 1, 1),
        smearing="gauss",
        degauss=0.01,
        nspin=1,
        mixing_beta=0.3,
    )
    dft_input = DFTInput(atoms=atoms, dft_params=dft_params)

    from pathlib import Path

    with patch("mlip_autopipec.modules.qe_input_generator.Espresso") as mock_espresso:
        input_generator = QEInputGenerator(dft_input)
        input_generator.write_input(Path("dummy_dir"))

        mock_espresso.assert_called_once()
        args, kwargs = mock_espresso.call_args
        input_data = kwargs["input_data"]
        assert input_data["ELECTRONS"]["mixing_beta"] == 0.3
