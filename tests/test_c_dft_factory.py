from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.modules.c_dft_factory import _recover, run_qe_calculation
from mlip_autopipec.schemas.dft import DFTInput
from mlip_autopipec.schemas.system_config import DFTParams


def test_run_qe_calculation_retry():
    """Test that run_qe_calculation retries on failure."""
    dft_input = DFTInput(
        atoms=Atoms("H"),
        dft_params=DFTParams(
            pseudopotentials={"H": "H.pbe-rrkjus.UPF"},
            cutoff_wfc=60,
            k_points=(8, 8, 8),
            smearing="gauss",
            degauss=0.01,
            nspin=1,
        ),
    )

    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = ""

    with patch(
        "mlip_autopipec.modules.qe_process_runner.QEProcessRunner.run",
        return_value=mock_process,
    ) as mock_run:
        with pytest.raises(RuntimeError, match="Quantum Espresso failed after 3 retries."):
            run_qe_calculation(dft_input, max_retries=3)
        assert mock_run.call_count == 3


def test_recover():
    """Test the _recover function."""
    dft_input = DFTInput(
        atoms=Atoms("H"),
        dft_params=DFTParams(
            pseudopotentials={"H": "H.pbe-rrkjus.UPF"},
            cutoff_wfc=60,
            k_points=(8, 8, 8),
            smearing="gauss",
            degauss=0.01,
            nspin=1,
        ),
    )

    recovered_dft_input = _recover(dft_input)
    assert recovered_dft_input.dft_params.mixing_beta == 0.3
