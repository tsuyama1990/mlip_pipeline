from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.calculators.calculator import CalculatorError

from mlip_autopipec.config.config_model import OracleConfig

# These imports will fail initially, which is expected
try:
    from mlip_autopipec.infrastructure.espresso.adapter import EspressoOracle
except ImportError:
    EspressoOracle = None  # type: ignore

try:
    from mlip_autopipec.infrastructure.espresso.recovery import RecoveryStrategy
except ImportError:
    RecoveryStrategy = None  # type: ignore

from mlip_autopipec.domain_models import Dataset


def test_oracle_config_validation(tmp_path: Path) -> None:
    """Test validation of OracleConfig for Espresso."""
    # Should fail if missing required fields
    with pytest.raises(ValueError, match="requires 'command'"):
        OracleConfig(type="espresso")

    with pytest.raises(ValueError, match="requires 'pseudo_dir'"):
        OracleConfig(type="espresso", command="pw.x")

    with pytest.raises(ValueError, match="requires 'pseudopotentials'"):
        OracleConfig(type="espresso", command="pw.x", pseudo_dir=tmp_path)

    # Should pass with all fields
    config = OracleConfig(
        type="espresso",
        command="pw.x",
        pseudo_dir=tmp_path / "pseudo",
        pseudopotentials={"Si": "Si.upf"},
    )
    assert config.type == "espresso"
    assert config.command == "pw.x"


@pytest.mark.skipif(EspressoOracle is None, reason="EspressoOracle not implemented")
class TestEspressoOracle:
    @pytest.fixture
    def config(self, tmp_path: Path) -> OracleConfig:
        return OracleConfig(
            type="espresso",
            command="pw.x",
            pseudo_dir=tmp_path / "pseudo",
            pseudopotentials={"Si": "Si.upf"},
        )

    @pytest.fixture
    def oracle(self, config: OracleConfig, tmp_path: Path) -> "EspressoOracle":
        return EspressoOracle(config, work_dir=tmp_path)

    def test_validate_command_security(self, config: OracleConfig, tmp_path: Path) -> None:
        """Test that dangerous commands are rejected."""
        # This is a bit tricky since we validate in __init__ or before run
        # Let's assume validation happens in __init__ or check_config

        dangerous_commands = [
            "pw.x; rm -rf /",
            "pw.x | bash",
            "pw.x && echo 'hack'",
            "$(cat /etc/passwd)",
            "pw.x > output.txt",  # Redirection might be disallowed if we handle it
        ]

        for cmd in dangerous_commands:
            config.command = cmd
            with pytest.raises(ValueError, match="Security violation"):
                EspressoOracle(config, work_dir=tmp_path)

    def test_validate_command_valid(self, config: OracleConfig, tmp_path: Path) -> None:
        """Test that safe commands are accepted."""
        safe_commands = [
            "pw.x",
            "mpirun -np 4 pw.x",
            "/usr/bin/pw.x",
            "mpiexec.hydra -n 8 pw.x -in",
        ]
        for cmd in safe_commands:
            config.command = cmd
            # Should not raise
            EspressoOracle(config, work_dir=tmp_path)

    @patch("mlip_autopipec.infrastructure.espresso.adapter.iread")
    @patch("mlip_autopipec.infrastructure.espresso.adapter.write")
    @patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
    def test_label_streaming(
        self,
        mock_espresso: MagicMock,
        mock_write: MagicMock,
        mock_iread: MagicMock,
        oracle: "EspressoOracle",
        tmp_path: Path,
    ) -> None:
        """Test that labeling streams structures and writes them incrementally."""
        dataset_path = tmp_path / "input.xyz"
        dataset_path.touch()
        dataset = Dataset(file_path=dataset_path)

        # Mock input structures
        import numpy as np

        atoms1 = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10], pbc=True)
        atoms2 = Atoms("Si2", positions=[[0, 0, 0], [2, 2, 2]], cell=[10, 10, 10], pbc=True)
        mock_iread.return_value = [atoms1, atoms2]

        # Mock calculator behavior
        mock_calc = MagicMock()
        mock_calc.get_potential_energy.return_value = -10.0
        mock_calc.get_forces.return_value = np.zeros((2, 3))
        mock_calc.get_stress.return_value = np.zeros(6)
        mock_espresso.return_value = mock_calc

        # Run label
        result_dataset = oracle.label(dataset)

        # Verify result points to a file
        assert result_dataset.file_path.exists()
        assert result_dataset.file_path.name.endswith(".extxyz")  # or .xyz

        # Verify streaming: read called on input
        mock_iread.assert_called_once_with(dataset_path)

        # Verify writing: called twice (once per atom) with append=True
        assert mock_write.call_count == 2
        args, kwargs = mock_write.call_args
        assert kwargs.get("append") is True

        # Verify calculator was attached and used
        assert atoms1.calc == mock_calc
        assert atoms2.calc == mock_calc

    @patch("mlip_autopipec.infrastructure.espresso.adapter.iread")
    @patch("mlip_autopipec.infrastructure.espresso.adapter.write")
    @patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
    def test_recovery_strategy(
        self,
        mock_espresso: MagicMock,
        mock_write: MagicMock,
        mock_iread: MagicMock,
        oracle: "EspressoOracle",
        tmp_path: Path,
    ) -> None:
        """Test that recovery strategy is applied when calculation fails."""
        dataset_path = tmp_path / "input.xyz"
        dataset_path.touch()
        dataset = Dataset(file_path=dataset_path)

        import numpy as np

        atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10], pbc=True)
        mock_iread.return_value = [atoms]

        # Mock calculator to fail first, then succeed
        mock_calc = MagicMock()
        mock_espresso.return_value = mock_calc

        # Side effect for get_potential_energy: Fail, Fail, Success
        mock_calc.get_potential_energy.side_effect = [
            CalculatorError("Convergence NOT achieved"),
            CalculatorError("Convergence NOT achieved"),
            -15.0,
        ]
        mock_calc.get_forces.return_value = np.zeros((2, 3))
        mock_calc.get_stress.return_value = np.zeros(6)

        # Run label
        oracle.label(dataset)

        # Verify that calculator was re-initialized or parameters updated
        # We expect 3 calls to get_potential_energy
        assert mock_calc.get_potential_energy.call_count == 3

        # Verify write was called once (success)
        mock_write.assert_called_once()


@pytest.mark.skipif(RecoveryStrategy is None, reason="RecoveryStrategy not implemented")
def test_recovery_strategy_recipes() -> None:
    """Test that recovery strategy yields expected recipes."""
    strategy = RecoveryStrategy()
    recipes = list(strategy.get_recipes())

    # Should have at least the base recipe (empty dict or default) + recovery recipes
    assert len(recipes) >= 2

    # First recipe might be empty (default params)
    assert recipes[0] == {}

    # Second recipe should modify something like mixing_beta
    assert "mixing_beta" in recipes[1] or "smearing" in recipes[1]
