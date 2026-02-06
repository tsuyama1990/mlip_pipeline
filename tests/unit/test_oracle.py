from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.config.config_model import OracleConfig

try:
    from mlip_autopipec.infrastructure.espresso.adapter import EspressoOracle
except ImportError:
    EspressoOracle = None  # type: ignore


class TestOracleConfig:
    def test_oracle_config_valid_mock(self) -> None:
        config = OracleConfig(type="mock")
        assert config.type == "mock"

    def test_oracle_config_valid_espresso(self, tmp_path: Path) -> None:
        config = OracleConfig(
            type="espresso",
            command="mpirun -np 4 pw.x",
            pseudo_dir=tmp_path / "pseudo",
            pseudopotentials={"Si": "Si.upf"},
            batch_size=5
        )
        assert config.type == "espresso"
        assert config.command == "mpirun -np 4 pw.x"
        assert config.batch_size == 5

    def test_oracle_config_invalid_espresso_missing_fields(self) -> None:
        """
        Espresso config requires command, pseudo_dir, pseudopotentials.
        """
        with pytest.raises(ValidationError) as excinfo:
            OracleConfig(type="espresso")

        assert "Field 'command' is required" in str(excinfo.value)

    def test_oracle_config_security_check(self, tmp_path: Path) -> None:
        """
        Command should not contain dangerous characters.
        """
        with pytest.raises(ValidationError) as excinfo:
            OracleConfig(
                type="espresso",
                command="pw.x; echo hack",
                pseudo_dir=tmp_path / "pseudo",
                pseudopotentials={}
            )
        assert "Command contains forbidden characters" in str(excinfo.value)

        with pytest.raises(ValidationError) as excinfo:
            OracleConfig(
                type="espresso",
                command="pw.x | bash",
                pseudo_dir=tmp_path / "pseudo",
                pseudopotentials={}
            )
        assert "Command contains forbidden characters" in str(excinfo.value)


@pytest.mark.skipif(EspressoOracle is None, reason="EspressoOracle not implemented yet")
class TestEspressoOracle:
    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> OracleConfig:
        return OracleConfig(
            type="espresso",
            command="pw.x",
            pseudo_dir=tmp_path / "pseudo",
            pseudopotentials={"H": "H.upf"},
            batch_size=2
        )

    def test_init(self, mock_config: OracleConfig, tmp_path: Path) -> None:
        oracle = EspressoOracle(mock_config, tmp_path)
        assert oracle.config == mock_config
        assert oracle.work_dir == tmp_path

    @patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
    def test_label_streaming_real_io(
        self,
        mock_espresso_cls: MagicMock,
        mock_config: OracleConfig,
        tmp_path: Path
    ) -> None:
        """
        Verifies that label method processes structures and writes to a REAL file.
        This tests the streaming and batching logic without mocking 'write'.
        """
        oracle = EspressoOracle(mock_config, tmp_path)

        # Create input dataset
        dataset_path = tmp_path / "input.xyz"
        from ase.io import write

        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        atoms2 = Atoms("He", positions=[[0, 0, 0]])
        atoms3 = Atoms("Li", positions=[[0, 0, 0]])

        write(dataset_path, [atoms1, atoms2, atoms3])

        dataset = MagicMock()
        dataset.file_path = dataset_path

        # Mock Espresso behavior
        mock_calc = MagicMock()
        mock_calc.get_potential_energy.return_value = -10.0
        # Fix: ensure forces shape matches atoms
        def get_forces_side_effect(atoms: Atoms) -> list[list[float]]:
            return [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
        mock_calc.get_forces.side_effect = get_forces_side_effect
        mock_calc.get_stress.return_value = [0]*6
        mock_espresso_cls.return_value = mock_calc

        # Run label
        result_dataset = oracle.label(dataset)

        # Verify output file exists and has content
        assert result_dataset.file_path.exists()
        assert result_dataset.file_path.stat().st_size > 0

        # Read back results using ASE
        from ase.io import read
        labeled_structures = read(result_dataset.file_path, index=":")
        assert isinstance(labeled_structures, list)
        assert len(labeled_structures) == 3
        assert labeled_structures[0].get_chemical_formula() == "H2"  # type: ignore[no-untyped-call]
        assert labeled_structures[1].get_chemical_formula() == "He"  # type: ignore[no-untyped-call]
        assert labeled_structures[2].get_chemical_formula() == "Li"  # type: ignore[no-untyped-call]

        # Verify batching (implicitly via file content correctness)
        # Verify calculator was called 3 times
        assert mock_espresso_cls.call_count >= 3

    def test_validate_command_security(self, mock_config: OracleConfig, tmp_path: Path) -> None:
        """
        Test whitelist and blacklist validation.
        """
        # Test Whitelist
        mock_config.command = "evil_script.sh"
        with pytest.raises(ValueError, match="whitelist"):
            EspressoOracle(mock_config, tmp_path)

        # Test Blacklist (redundant if whitelist works, but good for safety)
        mock_config.command = "pw.x; rm -rf /"
        with pytest.raises(ValueError, match="forbidden"):
            EspressoOracle(mock_config, tmp_path)

        # Test valid command
        mock_config.command = "mpirun -np 4 pw.x"
        # Should not raise (assuming whitelist logic handles 'mpirun')
        EspressoOracle(mock_config, tmp_path)

    def test_validate_command_sensitive(self, mock_config: OracleConfig, tmp_path: Path) -> None:
        """
        Test sensitive path detection.
        """
        mock_config.command = "pw.x /etc/passwd"
        with pytest.raises(ValueError, match="suspicious paths"):
            EspressoOracle(mock_config, tmp_path)

    @patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
    def test_recovery_strategy_success(self, mock_espresso_cls: MagicMock, mock_config: OracleConfig, tmp_path: Path) -> None:
        """
        Test that if calculator raises error, recovery strategy is triggered and succeeds.
        """
        oracle = EspressoOracle(mock_config, tmp_path)

        dataset_path = tmp_path / "input.xyz"
        dataset_path.write_text("dummy")
        dataset = MagicMock()
        dataset.file_path = dataset_path

        atoms = Atoms("H")
        from ase.calculators.calculator import CalculatorError

        with patch("mlip_autopipec.infrastructure.espresso.adapter.iread") as mock_iread:
            mock_iread.return_value = iter([atoms])
            mock_calc = MagicMock()
            # First call fails, second succeeds
            mock_calc.get_potential_energy.side_effect = [CalculatorError("SCF failed"), -15.0]
            mock_calc.get_forces.return_value = [[0, 0, 0]]
            mock_calc.get_stress.return_value = [0]*6
            mock_espresso_cls.return_value = mock_calc

            oracle.label(dataset)

            # Check that get_potential_energy was called twice
            assert mock_calc.get_potential_energy.call_count == 2

    @patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
    def test_recovery_strategy_all_fail_continuation(
        self,
        mock_espresso_cls: MagicMock,
        mock_config: OracleConfig,
        tmp_path: Path
    ) -> None:
        """
        Test that if all recovery attempts fail, the loop continues to the next structure.
        """
        oracle = EspressoOracle(mock_config, tmp_path)

        # Create input file with 2 atoms
        dataset_path = tmp_path / "input.xyz"
        from ase.io import write
        atoms1 = Atoms("H")
        atoms2 = Atoms("He")
        write(dataset_path, [atoms1, atoms2])

        dataset = MagicMock()
        dataset.file_path = dataset_path

        from ase.calculators.calculator import CalculatorError

        mock_calc_fail = MagicMock()
        mock_calc_fail.get_potential_energy.side_effect = CalculatorError("SCF failed forever")

        mock_calc_success = MagicMock()
        mock_calc_success.get_potential_energy.return_value = -20.0
        mock_calc_success.get_forces.return_value = [[0,0,0]]
        mock_calc_success.get_stress.return_value = [0]*6

        # We need side_effect on the CLASS to return different calculators for each instantiation
        # label loop creates new calculator for each atom (and each retry!)
        # First atom (H) fails 1 base + 4 retries = 5 times
        # Second atom (He) succeeds immediately = 1 time
        mock_espresso_cls.side_effect = [mock_calc_fail] * 5 + [mock_calc_success]

        result_dataset = oracle.label(dataset)

        # Verify result file contains only 1 structure (the successful one)
        from ase.io import read
        if result_dataset.file_path.stat().st_size > 0:
            results = read(result_dataset.file_path, index=":")
            if not isinstance(results, list):
                results = [results]
        else:
            results = []

        assert len(results) == 1
        assert results[0].get_chemical_formula() == "He"  # type: ignore[no-untyped-call]
