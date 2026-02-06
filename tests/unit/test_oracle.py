import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.config.config_model import OracleConfig
# We will import EspressoOracle inside the test functions or using a try-except block
# if we want to run config tests before implementation exists.
# But for TDD, we usually assume the class exists or we are about to create it.
# Let's assume we will create the skeleton immediately after.
try:
    from mlip_autopipec.infrastructure.espresso.adapter import EspressoOracle
except ImportError:
    EspressoOracle = None  # type: ignore


class TestOracleConfig:
    def test_oracle_config_valid_mock(self) -> None:
        config = OracleConfig(type="mock")
        assert config.type == "mock"

    def test_oracle_config_valid_espresso(self) -> None:
        config = OracleConfig(
            type="espresso",
            command="mpirun -np 4 pw.x",
            pseudo_dir=Path("/tmp/pseudo"),
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

        # Check that validation error mentions missing fields
        # Note: Pydantic aggregates errors, so we might check for one of them
        assert "Field 'command' is required" in str(excinfo.value)

    def test_oracle_config_security_check(self) -> None:
        """
        Command should not contain dangerous characters.
        """
        with pytest.raises(ValidationError) as excinfo:
            OracleConfig(
                type="espresso",
                command="pw.x; echo hack",
                pseudo_dir=Path("/tmp"),
                pseudopotentials={}
            )
        assert "Command contains forbidden characters" in str(excinfo.value)

        with pytest.raises(ValidationError) as excinfo:
            OracleConfig(
                type="espresso",
                command="pw.x | bash",
                pseudo_dir=Path("/tmp"),
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

    @patch("mlip_autopipec.infrastructure.espresso.adapter.write")
    @patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
    def test_label_streaming(
        self,
        mock_espresso_cls: MagicMock,
        mock_write: MagicMock,
        mock_config: OracleConfig,
        tmp_path: Path
    ) -> None:
        """
        Verifies that label method:
        1. Reads input file using streaming (mocked read returning iterator).
        2. Processes atoms in batches.
        3. Writes to output file.
        """
        oracle = EspressoOracle(mock_config, tmp_path)

        # Mock dataset
        dataset_path = tmp_path / "input.xyz"
        # Write dummy content so size > 0 check passes
        dataset_path.write_text("dummy")
        dataset = MagicMock()
        dataset.file_path = dataset_path

        # Mock atoms
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.1]])
        atoms3 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.2]])

        with patch("mlip_autopipec.infrastructure.espresso.adapter.iread") as mock_iread:
            mock_iread.return_value = iter([atoms1, atoms2, atoms3])

            # Mock Espresso instance
            mock_calc = MagicMock()
            mock_calc.get_potential_energy.return_value = -10.0
            mock_calc.get_forces.return_value = [[0, 0, 0], [0, 0, 0]]
            mock_calc.get_stress.return_value = [0]*6
            mock_espresso_cls.return_value = mock_calc

            result_dataset = oracle.label(dataset)

            # Verify iread was called on the input file
            mock_iread.assert_called_with(dataset_path, index=":")

            # Verify calculator was attached to atoms
            assert mock_espresso_cls.call_count >= 3

            # Verify write was called
            # Should be called to write results.
            # If batching is 2, and we have 3 atoms:
            # might write 2, then 1. Or all 3 if buffered differently.
            assert mock_write.called

            # Verify output file exists (in logic)
            # Since we mocked write, the file won't be written by write(), but logic should produce a path
            assert result_dataset.file_path.name.startswith("labeled_")

    def test_validate_command_security(self, mock_config: OracleConfig, tmp_path: Path) -> None:
        """
        Test that _validate_command raises error for forbidden commands.
        (Although config validates it, the adapter might have extra checks or use the config validator)
        """
        # Create a config with a "safe" command that might pass Pydantic but fail stricter adapter checks
        # or just verify the Pydantic check is respected.
        # If we bypass Pydantic, the adapter should re-check?
        # The adapter takes a validated Config object, so it assumes it's safe-ish.
        # But let's check if adapter does extra validation as implied by "Security" feedback.

        # Attempt to modify config to unsafe after validation (programmatically)
        mock_config.command = "pw.x; echo hack"
        # Note: Pydantic models are mutable by default.

        with pytest.raises(ValueError, match="forbidden"):
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
    def test_recovery_strategy_all_fail(self, mock_espresso_cls: MagicMock, mock_config: OracleConfig, tmp_path: Path) -> None:
        """
        Test that if all recovery attempts fail, the error is propagated (caught by label loop).
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
            # Always fail
            mock_calc.get_potential_energy.side_effect = CalculatorError("SCF failed forever")
            mock_espresso_cls.return_value = mock_calc

            # label() catches exceptions and continues, so it won't raise.
            # But we want to ensure it tried multiple times.
            oracle.label(dataset)

            # 1 base + 4 recipes = 5 calls
            assert mock_calc.get_potential_energy.call_count >= 5

    @patch("mlip_autopipec.infrastructure.espresso.recovery.logger")
    @patch("mlip_autopipec.infrastructure.espresso.adapter.logger")
    @patch("mlip_autopipec.infrastructure.espresso.adapter.write")
    @patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
    def test_label_exception_handling(
        self,
        mock_espresso_cls: MagicMock,
        mock_write: MagicMock,
        mock_adapter_logger: MagicMock,
        mock_recovery_logger: MagicMock,
        mock_config: OracleConfig,
        tmp_path: Path
    ) -> None:
        """
        Test that if labeling a structure fails completely, it is skipped and logged,
        but process continues.
        """
        oracle = EspressoOracle(mock_config, tmp_path)

        dataset_path = tmp_path / "input.xyz"
        dataset_path.write_text("dummy")
        dataset = MagicMock()
        dataset.file_path = dataset_path

        atoms1 = Atoms("H")
        atoms2 = Atoms("He")

        with patch("mlip_autopipec.infrastructure.espresso.adapter.iread") as mock_iread:
            mock_iread.return_value = iter([atoms1, atoms2])

            mock_calc = MagicMock()
            # First atom fails all attempts (RuntimeError raised by recovery after retries exhausted)
            # Second atom succeeds

            # We need to simulate recovery failure.
            # RecoveryStrategy raises exception if all attempts fail.
            # So if we make get_potential_energy ALWAYS fail:
            mock_calc.get_potential_energy.side_effect = Exception("Fatal Error")

            mock_espresso_cls.return_value = mock_calc

            # We expect label() to log error and continue.
            # But wait, if ALL recovery attempts fail, it raises exception.
            # The adapter loop catches exception and logs it, then continues.
            # So we should see successful completion of label(), but only 0 structures labeled (if both fail)
            # Let's make first fail, second succeed.

            # Side effect needs to handle multiple calls per atom due to retries.
            # Atom 1: Fails -> Retry 1 Fail -> ... -> Retry N Fail -> Raise
            # Atom 2: Succeeds

            # Easier to patch RecoveryStrategy? No, implementation uses it.
            # We can use side_effect with an iterator.

            # Assuming 4 retries in recovery + 1 base = 5 calls per failure.
            # We have 4 recipes + 1 base = 5 calls.
            failures = [Exception("Fail")] * 10 # Enough for one atom to fail completely
            success = [-10.0]

            # But wait, side_effect is called for each atom.
            # We need to distinguish atoms.
            # The calculator is re-instantiated for each atom.
            # So mock_espresso_cls is called twice.
            # We can return different mock calculators.

            mock_calc_fail = MagicMock()
            mock_calc_fail.get_potential_energy.side_effect = Exception("Fatal")

            mock_calc_success = MagicMock()
            mock_calc_success.get_potential_energy.return_value = -10.0
            mock_calc_success.get_forces.return_value = [[0,0,0]]
            mock_calc_success.get_stress.return_value = [0]*6

            mock_espresso_cls.side_effect = [mock_calc_fail, mock_calc_success]

            # Use side_effect to capture written atoms before buffer is cleared
            written_atoms = []
            def write_side_effect(filename, atoms, **kwargs):
                # Copy atoms because buffer is cleared in place
                written_atoms.extend(list(atoms))
            mock_write.side_effect = write_side_effect

            result_dataset = oracle.label(dataset)

            # Should have written 1 structure (the successful one)
            assert len(written_atoms) == 1
            assert written_atoms[0].get_chemical_formula() == "He"
