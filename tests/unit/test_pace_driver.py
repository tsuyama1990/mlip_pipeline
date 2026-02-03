from io import StringIO
from unittest.mock import MagicMock, patch

from mlip_autopipec.inference import pace_driver


def test_pace_driver_logic() -> None:
    # Mock PaceCalculator directly on the module
    mock_calc_class = MagicMock()
    mock_calc_instance = mock_calc_class.return_value

    # Setup mock return values
    mock_results = MagicMock()
    mock_results.energy = -10.5
    mock_results.forces = [[0.1, 0.2, 0.3]]
    mock_results.gamma = 0.5  # Low uncertainty
    mock_calc_instance.calculate.return_value = mock_results

    # Patch the PaceCalculator in the module
    with patch.object(pace_driver, "PaceCalculator", mock_calc_class):
        # Prepare input
        input_data = """2
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3
Cu 0.0 0.0 0.0
Cu 1.5 1.5 1.5
"""
        # Mock stdin and stdout
        with (
            patch("sys.stdin", StringIO(input_data)),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            exit_code = pace_driver.run_driver()

            assert exit_code == 0
            output = mock_stdout.getvalue()

            assert "-1.0500000000000000e+01" in output
            assert "1.0000000000000001e-01" in output


def test_pace_driver_high_uncertainty() -> None:
    mock_calc_class = MagicMock()
    mock_calc_instance = mock_calc_class.return_value

    mock_results = MagicMock()
    mock_results.gamma = 10.0  # High uncertainty
    mock_calc_instance.calculate.return_value = mock_results

    with patch.object(pace_driver, "PaceCalculator", mock_calc_class):
        input_data = """2
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3
Cu 0.0 0.0 0.0
Cu 1.5 1.5 1.5
"""
        with (
            patch("sys.stdin", StringIO(input_data)),
            patch("sys.stdout", new_callable=StringIO),
        ):
            # Should return 100
            exit_code = pace_driver.run_driver()
            assert exit_code == 100


def test_pace_driver_missing_calculator() -> None:
    # Patch PaceCalculator to be None
    with (
        patch.object(pace_driver, "PaceCalculator", None),
        patch("sys.stdin", StringIO("dummy")),
        patch("mlip_autopipec.inference.pace_driver.read_eon_geometry", return_value=MagicMock()),
        patch("sys.stderr", new_callable=StringIO) as mock_stderr,
    ):
        exit_code = pace_driver.run_driver()
        assert exit_code == 1
        assert "pyacemaker not installed" in mock_stderr.getvalue()


def test_pace_driver_read_fail() -> None:
    # Force read to fail (both attempts)
    with (
        patch("mlip_autopipec.inference.pace_driver.read", side_effect=Exception("Bad format")),
        patch(
            "mlip_autopipec.inference.pace_driver.read_eon_geometry",
            side_effect=Exception("Bad EON"),
        ),
        patch("sys.stdin", StringIO("bad data")),
        patch("sys.stderr", new_callable=StringIO),
    ):
        exit_code = pace_driver.run_driver()
        assert exit_code == 1


def test_pace_driver_eon_format() -> None:
    # Test strict EON format
    input_data = """2
10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0
Cu 0.0 0.0 0.0
Cu 1.5 1.5 1.5
"""
    mock_calc_class = MagicMock()
    mock_calc_instance = mock_calc_class.return_value
    mock_results = MagicMock()
    mock_results.energy = -10.0
    mock_results.forces = [[0, 0, 0]]
    mock_results.gamma = 0.0  # Low uncertainty
    mock_calc_instance.calculate.return_value = mock_results

    with (
        patch.object(pace_driver, "PaceCalculator", mock_calc_class),
        patch("sys.stdin", StringIO(input_data)),
        patch("sys.stdout", new_callable=StringIO),
    ):
        exit_code = pace_driver.run_driver()
        assert exit_code == 0
