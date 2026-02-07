from pathlib import Path

from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_run_success(temp_config: Path, tmp_path: Path) -> None:
    # Run from tmp_path to keep artifacts clean
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # We need to copy temp_config here or use absolute path
        # Using absolute path is easier
        result = runner.invoke(app, [str(temp_config)])
        assert result.exit_code == 0
        # Logging might not be captured in result.output depending on configuration
        # but exit code 0 means success.


def test_cli_file_not_found() -> None:
    result = runner.invoke(app, ["nonexistent.yaml"])
    assert result.exit_code == 1
    # Check stderr as well since we used err=True
    assert "Error: Configuration file nonexistent.yaml not found." in result.stderr
