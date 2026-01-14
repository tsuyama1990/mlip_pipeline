from typer.testing import CliRunner

from mlip_autopipec.cli import app

runner = CliRunner()


def test_app() -> None:
    """Test the CLI."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
