from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_app() -> None:
    """Test the CLI."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
