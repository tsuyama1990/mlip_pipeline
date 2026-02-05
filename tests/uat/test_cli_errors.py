import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_config_validation_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config_data = {
        "execution_mode": "mock",
        "max_cycles": "not_a_number",
    }
    config_file = tmp_path / "bad_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_cli_production_fallback(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config_data = {
        "execution_mode": "production",
        "max_cycles": 1,
        "exploration": {"strategy_name": "random"},
        "dft": {"calculator": "espresso"},
        "training": {"fitting_code": "pacemaker"},
    }
    config_file = tmp_path / "prod_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Production mode not implemented yet" in result.stdout
