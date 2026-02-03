from pathlib import Path
from unittest.mock import MagicMock

from mlip_autopipec.infrastructure.production import ProductionDeployer


def test_deploy_basic(tmp_path: Path) -> None:
    config = MagicMock()
    deployer = ProductionDeployer(config)

    potential_path = tmp_path / "pot.yace"
    potential_path.write_text("content")

    manifest = MagicMock()
    manifest.version = "1.0.0"
    manifest.model_dump_json.return_value = "{}"

    report_path = tmp_path / "report.html"
    report_path.write_text("html")

    output_dir = tmp_path / "release"

    # We do NOT patch ZipFile, so it writes to disk.
    zip_path = deployer.deploy(potential_path, manifest, report_path, output_dir)

    assert output_dir.exists()
    assert zip_path.exists()
    assert zip_path.name == "release_v1.0.0.zip"
