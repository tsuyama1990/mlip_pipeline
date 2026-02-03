import json
import zipfile
from unittest.mock import MagicMock

from mlip_autopipec.config.config_model import Config
from mlip_autopipec.domain_models.production import ProductionManifest
from mlip_autopipec.infrastructure.production import ProductionDeployer


def test_deployer_creates_zip(tmp_path):
    # Setup
    config = MagicMock(spec=Config)
    deployer = ProductionDeployer(config)

    potential_path = tmp_path / "final.yace"
    potential_path.write_text("potential data")

    report_path = tmp_path / "report.html"
    report_path.write_text("<html></html>")

    manifest = ProductionManifest(
        version="1.0.0",
        author="Tester",
        training_set_size=100,
        validation_metrics={"score": 1.0}
    )

    output_dir = tmp_path / "release"
    output_dir.mkdir()

    # Execute
    zip_path = deployer.deploy(
        potential_path=potential_path,
        manifest=manifest,
        report_path=report_path,
        output_dir=output_dir
    )

    # Verify
    assert zip_path.exists()
    assert zip_path.suffix == ".zip"

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert "potential.yace" in names
        assert "manifest.json" in names
        assert "report.html" in names

        # Check manifest content
        with z.open("manifest.json") as f:
            data = json.load(f)
            assert data["version"] == "1.0.0"
            assert data["author"] == "Tester"
