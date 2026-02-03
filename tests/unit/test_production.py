import json
import zipfile
from pathlib import Path

from mlip_autopipec.domain_models.production import ProductionManifest
from mlip_autopipec.infrastructure.production import ProductionDeployer


def test_manifest_creation() -> None:
    manifest = ProductionManifest(
        version="1.0.0",
        author="User",
        training_set_size=100,
        validation_metrics={"rmse_E": 0.001},
        creation_date="2023-10-27T10:00:00"
    )
    assert manifest.version == "1.0.0"
    assert manifest.training_set_size == 100

def test_deployer_creates_zip(tmp_path: Path) -> None:
    potential_path = tmp_path / "potential.yace"
    potential_path.write_text("potential data")

    report_path = tmp_path / "report.html"
    report_path.write_text("<html></html>")

    manifest_data = ProductionManifest(
        version="1.0.0",
        author="Tester",
        training_set_size=50,
        validation_metrics={},
        creation_date="2023-01-01"
    )

    deployer = ProductionDeployer(output_dir=tmp_path)
    zip_path = deployer.deploy(potential_path, manifest_data, report_path)

    assert zip_path.exists()
    assert zip_path.suffix == ".zip"

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert "potential.yace" in names
        assert "manifest.json" in names
        assert "report.html" in names
        assert "LICENSE" in names

        # Verify manifest content
        with z.open("manifest.json") as f:
            data = json.load(f)
            assert data["version"] == "1.0.0"
            assert data["author"] == "Tester"

def test_deployer_handles_missing_report(tmp_path: Path) -> None:
    potential_path = tmp_path / "potential.yace"
    potential_path.write_text("potential data")

    manifest_data = ProductionManifest(
        version="1.0.0",
        author="Tester",
        training_set_size=50,
        validation_metrics={},
        creation_date="2023-01-01"
    )

    deployer = ProductionDeployer(output_dir=tmp_path)
    zip_path = deployer.deploy(potential_path, manifest_data, report_path=None)

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert "report.html" not in names
