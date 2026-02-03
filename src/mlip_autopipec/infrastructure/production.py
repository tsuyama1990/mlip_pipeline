import logging
import zipfile
from pathlib import Path

from mlip_autopipec.domain_models.production import ProductionManifest

logger = logging.getLogger(__name__)

class ProductionDeployer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def deploy(self,
               potential_path: Path,
               manifest_data: ProductionManifest,
               report_path: Path | None = None) -> Path:

        version = manifest_data.version
        zip_name = f"release_v{version}.zip"
        zip_path = self.output_dir / zip_name

        logger.info(f"Creating production release: {zip_path}")

        with zipfile.ZipFile(zip_path, "w") as zf:
            # Add Potential
            if potential_path.exists():
                zf.write(potential_path, arcname=potential_path.name)
            else:
                logger.warning(f"Potential file not found: {potential_path}")

            # Add Manifest
            manifest_json = manifest_data.model_dump_json(indent=2)
            zf.writestr("manifest.json", manifest_json)

            # Add Report
            if report_path and report_path.exists():
                zf.write(report_path, arcname="report.html")

            # Add License (Dummy for now)
            zf.writestr("LICENSE", "MIT License\n\nCopyright (c) 2023")

        return zip_path
