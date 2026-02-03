import logging
import zipfile
from pathlib import Path

from mlip_autopipec.domain_models.production import ProductionManifest

logger = logging.getLogger(__name__)


class ProductionDeployer:
    def deploy(
        self,
        potential_path: Path,
        manifest: ProductionManifest,
        report_path: Path | None,
        output_dir: Path,
    ) -> Path:
        """
        Creates a production release zip.
        """
        version = manifest.version
        release_name = f"release_v{version}"
        zip_path = output_dir / f"{release_name}.zip"

        logger.info(f"Creating production release: {zip_path}")

        with zipfile.ZipFile(zip_path, "w") as zf:
            # Add potential
            zf.write(potential_path, arcname="potential.yace")

            # Add manifest
            manifest_json = manifest.model_dump_json(indent=2)
            zf.writestr("manifest.json", manifest_json)

            # Add report
            if report_path and report_path.exists():
                zf.write(report_path, arcname="report.html")

            # Add License
            license_path = Path("LICENSE")
            if license_path.exists():
                zf.write(license_path, arcname="LICENSE")
            else:
                logger.warning("LICENSE file not found in root.")

        return zip_path
