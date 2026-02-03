import logging
import zipfile
from pathlib import Path

from mlip_autopipec.config.config_model import Config
from mlip_autopipec.domain_models.production import ProductionManifest

logger = logging.getLogger(__name__)

class ProductionDeployer:
    def __init__(self, config: Config):
        self.config = config

    def deploy(
        self,
        potential_path: Path,
        manifest: ProductionManifest,
        report_path: Path | None,
        output_dir: Path
    ) -> Path:
        """
        Creates a release package (zip file).
        Returns the path to the created zip file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define zip filename
        version = manifest.version
        zip_filename = f"release_v{version}.zip"
        zip_path = output_dir / zip_filename

        logger.info(f"Creating release package: {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Add Potential
            if potential_path.exists():
                zf.write(potential_path, arcname="potential.yace")
            else:
                logger.error(f"Potential file {potential_path} does not exist!")

            # 2. Add Manifest
            manifest_json = manifest.model_dump_json(indent=2)
            zf.writestr("manifest.json", manifest_json)

            # 3. Add Report
            if report_path and report_path.exists():
                zf.write(report_path, arcname="report.html")
            elif report_path:
                logger.warning(f"Report file {report_path} not found.")

            # 4. Add License (Placeholder)
            # We look for LICENSE file in current dir or root
            license_path = Path("LICENSE")
            if license_path.exists():
                zf.write(license_path, arcname="LICENSE")
            else:
                # Create a dummy license if missing
                zf.writestr("LICENSE", "Copyright (c) 2024 MLIP Pipeline")

        logger.info("Release package created successfully.")
        return zip_path
