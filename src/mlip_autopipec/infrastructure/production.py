import zipfile
import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.domain_models.production import ProductionManifest
from mlip_autopipec.domain_models.validation import ValidationResult

logger = logging.getLogger("mlip_autopipec.infrastructure.production")

class ProductionDeployer:
    """
    Handles packaging of the potential for production/release.
    """

    def __init__(self, config: Config, output_dir: Path = Path("dist")):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def deploy(
        self,
        state: WorkflowState,
        version: str,
        author: str,
        description: str = "Automated Release"
    ) -> Path:
        """
        Creates a distribution zip file.
        """
        logger.info(f"Preparing production release {version} by {author}...")

        # 1. Locate Artifacts
        potential_path = state.latest_potential_path
        if not potential_path or not potential_path.exists():
            raise FileNotFoundError("No potential file found in state.")

        # Locate Validation Report
        report_path = self.config.validation.report_path
        if not report_path.exists():
             logger.warning(f"Validation report not found at {report_path}")
             # Try absolute?
             pass

        # 2. Get Validation Metrics
        # Find the result corresponding to this potential
        # The state tracks validation history by cycle (generation).
        # We assume the last validation result applies.
        last_gen = max(state.validation_history.keys()) if state.validation_history else None

        validation_result: ValidationResult
        if last_gen and last_gen in state.validation_history:
            validation_result = state.validation_history[last_gen]
        else:
            # Create a dummy result if none exists (e.g. forced deploy)
            # Audit Requirement: "Add validation to ensure the validation history is not empty before accessing it."
            # We are handling the empty case here by creating a dummy result with WARN status.
            logger.warning("No validation history found. Using empty metrics.")
            validation_result = ValidationResult(
                potential_id=str(potential_path),
                metrics=[],
                overall_status="WARN"
            )

        # 3. Create Manifest
        # Training set size is not easily available in state unless we track it.
        # We assume None if not tracked, compliant with schema optionality.
        training_size: Optional[int] = None
        if state.dataset_path and state.dataset_path.exists():
            # TODO: Implement fast counting logic if needed.
            pass

        manifest = ProductionManifest(
            version=version,
            author=author,
            training_set_size=training_size,
            validation_metrics=validation_result,
            description=description
        )

        # 4. Package
        # Naming convention configurable via ValidationConfig (where reporting settings live)
        # or we should have a separate ProductionConfig. For now, ValidationConfig is closest fit or defaults.
        # But wait, ValidationConfig is for Validation.
        # I added package_name_format to ValidationConfig in previous step as it's part of 'release'.
        fmt = self.config.validation.package_name_format
        try:
            zip_name = fmt.format(version=version, author=author)
        except KeyError:
            # Fallback if format string is invalid or missing keys
            logger.warning(f"Invalid package name format '{fmt}'. Using default.")
            zip_name = f"mlip_package_{version}.zip"

        zip_path = self.output_dir / zip_name

        with zipfile.ZipFile(zip_path, "w") as zf:
            # Add Potential
            zf.write(potential_path, arcname="potential.yace")

            # Add Report
            if report_path.exists():
                zf.write(report_path, arcname="validation_report.html")

            # Add Manifest
            zf.writestr("metadata.json", manifest.model_dump_json(indent=2))

            logger.info(f"Packaged {zip_path}")

        return zip_path
