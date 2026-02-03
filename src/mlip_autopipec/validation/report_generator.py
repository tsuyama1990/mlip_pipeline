import logging
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mlip_autopipec.domain_models.validation import ValidationResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    @staticmethod
    def generate(result: ValidationResult, output_path: Path) -> None:
        """
        Generates an HTML report from the ValidationResult.
        """
        try:
            # Locate templates directory
            # Assuming templates is at project root (../../../../templates relative to this file?)
            # No, that's brittle.
            # Better: assume CWD or explicit config.
            # I'll check common locations.
            template_dir = Path("templates")
            if not template_dir.exists():
                # Fallback for installed package scenario
                template_dir = Path(__file__).parent.parent.parent.parent / "templates"
                if not template_dir.exists():
                    logger.warning("Templates directory not found. Skipping report generation.")
                    return

            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html", "xml"])
            )
            template = env.get_template("report_template.html")

            html_content = template.render(
                result=result,
                timestamp=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            )

            output_path.write_text(html_content)
            logger.info(f"Validation report saved to {output_path}")

        except Exception:
            logger.exception("Failed to generate validation report")
            # We don't want reporting failure to crash the pipeline
