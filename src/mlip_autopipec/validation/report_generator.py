import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from mlip_autopipec.domain_models.validation import ValidationResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self) -> None:
        template_dir = Path(__file__).parent.parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

    def generate(self, result: ValidationResult, work_dir: Path) -> Path:
        """
        Generates an HTML report for the validation result.

        Args:
            result: The validation result object.
            work_dir: The directory where the report will be saved.

        Returns:
            The path to the generated HTML report.
        """
        try:
            template = self.env.get_template("report_template.html")

            # Prepare data
            report_data = {
                "potential_name": "Potential",  # Could be improved to accept name
                "passed": result.passed,
                "reason": result.reason,
                "metrics": [],
            }

            for m in result.metrics:
                metric_dict = m.model_dump()
                if m.plot_path:
                    # Make relative path for HTML so it works if folder moves
                    try:
                        metric_dict["plot_path"] = Path(m.plot_path).relative_to(
                            work_dir
                        )
                    except ValueError:
                        # Fallback if not relative (e.g. absolute path elsewhere)
                        metric_dict["plot_path"] = m.plot_path
                report_data["metrics"].append(metric_dict)  # type: ignore

            html_content = template.render(report_data)

            report_path = work_dir / "report.html"
            report_path.write_text(html_content)
            logger.info(f"Validation report generated at {report_path}")

        except Exception:
            logger.exception("Failed to generate report")
            raise

        return report_path
