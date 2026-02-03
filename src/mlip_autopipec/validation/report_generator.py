from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from mlip_autopipec.domain_models.validation import ValidationResult


class ReportGenerator:
    @staticmethod
    def generate(result: ValidationResult, work_dir: Path) -> Path:
        """
        Generates an HTML report from the validation result.
        """
        # Determine template directory relative to project root (source layout)
        # src/mlip_autopipec/validation/report_generator.py -> ../../../../templates
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        template_dir = project_root / "templates"

        # If not found (e.g. installed package), might need another strategy.
        # For now, we assume development environment structure.
        if not template_dir.exists():
            # Try finding it relative to cwd if running from root
            template_dir = Path("templates").resolve()

        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
        template = env.get_template("report_template.html")

        html_content = template.render(
            passed=result.passed,
            metrics=result.metrics,
            reason=result.reason,
        )

        report_path = work_dir / "validation_report.html"
        report_path.write_text(html_content)

        return report_path
