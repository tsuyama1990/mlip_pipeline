"""Validation report generator."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from pyacemaker.domain_models.validator import ValidationResult


class ReportGenerator:
    """Generates HTML validation report."""

    def __init__(self, template_dir: Path | None = None) -> None:
        """Initialize report generator."""
        if template_dir is None:
            # Assume templates are in same directory as this file + /templates
            template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)

    def generate(self, result: ValidationResult, output_path: Path) -> None:
        """Generate HTML report."""
        template = self.env.get_template("report.html")
        content = template.render(result=result)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
