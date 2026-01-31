from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from mlip_autopipec.domain_models.validation import ValidationResult


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        # Assuming the template file is in src/mlip_autopipec/physics/reporting/templates/
        self.template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))

    def generate(self, result: ValidationResult) -> Path:
        template = self.env.get_template("report.html")
        html_content = template.render(result=result)

        report_path = self.output_dir / "validation_report.html"
        report_path.write_text(html_content)
        return report_path
