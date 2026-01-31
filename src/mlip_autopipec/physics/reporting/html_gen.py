import base64
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from mlip_autopipec.domain_models.validation import ValidationResult


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir), autoescape=True
        )

    def generate(
        self, result: ValidationResult, filename: str = "validation_report.html"
    ) -> Path:
        template = self.env.get_template("report.html")

        plots_base64 = {}
        for name, path in result.plots.items():
            if path.exists():
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                    plots_base64[name] = f"data:image/png;base64,{encoded}"

        html = template.render(result=result, plots=plots_base64)

        output_path = self.output_dir / filename
        output_path.write_text(html)
        return output_path
