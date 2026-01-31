from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import base64
from typing import Optional

from mlip_autopipec.domain_models.validation import ValidationResult

class ReportGenerator:
    def __init__(self, template_dir: Path = Path("src/mlip_autopipec/physics/reporting/templates")):
        self.template_dir = template_dir
        self.env: Optional[Environment] = None
        if template_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )

    def generate(self, result: ValidationResult, output_path: Path) -> None:
        template_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report</title>
            <style>
                body { font-family: sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .pass { background-color: #d4edda; }
                .fail { background-color: #f8d7da; }
                .warn { background-color: #fff3cd; }
            </style>
        </head>
        <body>
            <h1>Validation Report: {{ result.potential_id }}</h1>
            <h2>Overall Status: <span class="{{ 'pass' if result.overall_status == 'PASS' else 'fail' }}">{{ result.overall_status }}</span></h2>

            <h3>Metrics</h3>
            <table>
                <tr><th>Name</th><th>Value</th><th>Passed</th><th>Message</th></tr>
                {% for metric in result.metrics %}
                <tr class="{{ 'pass' if metric.passed else 'fail' }}">
                    <td>{{ metric.name }}</td>
                    <td>{{ "%.4f"|format(metric.value) }}</td>
                    <td>{{ metric.passed }}</td>
                    <td>{{ metric.message }}</td>
                </tr>
                {% endfor %}
            </table>

            <h3>Plots</h3>
            {% for name, path in result.plots.items() %}
                <h4>{{ name }}</h4>
                <img src="data:image/png;base64,{{ get_base64_image(path) }}" width="800">
            {% endfor %}
        </body>
        </html>
        """

        if self.env and (self.template_dir / "report.html").exists():
            template = self.env.get_template("report.html")
            html_out = template.render(result=result, get_base64_image=self._get_base64_image)
        else:
            template = Template(template_content)
            html_out = template.render(result=result, get_base64_image=self._get_base64_image)

        output_path.write_text(html_out)

    def _get_base64_image(self, path: Path) -> str:
        if not path.exists():
            return ""
        data = path.read_bytes()
        return base64.b64encode(data).decode('utf-8')
