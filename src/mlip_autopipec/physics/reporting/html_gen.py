import base64
from pathlib import Path

from jinja2 import Environment, select_autoescape

from mlip_autopipec.domain_models.validation import ValidationResult

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report: {{ result.potential_id }}</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; }
        th { background-color: #f2f2f2; }
        .img-container { text-align: center; }
        img { max-width: 800px; }
    </style>
</head>
<body>
    <h1>Validation Report</h1>
    <p>Potential: {{ result.potential_id }}</p>
    <p>Status: <span class="{{ result.overall_status.lower() }}">{{ result.overall_status }}</span></p>

    <h2>Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Unit</th>
            <th>Passed</th>
        </tr>
        {% for m in result.metrics %}
        <tr>
            <td>{{ m.name }}</td>
            <td>{{ "%.4f"|format(m.value) }}</td>
            <td>{{ m.unit or "-" }}</td>
            <td class="{{ 'pass' if m.passed else 'fail' }}">{{ "PASS" if m.passed else "FAIL" }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Plots</h2>
    {% for name, path in result.plots.items() %}
    <div class="img-container">
        <h3>{{ name }}</h3>
        <img src="data:image/png;base64,{{ get_base64(path) }}" alt="{{ name }}">
    </div>
    {% endfor %}
</body>
</html>
"""


def generate_report(result: ValidationResult, output_path: Path) -> None:
    env = Environment(autoescape=select_autoescape(["html", "xml"]))

    def get_base64(path: Path) -> str:
        if not path.exists():
            return ""
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    template = env.from_string(TEMPLATE)
    html = template.render(result=result, get_base64=get_base64)

    output_path.write_text(html)
