import logging
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Template

from mlip_autopipec.domain_models.validation import ValidationResult

logger = logging.getLogger(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .passed { color: green; font-weight: bold; }
        .failed { color: red; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .module-section { margin-bottom: 40px; border: 1px solid #ccc; padding: 20px; border-radius: 5px; }
        .details-box { background-color: #f9f9f9; padding: 10px; border-left: 5px solid #ccc; font-family: monospace; white-space: pre-wrap; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Validation Report</h1>
    <p><strong>Date:</strong> {{ date }}</p>
    <p><strong>Potential:</strong> {{ potential }}</p>

    <h2>Summary</h2>
    <table>
        <tr>
            <th>Module</th>
            <th>Status</th>
            <th>Metrics</th>
        </tr>
        {% for res in results %}
        <tr>
            <td>{{ res.module }}</td>
            <td class="{{ 'passed' if res.passed else 'failed' }}">
                {{ 'PASS' if res.passed else 'FAIL' }}
            </td>
            <td>
                <ul>
                {% for m in res.metrics %}
                    <li>{{ m.name }}: {{ m.value }} {{ m.unit or '' }}
                        {% if not m.passed %} <span class="failed">(X)</span> {% endif %}
                    </li>
                {% endfor %}
                </ul>
                {% if res.error %}
                <div class="failed">Error: {{ res.error }}</div>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>

    <h2>Details</h2>
    {% for res in results %}
    <div class="module-section">
        <h3>{{ res.module }} <span class="{{ 'passed' if res.passed else 'failed' }}">({{ 'PASS' if res.passed else 'FAIL' }})</span></h3>

        {% if res.error %}
            <p class="failed">Error: {{ res.error }}</p>
        {% endif %}

        {% for m in res.metrics %}
            <h4>{{ m.name }}</h4>
            <div class="details-box">
                {% for k, v in m.details.items() %}
                    {% if k != 'plot_path' %}
                    {{ k }}: {{ v }}<br>
                    {% endif %}
                {% endfor %}
            </div>
            {% if m.details.get('plot_path') %}
                <div>
                    <strong>Plot:</strong><br>
                    <img src="{{ m.details.get('plot_path') }}" alt="{{ m.name }} plot">
                </div>
            {% endif %}
        {% endfor %}
    </div>
    {% endfor %}

</body>
</html>
"""


class ReportGenerator:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results: list[ValidationResult], potential_path: Path) -> Path:
        logger.info("Generating validation report...")

        template = Template(TEMPLATE)

        html_content = template.render(
            date=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            potential=str(potential_path),
            results=results,
        )

        report_path = self.work_dir / "validation_report.html"
        report_path.write_text(html_content)

        logger.info(f"Report saved to {report_path}")
        return report_path
