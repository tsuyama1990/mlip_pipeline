from pathlib import Path

from jinja2 import Environment

from mlip_autopipec.domain_models.validation import ValidationResult


class ReportGenerator:
    @staticmethod
    def generate(result: ValidationResult, output_path: Path) -> None:
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report</title>
            <style>
                body { font-family: sans-serif; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .pass { background-color: #d4edda; }
                .fail { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <h1>Validation Report</h1>
            <p><strong>Potential ID:</strong> {{ result.potential_id }}</p>
            <p><strong>Overall Status:</strong> <span class="{{ 'pass' if result.overall_status == 'PASS' else 'fail' }}">{{ result.overall_status }}</span></p>

            <h2>Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Reference</th>
                    <th>Passed</th>
                    <th>Error/Msg</th>
                </tr>
                {% for m in result.metrics %}
                <tr class="{{ 'pass' if m.passed else 'fail' }}">
                    <td>{{ m.name }}</td>
                    <td>{{ "%.4f"|format(m.value) }}</td>
                    <td>{{ m.reference if m.reference is not none else '-' }}</td>
                    <td>{{ m.passed }}</td>
                    <td>{{ m.error if m.error else '-' }}</td>
                </tr>
                {% endfor %}
            </table>

            <h2>Plots</h2>
            {% for name, path in result.plots.items() %}
            <div class="plot">
                <h3>{{ name }}</h3>
                <img src="{{ path }}" alt="{{ name }}" style="max_width: 100%;">
            </div>
            {% endfor %}
        </body>
        </html>
        """

        env = Environment(autoescape=True)
        template = env.from_string(template_str)
        html = template.render(result=result)

        output_path.write_text(html)
