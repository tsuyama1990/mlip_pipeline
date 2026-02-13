import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from jinja2 import Template

from mlip_autopipec.validator.elastic import ElasticResults
from mlip_autopipec.validator.eos import EOSResults
from mlip_autopipec.validator.phonon import PhononResults

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Potential Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .section { margin-bottom: 30px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }
        table { border-collapse: collapse; width: 100%; max-width: 600px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .success { color: green; font-weight: bold; }
        .failure { color: red; font-weight: bold; }
        .plot { margin-top: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>Potential Validation Report</h1>

    <div class="section">
        <h2>Summary</h2>
        <p><strong>Overall Status:</strong>
            <span class="{{ 'success' if passed else 'failure' }}">
                {{ "PASSED" if passed else "FAILED" }}
            </span>
        </p>
    </div>

    <div class="section">
        <h2>Elastic Constants</h2>
        {% if elastic %}
        <table>
            <tr><th>Property</th><th>Value (GPa)</th></tr>
            <tr><td>C11</td><td>{{ "%.2f"|format(elastic.C11) }}</td></tr>
            <tr><td>C12</td><td>{{ "%.2f"|format(elastic.C12) }}</td></tr>
            <tr><td>C44</td><td>{{ "%.2f"|format(elastic.C44) }}</td></tr>
            <tr><td>Bulk Modulus (B)</td><td>{{ "%.2f"|format(elastic.bulk_modulus) }}</td></tr>
            <tr><td>Shear Modulus (G)</td><td>{{ "%.2f"|format(elastic.shear_modulus) }}</td></tr>
        </table>
        {% else %}
        <p>No elastic analysis performed.</p>
        {% endif %}
    </div>

    <div class="section">
        <h2>Equation of State (EOS)</h2>
        {% if eos %}
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Equilibrium Volume (V0)</td><td>{{ "%.3f"|format(eos.volume) }} A^3</td></tr>
            <tr><td>Equilibrium Energy (E0)</td><td>{{ "%.3f"|format(eos.energy) }} eV</td></tr>
            <tr><td>Bulk Modulus (B0)</td><td>{{ "%.2f"|format(eos.bulk_modulus) }} GPa</td></tr>
            <tr><td>Derivative (B0')</td><td>{{ "%.2f"|format(eos.bulk_modulus_derivative) }}</td></tr>
        </table>
        {% else %}
        <p>No EOS analysis performed.</p>
        {% endif %}
    </div>

    <div class="section">
        <h2>Phonon Stability</h2>
        {% if phonon %}
        <p><strong>Stability:</strong>
            <span class="{{ 'success' if phonon.is_stable else 'failure' }}">
                {{ "Stable" if phonon.is_stable else "Unstable" }}
            </span>
        </p>
        <p>Max Imaginary Frequency: {{ "%.4f"|format(phonon.max_imaginary_freq) }} THz</p>

        {% if phonon_plot %}
        <h3>Phonon Dispersion</h3>
        <img src="{{ phonon_plot }}" alt="Phonon Dispersion" class="plot" width="600">
        {% endif %}
        {% else %}
        <p>No phonon analysis performed.</p>
        {% endif %}
    </div>

</body>
</html>
"""

class ReportGenerator:
    """Generates HTML validation report."""

    def generate(
        self,
        output_dir: Path,
        passed: bool,
        elastic_results: ElasticResults | None = None,
        eos_results: EOSResults | None = None,
        phonon_results: PhononResults | None = None
    ) -> Path:
        """
        Generates the report.

        Args:
            output_dir: Directory to save report and plots.
            passed: Whether validation passed.
            elastic_results: Results from ElasticAnalyzer.
            eos_results: Results from EOSAnalyzer.
            phonon_results: Results from PhononAnalyzer.

        Returns:
            Path to the generated HTML file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        phonon_plot_rel_path = None
        if phonon_results and phonon_results.band_structure_plot_data:
            phonon_plot_path = output_dir / "phonon_dispersion.png"
            self._plot_phonon_bands(phonon_results.band_structure_plot_data, phonon_plot_path)
            phonon_plot_rel_path = phonon_plot_path.name

        template = Template(REPORT_TEMPLATE)
        html_content = template.render(
            passed=passed,
            elastic=elastic_results,
            eos=eos_results,
            phonon=phonon_results,
            phonon_plot=phonon_plot_rel_path
        )

        report_path = output_dir / "validation_report.html"
        report_path.write_text(html_content)
        logger.info(f"Validation report generated at {report_path}")

        return report_path

    def _plot_phonon_bands(self, band_data: dict[str, Any], output_path: Path) -> None:
        """Plots phonon band structure."""
        try:
            distances = band_data["distances"] # list of arrays (path segments)
            frequencies = band_data["frequencies"] # list of arrays (bands)

            plt.figure(figsize=(8, 6))

            for d_seg, f_seg in zip(distances, frequencies, strict=False):
                plt.plot(d_seg, f_seg, color='red', linewidth=1)


            plt.ylabel("Frequency (THz)")
            plt.xlabel("Wave Vector")
            plt.title("Phonon Dispersion")
            plt.grid(True, linestyle='--', alpha=0.5)

            plt.savefig(output_path, dpi=100)
            plt.close()
        except Exception:
            logger.warning("Failed to plot phonon bands.", exc_info=True)
