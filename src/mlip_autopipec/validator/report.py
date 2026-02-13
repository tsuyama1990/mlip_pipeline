import logging
from pathlib import Path

from mlip_autopipec.validator.elastic import ElasticResults
from mlip_autopipec.validator.eos import EOSResults
from mlip_autopipec.validator.phonon import PhononResults

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates HTML report for validation results."""

    def generate_report(
        self,
        eos_results: EOSResults,
        elastic_results: ElasticResults,
        phonon_results: PhononResults,
        output_path: Path
    ) -> Path:
        """
        Generate HTML report.

        Args:
            eos_results: EOS fitting results.
            elastic_results: Elastic constants results.
            phonon_results: Phonon stability results.
            output_path: Path to write report to.

        Returns:
            Path to the generated report.
        """
        # Simple HTML template
        html = f"""
        <html>
        <head>
            <title>Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .section {{ margin-bottom: 20px; border: 1px solid #ccc; padding: 10px; border-radius: 5px; }}
                .metric {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Validation Report</h1>
            <div class="section">
                <h2>Equation of State (EOS)</h2>
                <div class="metric">E0: {eos_results.E0:.4f} eV</div>
                <div class="metric">V0: {eos_results.V0:.4f} A^3</div>
                <div class="metric">B0: {eos_results.B0:.2f} GPa</div>
                <div class="metric">B0': {eos_results.B0_prime:.2f}</div>
            </div>
            <div class="section">
                <h2>Elastic Constants</h2>
                <div class="metric">C11: {elastic_results.C11:.2f} GPa</div>
                <div class="metric">C12: {elastic_results.C12:.2f} GPa</div>
                <div class="metric">C44: {elastic_results.C44:.2f} GPa</div>
                <div class="metric">Bulk Modulus (B): {elastic_results.B:.2f} GPa</div>
                <div class="metric">Shear Modulus (G): {elastic_results.G:.2f} GPa</div>
            </div>
            <div class="section">
                <h2>Phonon Stability</h2>
                <div class="metric">Stable: {phonon_results.is_stable}</div>
                <div class="metric">Max Imaginary Freq: {phonon_results.max_imaginary_freq:.4f} THz</div>
            </div>
        </body>
        </html>
        """

        output_path.write_text(html)
        return output_path
