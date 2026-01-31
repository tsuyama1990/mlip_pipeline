from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from mlip_autopipec.domain_models.validation import ValidationResult

class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup Jinja2
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def generate_report(self, results: list[ValidationResult]) -> Path:
        template = self.env.get_template("report.html")

        # Prepare context
        # We might need to adjust image paths to be relative to output_dir
        # But ValidationResult paths are likely absolute or relative to run dir.
        # If output_dir is project root, and images are in validation_work/..., it should be fine.

        # We create a view model to ensure paths are strings relative to report location if needed.
        # For simplicity, we assume browsers can resolve paths if relative to CWD or absolute.
        # Absolute paths (file://) might be blocked by browsers for security if not local.
        # Ideally we make them relative to self.output_dir.

        processed_results = []
        for res in results:
            new_plots = {}
            for k, v in res.plots.items():
                # If v is absolute, try to make it relative to output_dir
                try:
                    rel_path = v.absolute().relative_to(self.output_dir.absolute())
                    new_plots[k] = str(rel_path)
                except ValueError:
                    # Not relative, keep absolute
                    new_plots[k] = str(v.absolute())

            # Create a copy with updated plots (Pydantic model copy)
            new_res = res.model_copy(update={"plots": new_plots})
            processed_results.append(new_res)

        html_content = template.render(
            results=processed_results,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        report_path = self.output_dir / "validation_report.html"
        report_path.write_text(html_content)

        return report_path
