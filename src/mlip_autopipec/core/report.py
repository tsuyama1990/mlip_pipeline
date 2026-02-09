import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from mlip_autopipec.domain_models.config import GlobalConfig

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates HTML report for the active learning pipeline.
    """

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.workdir = config.workdir

    def collect_metrics(self) -> pd.DataFrame:
        """
        Collect metrics from all cycles.
        """
        data = []
        cycle = 1
        while True:
            cycle_dir_name = self.config.orchestrator.cycle_dir_pattern.format(cycle=cycle)
            cycle_dir = self.workdir / cycle_dir_name
            metrics_file = cycle_dir / "metrics.json"

            if not metrics_file.exists():
                break

            try:
                with metrics_file.open("r") as f:
                    metrics = json.load(f)
                data.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to load metrics from {metrics_file}: {e}")

            cycle += 1

        return pd.DataFrame(data)

    def generate_report(self) -> str:
        """
        Generate HTML report content.
        """
        df = self.collect_metrics()

        if df.empty:
            return "<html><body><h1>No metrics found</h1></body></html>"

        # Generate Plots
        plots_html = self._generate_plots(df)

        # Summary Table
        table_html = df.to_html(classes="table table-striped", index=False)

        # Status
        status = "Unknown"
        state_file = self.workdir / self.config.orchestrator.state_filename
        if state_file.exists():
            try:
                with state_file.open("r") as f:
                    state = json.load(f)
                status = state.get("status", "Unknown")
            except Exception:
                pass

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MLIP Pipeline Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 8px; }}
        .table tr:nth-child(even){{ background-color: #f2f2f2; }}
        .table th {{ padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #04AA6D; color: white; }}
        .plot {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>MLIP Pipeline Report</h1>
    <p><strong>Status:</strong> {status}</p>
    <p><strong>Workdir:</strong> {self.workdir}</p>

    <h2>Learning Curves</h2>
    <div class="plot">
        {plots_html}
    </div>

    <h2>Cycle Metrics</h2>
    {table_html}
</body>
</html>
"""
        return html

    def _generate_plots(self, df: pd.DataFrame) -> str:
        """Generate plots using matplotlib and return as base64 img tags."""
        img_tags = ""

        # 1. RMSE vs Cycle
        if "energy_rmse" in df.columns or "force_rmse" in df.columns:
            plt.figure(figsize=(10, 6))
            if "energy_rmse" in df.columns:
                plt.plot(df["cycle"], df["energy_rmse"], 'o-', label="Energy RMSE (eV/atom)")
            if "force_rmse" in df.columns:
                plt.plot(df["cycle"], df["force_rmse"], 's-', label="Force RMSE (eV/A)")
            plt.xlabel("Cycle")
            plt.ylabel("RMSE")
            plt.title("Training Error vs Cycle")
            plt.legend()
            plt.grid(True)
            img_tags += self._fig_to_base64(plt.gcf())
            plt.close()

        # 2. Dataset Size vs Cycle
        if "dataset_size" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["cycle"], df["dataset_size"], 'o-', color='orange')
            plt.xlabel("Cycle")
            plt.ylabel("Dataset Size")
            plt.title("Dataset Growth")
            plt.grid(True)
            img_tags += self._fig_to_base64(plt.gcf())
            plt.close()

        return img_tags

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return f'<img src="data:image/png;base64,{img_str}" />'

    def save_report(self, filename: str = "report.html") -> Path:
        content = self.generate_report()
        path = self.workdir / filename
        path.write_text(content)
        logger.info(f"Report saved to {path}")
        return path
