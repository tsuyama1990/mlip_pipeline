import base64
import datetime
import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.orchestration.models import DashboardData

logger = logging.getLogger(__name__)

class Dashboard:
    """
    Generates a simple HTML dashboard for monitoring the workflow.

    This class is responsible for visualizing the progress of the active learning loop,
    including RMSE metrics and database statistics.
    """
    def __init__(self, output_dir: Path, db_manager: DatabaseManager) -> None:
        """
        Initialize the Dashboard.

        Args:
            output_dir: Directory to save the dashboard (status.html).
            db_manager: Database manager to query for statistics.
        """
        self.output_dir = output_dir
        self.db_manager = db_manager
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_dir / "dashboard.html"

    def update(self, data: DashboardData) -> None:
        """
        Update the dashboard with new data.

        This method generates plots using Matplotlib and embeds them into an HTML report.

        Args:
            data: The DashboardData object containing current stats.
        """
        logger.info("Updating dashboard...")

        # Generate plot
        plot_img = self._generate_plot(data)

        # Generate HTML
        html_content = self._generate_html(data, plot_img)

        # Write to file
        self.report_path.write_text(html_content)
        logger.info(f"Dashboard generated at {self.report_path}")

    def _generate_plot(self, data: DashboardData) -> str:
        """
        Generate a learning curve plot and return as base64 string.

        Args:
            data: Dashboard data containing metrics history.

        Returns:
            Base64 encoded PNG image string.
        """
        if not data.generations:
             return ""

        plt.figure(figsize=(10, 6))

        # RMSE Plot
        plt.subplot(2, 1, 1)
        plt.plot(data.generations, data.rmse_values, 'b-o', label='RMSE')
        plt.ylabel('RMSE (eV/A)')
        plt.title('Learning Curve')
        plt.grid(True)
        plt.legend()

        # Structure Count Plot
        plt.subplot(2, 1, 2)
        plt.plot(data.generations, data.structure_counts, 'g-s', label='Structures')
        plt.xlabel('Generation')
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"

    def _generate_html(self, data: DashboardData, plot_img: str) -> str:
        """
        Generate the HTML content.

        Args:
            data: Dashboard data for status display.
            plot_img: Base64 string of the generated plot.

        Returns:
            String containing the full HTML report.
        """
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLIP-AutoPipe Dashboard</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                .container {{ max_width: 800px; margin: 0 auto; }}
                .status-box {{ padding: 20px; border: 1px solid #ccc; margin-bottom: 20px; border-radius: 5px; }}
                .status-running {{ background-color: #e6f7ff; }}
                .status-idle {{ background-color: #f0f0f0; }}
                .status-error {{ background-color: #ffe6e6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MLIP-AutoPipe Status</h1>
                <p>Last Updated: {timestamp} UTC</p>

                <div class="status-box status-{data.status.lower()}">
                    <h2>Current Status: {data.status}</h2>
                    <ul>
                        <li>Current Generation: {data.generations[-1] if data.generations else 0}</li>
                        <li>Total Structures: {data.structure_counts[-1] if data.structure_counts else 0}</li>
                        <li>Latest RMSE: {data.rmse_values[-1] if data.rmse_values else 'N/A'}</li>
                    </ul>
                </div>

                <h2>Learning Progress</h2>
                {f'<img src="{plot_img}" alt="Learning Curve" style="width:100%">' if plot_img else '<p>No data yet.</p>'}
            </div>
        </body>
        </html>
        """
