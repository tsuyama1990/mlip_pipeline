import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
from ase.db import connect as ase_db_connect
from jinja2 import Environment, FileSystemLoader, TemplateError

from mlip_autopipec.config.models import DashboardData
from mlip_autopipec.config.schemas.monitoring import DatasetComposition
from mlip_autopipec.domain_models.state import WorkflowState

logger = logging.getLogger(__name__)


def _gather_data(project_dir: Path) -> DashboardData:
    """Gathers data from checkpoint and ASE database."""
    checkpoint_path = project_dir / "workflow_state.json"
    if not checkpoint_path.exists():
        msg = f"Checkpoint file not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    with checkpoint_path.open() as f:
        state_data = json.load(f)
        state = WorkflowState.model_validate(state_data)

    # Count structures in DB
    # TODO: WorkflowState does not store config. Need to load config separately.
    # db_path = state.system_config.training_config.data_source_db
    db_path = project_dir / "mlip.db"  # Default assumption

    dataset_composition: Counter[str] = Counter()
    completed_calcs = 0

    if db_path.exists():
        with ase_db_connect(db_path) as db:
            completed_calcs = len(db)
            for row in db.select():
                config_type = row.data.get("config_type", "unknown")
                dataset_composition[config_type] += 1
    else:
        logger.warning(f"Database not found at {db_path}. Assuming 0 completed calculations.")

    return DashboardData(
        project_name="MLIP Project",  # Placeholder
        current_generation=state.cycle_index,
        completed_calcs=completed_calcs,
        pending_calcs=len(state.active_tasks),
        training_history=[],  # Placeholder: state doesn't have history yet
        dataset_composition=DatasetComposition(dataset_composition),
    )


def _create_plots(data: DashboardData) -> dict[str, str]:
    """Generates Plotly plots as HTML strings."""
    plots = {}

    # Composition Pie Chart
    # Access via .root because it's a RootModel
    if data.dataset_composition.root:
        df_comp = pd.DataFrame(
            list(data.dataset_composition.root.items()), columns=["Type", "Count"]
        )
        fig_comp = px.pie(df_comp, values="Count", names="Type", title="Dataset Composition")
        plots["composition_plot"] = fig_comp.to_html(full_html=False, include_plotlyjs="cdn")
    else:
        plots["composition_plot"] = "<p>No data available</p>"

    # RMSE Plot
    if data.training_history:
        history_data = [
            {
                "Generation": m.generation,
                "Force RMSE": m.rmse_force,
                "Energy RMSE": m.rmse_energy,
            }
            for m in data.training_history
        ]
        df_hist = pd.DataFrame(history_data)

        fig_rmse = px.line(
            df_hist, x="Generation", y="Force RMSE", title="Force RMSE vs. Generation", markers=True
        )
        plots["rmse_plot"] = fig_rmse.to_html(full_html=False, include_plotlyjs=False)
    else:
        plots["rmse_plot"] = "<p>No training history available</p>"

    return plots


def _render_html(data: DashboardData, plots: dict[str, str]) -> str:
    """Renders the Jinja2 template."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    try:
        template = env.get_template("dashboard.html.j2")
        return template.render(data=data, plots=plots)
    except TemplateError as e:
        logger.exception("Failed to render dashboard template.")
        msg = "Dashboard generation failed due to template error."
        raise RuntimeError(msg) from e


def generate_dashboard(project_dir: Path) -> Path:
    """Main function to generate the dashboard."""
    logger.info("Gathering data from project directory...")
    data = _gather_data(project_dir)

    logger.info("Generating plots and rendering HTML...")
    plots = _create_plots(data)
    html_content = _render_html(data, plots)

    output_path = project_dir / "dashboard.html"
    output_path.write_text(html_content)
    logger.info(f"Dashboard generated at: {output_path}")

    return output_path
