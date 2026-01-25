# Cycle 08 Specification: Full Loop Integration & Reporting

## 1. Summary

Cycle 08 is the final integration phase. We bring together all the components developed in Cycles 01-07 into a seamless, user-friendly application. The primary deliverables are the polished Command Line Interface (CLI) and the "Dashboard" module.

The Dashboard is crucial for user trust. It generates a comprehensive HTML report that visualizes the entire active learning history: the convergence of RMSE, the number of structures added per cycle, the results of physical validation tests (phonon bands, EOS curves), and the evolution of uncertainty.

This cycle also involves the final End-to-End (E2E) testing of the "Zero-Config" workflow, ensuring that a user can go from `config.yaml` to a production-ready `potential.yace` without touching a single line of Python code.

## 2. System Architecture

**Files to be created/modified in this cycle are marked in Bold.**

```ascii
src/mlip_autopipec/
├── **app.py**                  # (Finalized CLI)
├── orchestration/
│   └── **dashboard.py**        # HTML Report Generator
├── templates/
│   └── **report.html**         # Jinja2 template for the dashboard
└── **__main__.py**             # Execution entry point
```

## 3. Design Architecture

### `Dashboard` (in `orchestration/dashboard.py`)
*   **Responsibility**: Visualize the system state.
*   **Inputs**: `DatabaseManager` (for training history), `ValidationRunner` (for physics checks).
*   **Outputs**: `report.html` (Interactive HTML with Plotly/Matplotlib).
*   **Content**:
    *   **Convergence Plot**: Energy/Force RMSE vs. Generation.
    *   **Data Efficiency**: Number of DFT calculations vs. Time.
    *   **Validation Section**: Tabs showing Phonon dispersion, EOS curves, and Elastic tables.
    *   **Logs**: Tail of the system log.

### `CLI` (in `app.py` and `__main__.py`)
*   **Commands**:
    *   `mlip-auto run config.yaml`: Start the main loop.
    *   `mlip-auto status`: Show current cycle and state.
    *   `mlip-auto report`: Force regeneration of the HTML dashboard.
    *   `mlip-auto validate potential.yace`: Run the validation suite manually.

## 4. Implementation Approach

1.  **Dashboard**: Create a Jinja2 template. Use `plotly` for interactive graphs (RMSE, EOS).
2.  **CLI Wiring**: Use `typer` or `argparse` to expose the commands. Ensure graceful Ctrl+C handling.
3.  **Final Polish**: Update `README`, docstrings, and remove any debug prints.

## 5. Test Strategy

### Unit Testing
*   **Dashboard**: Pass a mock history dictionary to `generate_report()`. Verify HTML output contains the expected sections and data points.

### System Testing (The "Grand Final")
*   **Zero-Config Run**:
    *   Prepare a `config.yaml` for a simple system (e.g., Aluminum).
    *   Run `mlip-auto run`.
    *   Wait for 3-5 cycles.
    *   **Verify**:
        *   `experiments/` directory is populated.
        *   `potential.yace` is created.
        *   `report.html` is generated and viewable.
        *   Validation tests passed.
