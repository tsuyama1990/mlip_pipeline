# CYCLE08: Monitoring and Usability (SPEC.md)

## 1. Summary

This document provides the detailed technical specification for Cycle 8, the final core development cycle of the MLIP-AutoPipe project. The focus of this cycle is on **monitoring and usability**, addressing the "black box" problem inherent in any long-running, autonomous system. While the workflow is designed to be "Zero-Human," a user still needs to understand the status of their project, assess the performance of the current MLIP, and gain insight into the progress of the active learning strategy. This cycle delivers that insight.

The primary deliverable is a **Monitoring Dashboard**. This will not be a complex, continuously running web server, but rather a static HTML report that can be generated on-demand. This approach is lightweight, secure, and perfectly suited for typical HPC environments. A new `monitoring` module will be created, responsible for gathering data from the project's state files (the `checkpoint.json` and the ASE database), generating a series of informative plots and statistics, and rendering them into a single, self-contained `dashboard.html` file using a template engine.

To expose this functionality to the user, a new command will be added to the Command Line Interface: `mlip-auto status`. When a user runs this command from within their project directory, it will trigger the dashboard generation. This dashboard will provide crucial at-a-glance information, including:
-   The current active learning generation number.
-   The total number of DFT calculations completed and pending.
-   A plot of the training set size over time, broken down by data source (e.g., initial SQS, active learning).
-   A plot of the model's performance (e.g., Force RMSE) over successive training generations.

To support this, the checkpointing system designed in Cycle 6 will be enhanced to store a history of training metrics. By the end of this cycle, the MLIP-AutoPipe application will not only be a powerful automation engine but also a user-friendly scientific tool that provides the clear, quantitative feedback necessary for a productive research workflow.

## 2. System Architecture

The architecture for this final cycle adds a new capability for introspection, cleanly separated in its own module, and extends the CLI with a new user-facing command.

**File Structure for Cycle 8:**

The following ASCII tree highlights the new or modified files in **bold**.

```
mlip-autopipe/
├── dev_documents/
│   └── system_prompts/
│       └── CYCLE08/
│           ├── **SPEC.md**
│           └── **UAT.md**
├── mlip_autopipec/
│   ├── **app.py**              # Modified to add the 'status' command
│   ├── config/
│   │   └── **models.py**       # Updated to store training history
│   ├── **monitoring/**         # New package for dashboard generation
│   │   ├── **__init__.py**
│   │   ├── **dashboard.py**    # Core logic for generating the report
│   │   └── **templates/**
│   │       └── **dashboard.html.j2** # Jinja2 template for the report
│   └── ...
├── tests/
│   ├── **test_monitoring.py**  # Tests for the dashboard generation
│   └── ...
└── pyproject.toml
```

**Component Blueprint: `monitoring/dashboard.py`**

This new module contains the logic for creating the status report.

-   **`generate_dashboard(project_dir: Path) -> Path`**: The main public function. It takes the path to the project directory, orchestrates the data gathering, plot generation, and HTML rendering, and returns the path to the generated `dashboard.html`.
-   **`_gather_data(project_dir: Path) -> DashboardData`**: A helper that reads `checkpoint.json` and the ASE database, aggregating all necessary statistics into a `DashboardData` Pydantic model.
-   **`_create_plots(data: DashboardData) -> Dict[str, str]`**: This function takes the aggregated data and uses a plotting library like Plotly to generate interactive charts. It returns a dictionary where keys are plot names and values are the self-contained HTML/JS snippets for each plot.
-   **`_render_html(data: DashboardData, plots: Dict[str, str]) -> str`**: This function loads the Jinja2 template, passes the data and plot snippets to it, and renders the final HTML content as a string.

**Component Blueprint: `app.py` (Modified)**

-   A new Typer command will be added:
    -   **`@app.command()`**
    -   **`status(project_dir: Path = typer.Argument(".", help="Path to the project directory."), open_browser: bool = typer.Option(True, "--no-open", help="Open the dashboard in a web browser."))`**:
        -   Calls `monitoring.dashboard.generate_dashboard()`.
        -   Prints the path to the generated file.
        -   If `open_browser` is true, it uses the `webbrowser` standard library module to open the HTML file for the user.

**Component Blueprint: `config/models.py` (Modified)**

-   The `CheckpointState` model will be updated to include a new field to track model performance over time:
    -   **`training_history: List[TrainingRunMetrics]`**: A list to store metrics from each training run.
-   A new model, **`TrainingRunMetrics(BaseModel)`**, will be created:
    -   `generation: int`
    -   `num_structures: int`
    -   `rmse_forces: float`
    -   `rmse_energy_per_atom: float`

## 3. Design Architecture

The design is centered on creating a static, self-contained HTML file that can be easily viewed, shared, or archived.

**Pydantic Schema Definitions (New/Modified in `config/models.py`):**

1.  **`TrainingRunMetrics(BaseModel)`**: As described above, this captures the key performance indicators for a single training run.
2.  **`CheckpointState(BaseModel)`**: Will be updated with `training_history: List[TrainingRunMetrics] = []`.
3.  **`DashboardData(BaseModel)`**: A new, read-only model used to structure the data for the HTML template.
    -   `project_name: str`
    -   `current_generation: int`
    -   `completed_calcs: int`
    -   `pending_calcs: int`
    -   `training_history: List[TrainingRunMetrics]`
    -   `dataset_composition: Dict[str, int]`: A dictionary mapping `config_type` to its count in the database (e.g., `{'sqs': 100, 'active_learning': 50}`).

**Data Flow:**
1.  During the main workflow run, after each successful training in the `WorkflowManager`, a `TrainingRunMetrics` object is created and appended to the `training_history` list in the `CheckpointState` before it is saved.
2.  The user runs `$ mlip-auto status .`
3.  The `status` command in `app.py` calls `generate_dashboard()`.
4.  `_gather_data` is called. It reads `checkpoint.json` to get the high-level status and training history. It connects to the ASE DB and performs a query to count the structures grouped by their `config_type` metadata tag. It populates a `DashboardData` object.
5.  `_create_plots` is called. It uses the `training_history` to plot RMSE vs. Generation. It uses `dataset_composition` to create a pie chart. It returns these plots as HTML strings.
6.  `_render_html` is called. It passes the `DashboardData` object and the plot strings to the `dashboard.html.j2` template.
7.  The Jinja2 template (`<p>Project: {{ data.project_name }}</p><div>{{ plots.rmse_plot }}</div>`) renders the final HTML.
8.  The HTML content is written to `dashboard.html` in the project directory.

## 4. Implementation Approach

1.  **Add Dependencies:** Add `plotly`, `jinja2`, and `pandas` (for easier DB queries) to `pyproject.toml`.
2.  **Update Data Models:** Add the `TrainingRunMetrics` model and update the `CheckpointState` model in `config/models.py`.
3.  **Update Workflow:** Modify the `WorkflowManager` and `PacemakerTrainer`. The trainer needs to be able to extract the final RMSE values from the Pacemaker output. The `WorkflowManager` needs to take these values, create a `TrainingRunMetrics` object, and save it as part of the checkpoint.
4.  **Create Monitoring Module:** Create the `mlip_autopipec/monitoring` package and the `dashboard.py` file.
5.  **Implement Data Gathering (`_gather_data`):** Implement the logic to read and parse the `checkpoint.json` and to connect to the ASE DB. Use `pandas.read_sql` or equivalent to efficiently get the counts of different structure types.
6.  **Implement Plotting (`_create_plots`):** Use Plotly to create functions that return `fig.to_html(full_html=False, include_plotlyjs='cdn')`. This creates a portable `div` that can be embedded in any HTML page.
7.  **Create Jinja2 Template:** Create the `dashboard.html.j2` file. It will have a basic HTML structure with placeholders for the data and plots.
8.  **Implement Rendering (`_render_html` and `generate_dashboard`):** Implement the Jinja2 rendering logic and the main function that writes the final file to disk.
9.  **Update CLI (`app.py`):** Add the new `status` command to `app.py`, including the logic to call the dashboard generator and the `webbrowser.open()` function.

## 5. Test Strategy

Testing will focus on the data aggregation and report generation, ensuring the final HTML is produced correctly from mock data.

**Unit Testing Approach (Min 300 words):**

-   **Test Data Aggregation:** The `_gather_data` function will be tested by pointing it to a temporary directory containing a mock `checkpoint.json` file and a mock ASE database (an SQLite file created programmatically during the test). The test will call the function and assert that the returned `DashboardData` object contains the correct, aggregated values (e.g., the `completed_calcs` count matches the number of entries in the mock database).
-   **Test Plotting Functions:** Each plotting function (e.g., `_create_rmse_plot`) will be tested in isolation. It will be passed a pre-defined `DashboardData` object. The test will not check the visual appearance of the plot. Instead, it will assert that the function returns a non-empty string that contains expected substrings, like `<div class="plotly-graph-div"` and `"title": {"text": "Force RMSE vs. Generation"}`. This confirms the plot is being generated without errors.
-   **Test HTML Rendering:** The `generate_dashboard` function will be tested by running it on a mock data source. The test will not parse the entire HTML output. It will read the generated HTML as a string and assert that it contains specific key pieces of data that were in the mock input, for example, `<h1>Project: Mock Project</h1>` and the HTML snippet from the mocked plotting function.

**Integration Testing Approach (Min 300 words):**

The integration test will verify the `status` CLI command from end to end.

-   **Test `status` Command:**
    1.  **Setup:** The test will create a temporary project directory. It will programmatically create a realistic `checkpoint.json` file (with some training history) and a small ASE database, simulating the state of a workflow that has run for a few generations.
    2.  **Execution:** It will use `typer.testing.CliRunner` to invoke the `status` command on this directory: `runner.invoke(app, ["status", "."])`.
    3.  **Assertion:**
        -   The test will assert that the command's exit code is 0.
        -   It will assert that the console output includes the line "Dashboard generated at: ./dashboard.html".
        -   It will assert that the file `./dashboard.html` now exists in the temporary directory.
        -   It will read the contents of the generated HTML file and assert that it contains the project name from the mock checkpoint file, confirming that real data was read and rendered into the final report. This validates the entire chain from the user command to the final file output.
