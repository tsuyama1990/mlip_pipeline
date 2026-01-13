# CYCLE04 Specification: The User Experience

## 1. Summary

This document provides the detailed technical specification for CYCLE04 of the MLIP-AutoPipe project. With the core autonomous learning engine now fully implemented, the focus of this cycle shifts from backend intelligence to the user-facing experience. The primary objective is to make the powerful, complex system developed in previous cycles accessible, understandable, and easy to use for the target audience of materials scientists. This will be achieved through three main deliverables: a polished and intuitive **Command-Line Interface (CLI)**, a **Heuristic Engine** for configuration simplification, and a **Web-Based Monitoring Dashboard**.

First, the existing ad-hoc CLI (`main.py`) will be re-engineered into a professional, user-friendly tool using a library like Typer. The CLI will be the main entry point for users to interact with the system. It will have a clear, hierarchical command structure (e.g., `mlip-auto run`, `mlip-auto train`, `mlip-auto report`), comprehensive help messages, and robust error handling. This will transform the system from a collection of scripts into a cohesive and professional piece of software.

Second, and critically, this cycle will implement the **Heuristic Engine**. This engine is the key to the "zero-human" philosophy from the user's perspective. It will be responsible for taking a very high-level, minimal user configuration (the `UserConfig` schema) and expanding it into the massive, detailed `SystemConfig` that the backend modules require. For example, a user will simply specify `elements: [Fe, Ni]` and `goal: 'diffusion'`, and the Heuristic Engine will automatically determine the appropriate DFT parameters, generation strategies, simulation temperatures, and OTF settings. This engine will encapsulate a significant amount of expert knowledge, making the system accessible to non-experts.

Third, to provide a window into the complex, long-running autonomous process, a simple **Web-Based Monitoring Dashboard** will be developed. This will be a read-only web application that visualizes the state and progress of a run. Users will be able to see the training curve (RMSE vs. time), the number of structures in the DFT queue, the current uncertainty level of the OTF simulation, and key physical properties being calculated. This dashboard will be crucial for building user trust and providing insight into the system's decision-making process.

By the end of CYCLE04, the MLIP-AutoPipe system will not only be intelligent but also elegant and accessible, significantly lowering the barrier to entry for advanced materials simulations.

## 2. System Architecture

The architecture of CYCLE04 primarily involves adding a new "UI" layer on top of the existing backend, consisting of the CLI, the Heuristic Engine, and the Web Dashboard. The backend modules remain largely unchanged, but their execution is now orchestrated through the new, more sophisticated user-facing components.

**File Structure for CYCLE04:**

The file structure expands to include dedicated components for the CLI and the new web dashboard.

```
mlip_autopipec/
├── src/
│   ├── mlip_autopipec/
│   │   ├── __init__.py
│   │   ├── **cli.py**              # New, dedicated CLI application using Typer
│   │   ├── **heuristic_engine.py** # The logic for expanding user config
│   │   ├── main.py               # May be kept as a simple entry point to cli.py
│   │   ├── settings.py
│   │   ├── schemas/
│   │   │   └── ...
│   │   ├── modules/
│   │   │   └── ...
│   │   ├── utils/
│   │   │   └── ...
│   │   └── **dashboard/**
│   │       ├── __init__.py
│   │       ├── **app.py**            # FastAPI/Flask application
│   │       ├── **static/**           # CSS, JS files
│   │       └── **templates/**        # HTML templates (e.g., index.html)
├── tests/
│   └── ...
│   ├── **test_cli.py**
│   └── **test_heuristic_engine.py**
├── pyproject.toml
└── README.md
```

**Architectural Blueprint:**

1.  **User Interaction**: The user's primary interaction is now with the `mlip-auto` command, defined in `cli.py`.
2.  **Configuration Flow**:
    *   A user creates a minimal `input.yaml` file (e.g., specifying elements and a goal).
    *   They run a command like `mlip-auto run --input input.yaml`.
    *   The CLI (`cli.py`) parses this input file and validates it against the `UserConfig` Pydantic schema.
    *   The validated `UserConfig` object is then passed to the **Heuristic Engine (`heuristic_engine.py`)**.
    *   The Heuristic Engine applies a set of rules and expert knowledge to translate the high-level request into the detailed, low-level `SystemConfig`. For instance, if the goal is `'diffusion'`, it might set the MD simulation temperature to a range around 60% of the material's melting point and select a generation strategy that includes atomic vacancies.
    *   This fully populated `SystemConfig` object is then used to configure and launch the backend OTF runner (Module E) and its associated workers, as developed in CYCLE03.
3.  **Monitoring and Data Flow**:
    *   During the run, the backend modules (especially the trainer and inference engine) will be modified to log key metrics to a structured, machine-readable log file (e.g., a JSONL file or a simple SQLite database). This log will contain time-series data like RMSE, number of queued items, `extrapolation_grade`, and calculated physical properties.
    *   The **Web Dashboard**, a separate process started by the `app.py` script (using a lightweight framework like FastAPI or Flask), runs a simple web server.
    *   When a user accesses the dashboard URL in their browser, the web application reads the structured log file.
    *   The backend of the web app provides API endpoints (e.g., `/api/metrics`) that the frontend can call to get the latest data.
    *   The frontend (`index.html` and its associated JavaScript) uses a library like Chart.js to periodically fetch this data from the API and update the plots, creating a live view of the system's performance.

This architecture decouples the user-facing components from the core logic. The backend doesn't need to know about the dashboard; it only needs to write to a log file. This makes the system more robust and easier to develop and maintain.

## 3. Design Architecture

The design of CYCLE04 focuses on creating clean interfaces for the user (CLI), for the configuration logic (Heuristic Engine), and for the monitoring data (Dashboard).

**CLI Design (`cli.py`):**

*   The CLI will be implemented using `Typer`.
*   A main `app` object will be created.
*   Commands will be implemented as functions decorated with `@app.command()`.
*   The command structure will be:
    *   `mlip-auto run`: Starts a new, full OTF run. Takes an input YAML file as an argument.
    *   `mlip-auto train`: A standalone command to run only the training part on an existing database (retains functionality from CYCLE02).
    *   `mlip-auto dashboard`: Starts the web-based monitoring dashboard. Takes a run directory or log file as an argument.
*   Typer will be used for automatic help generation (`--help`), argument type validation, and providing informative error messages.

**Heuristic Engine Design (`heuristic_engine.py`):**

*   The engine will be a class, `HeuristicEngine`.
*   The main method will be `generate_system_config(user_config: UserConfig) -> SystemConfig`.
*   The internal logic will be a series of "rules" or "policies" implemented as private methods. For example:
    *   `_get_dft_params(elements)`: This method will contain the logic for choosing pseudopotentials, cutoffs, etc., based on the elements. This formalizes the logic that was prototyped in CYCLE01.
    *   `_get_simulation_params(goal, structure)`: This is the most complex part. It will be a large dispatcher that, based on the `simulation_goal` (e.g., 'elastic', 'phase_diagram', 'diffusion'), sets the appropriate parameters for the MD simulation (temperature, pressure, ensemble) and the structure generation strategy.
    *   The engine will contain a database of known material properties (e.g., melting points, crystal structures) to make informed decisions.
*   The design emphasizes modularity, allowing new rules and simulation goals to be added easily in the future.

**Dashboard Design (`dashboard/`):**

*   **Backend (`app.py`):**
    *   A `FastAPI` application will be used.
    *   A single API endpoint, `/api/metrics`, will be created.
    *   When this endpoint is called, it will read the last N lines of the metrics log file (e.g., `metrics.jsonl`), parse the JSON, and return it. This is a simple, stateless approach that is easy to implement and robust.
    *   A main endpoint `/` will serve the `index.html` file.
*   **Frontend (`templates/index.html`, `static/`):**
    *   A single-page application using vanilla JavaScript or a very lightweight framework.
    *   It will use the browser's `fetch` API to call the `/api/metrics` endpoint every few seconds (e.g., 5-10 seconds).
    *   It will use `Chart.js` to render the data into several plots:
        *   A line chart for RMSE vs. training iteration.
        *   A line chart for the number of structures in the DFT queue over time.
        *   A gauge or line chart for the current `extrapolation_grade`.
    *   The design will be clean and simple, focusing on presenting the most important information clearly.

**Metrics Logging:**

*   The backend modules (`d_trainer.py`, `e_inference.py`) will be updated to include logging calls.
*   A dedicated logger will be configured to write structured JSON logs to a file (e.g., `metrics.jsonl`). Each log entry will be a JSON object with a timestamp, a metric name, and a value (e.g., `{"timestamp": "...", "metric": "rmse", "value": 0.05}`).

This design provides a clear and robust implementation path for each of the three major user-facing components of this cycle.

## 4. Implementation Approach

The implementation will be done in parallel where possible (e.g., CLI and Dashboard can be developed independently), but the Heuristic Engine is the central, most critical piece.

**Step 1: Develop the Heuristic Engine**
*   Create `heuristic_engine.py`.
*   Begin by writing the main class structure and the `generate_system_config` method signature.
*   Implement the rule methods one by one. Start with the simplest, `_get_dft_params`, which can be adapted from the existing logic.
*   Implement the `_get_simulation_params` method. This will be the most substantial part. Start with one or two simulation goals (e.g., 'equilibrate' and 'diffusion') and hard-code the logic for them. A simple internal dictionary can be used to store material property data.
*   Write extensive unit tests for the engine in `tests/test_heuristic_engine.py`. For a given `UserConfig` input, assert that the resulting `SystemConfig` has the expected values for key parameters.

**Step 2: Re-engineer the CLI**
*   Create `cli.py` and set up the main Typer application.
*   Implement the `run` command. This command's logic will be:
    1.  Load the `input.yaml` into a `UserConfig` object.
    2.  Instantiate the `HeuristicEngine`.
    3.  Call `engine.generate_system_config(user_config)`.
    4.  Use the resulting `SystemConfig` to instantiate and start the `OTFRunner` from CYCLE03.
*   Implement the `train` and `dashboard` commands. The `dashboard` command will simply use `subprocess.Popen` to launch the FastAPI/Uvicorn server.
*   Write integration tests for the CLI in `test_cli.py` using Typer's built-in test runner.

**Step 3: Implement the Metrics Logging**
*   Choose a standard logging library that supports structured JSON output (Python's built-in `logging` can be configured for this, or a library like `structlog`).
*   Go into `d_trainer.py` and `e_inference.py`. After key events (e.g., a training run finishes, an uncertainty check is performed), add a logging call to record the relevant metric to the `metrics.jsonl` file.

**Step 4: Develop the Web Dashboard**
*   Create the `dashboard/` directory structure.
*   In `app.py`, set up the FastAPI server. Implement the `/` endpoint to serve the HTML file and the `/api/metrics` endpoint to read and return the log file data.
*   In `index.html`, create the basic layout with placeholders for the charts.
*   Write the JavaScript code in a `<script>` tag or a separate `.js` file.
    *   Use `setInterval` to create a loop that calls `fetch('/api/metrics')` every 5 seconds.
    *   Write the logic to parse the returned JSON data and update the `Chart.js` chart objects. This will cause the charts to update live in the user's browser.

This approach ensures that the most critical dependency, the Heuristic Engine, is built first, and then the other components are built around it.

## 5. Test Strategy

The test strategy for CYCLE04 shifts focus to the user interface and the "expert knowledge" encapsulated in the Heuristic Engine. We need to ensure the CLI is intuitive and robust, and that the engine makes sensible decisions.

**Unit Testing Approach (Min 300 words):**

Unit tests will be heavily focused on the Heuristic Engine, as this is where the most complex new logic resides.

*   **Heuristic Engine (`test_heuristic_engine.py`):** This will be the most important test suite of the cycle. We will create a series of test cases, each with a different `UserConfig` input, and assert that the generated `SystemConfig` is correct and physically sensible.
    *   **Test Case 1: Simple Metal (Aluminum)**. Input: `elements: [Al]`, `goal: 'equilibrate'`. Assertions: The DFT parameters should be correct for Al (correct SSSP pseudo, etc.). The simulation temperature should be set to a reasonable value like 300K. The generation strategy should be a simple rattle of the perfect crystal.
    *   **Test Case 2: Alloy Diffusion (FeNi)**. Input: `elements: [Fe, Ni]`, `goal: 'diffusion'`. Assertions: The system should enable spin-polarized DFT (`nspin=2`) because it knows Fe and Ni are magnetic. The simulation temperature should be high, e.g., 1200K, suitable for observing diffusion. The structure generation should be configured to include vacancies, as these are essential for diffusion mechanisms.
    *   **Test Case 3: Invalid Input**. Input: `elements: [Fe]`, `goal: 'invalid_goal'`. Assertions: The engine should raise a `ValueError` or a custom exception, indicating that it does not know how to handle this goal.
    *   These tests are critical as they validate the "expert knowledge" we are encoding into the system. Each test acts as a regression check to ensure that future changes don't break the logic for existing, well-defined use cases.

*   **CLI (`test_cli.py`):** We will use Typer's `CliRunner` to test the command-line interface. We will simulate command-line invocations and check the output.
    *   We will test that `mlip-auto run --help` produces the expected help message and exits cleanly.
    *   We will test the `run` command with a valid and an invalid YAML file path, asserting that it succeeds in one case and prints an informative error message in the other.
    *   We will mock the backend (`OTFRunner`) so that we can test the CLI's logic (parsing, calling the engine) without actually starting a long-running process.

**Integration Testing Approach (Min 300 words):**

The integration test will verify that all the new user-facing components work together seamlessly with the existing backend pipeline.

*   **Test Scenario: A Complete User Journey for an Equilibrate Run**
    *   **Objective**: To simulate a complete user workflow from creating a minimal input file to monitoring the run on the dashboard.
    *   **Setup:**
        1.  A simple `input.yaml` file will be created for a well-known system like Aluminum (`Al`). The goal will be a simple `'equilibrate'` run at 300K.
        2.  A temporary, empty directory for the run will be created.
    *   **Execution:**
        1.  The test script will first invoke `mlip-auto dashboard --run-dir /path/to/tmp/dir &` to start the dashboard in the background.
        2.  It will then invoke `mlip-auto run --input input.yaml --run-dir /path/to/tmp/dir`. This will start the main OTF process. For this test, the backend will be configured to run for only a very short time (e.g., a few hundred MD steps) and to use a mock DFT calculator to make it fast. The backend will be configured to write metrics to the `metrics.jsonl` file inside the run directory.
    *   **Validation:**
        1.  **Heuristic Engine Check**: The test will first check the generated `system_config.yaml` in the run directory to ensure the Heuristic Engine correctly expanded the minimal input. It will assert that the simulation temperature is 300K and the DFT parameters are correct for Al.
        2.  **Backend Run**: The test will assert that the `run` command completes successfully (exit code 0).
        3.  **Dashboard Check**: While the run is proceeding, the test script will use an HTTP client (like `requests`) to connect to the running dashboard's `/api/metrics` endpoint. It will fetch the data from the API and assert that it is valid JSON and contains the expected metrics (e.g., `rmse`, `extrapolation_grade`). This confirms that the metrics logging and the dashboard's API are working correctly.
        4.  Finally, the test will kill the dashboard process.

This E2E test validates the entire user experience layer. It proves that the CLI can correctly launch a run based on a minimal user input, that the Heuristic Engine makes the correct decisions, that the backend correctly logs its progress, and that the dashboard can correctly read and serve this progress data.
