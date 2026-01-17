# CYCLE07: User Interface (CLI) (SPEC.md)

## 1. Summary

This document provides the detailed technical specification for Cycle 7 of the MLIP-AutoPipe project. With the core logic and backend components now designed, this cycle focuses on **usability and the primary user interface**. The goal is to create a clean, professional, and intuitive Command Line Interface (CLI) that serves as the single entry point for the entire application. This CLI is the "front door" to the "Zero-Human" protocol; it must be simple to use, provide helpful feedback, and handle user errors gracefully.

The key deliverable for this cycle is a new `app.py` module that will house a CLI application built using the Typer library. Typer is chosen for its ability to create modern, user-friendly CLIs with minimal boilerplate code, automatically generating help menus and performing input validation. The main command to be implemented will be `mlip-auto run`, which will accept a single argument: the path to the user's `input.yaml` configuration file.

The responsibilities of the CLI module will be strictly limited to interface concerns. It will be a thin wrapper that orchestrates the major components already designed. Its workflow will be:
1.  Parse the command-line arguments.
2.  Load and validate the specified `input.yaml` file, leveraging the `UserInputConfig` Pydantic model from Cycle 5 to provide immediate, clear feedback on any configuration errors.
3.  Instantiate the `WorkflowManager` from Cycle 5.
4.  Invoke the main `run()` method on the `WorkflowManager` instance.
5.  Catch any exceptions that propagate up from the workflow and present them to the user in a clean, readable format.

By the end of this cycle, the project will be a fully-fledged application that can be installed and run from the command line, providing a polished and professional user experience that abstracts the immense complexity of the underlying workflow into a single, simple command.

## 2. System Architecture

The architecture for Cycle 7 introduces the top-level application layer, `app.py`, which integrates all previously designed components into a cohesive, runnable program.

**File Structure for Cycle 7:**

The new files for this cycle are highlighted in **bold**.

```
mlip-autopipe/
├── dev_documents/
│   └── system_prompts/
│       └── CYCLE07/
│           ├── **SPEC.md**
│           └── **UAT.md**
├── mlip_autopipec/
│   ├── **app.py**              # CLI entry point using Typer
│   ├── workflow_manager.py
│   ├── config/
│   │   └── models.py
│   └── ...
├── tests/
│   ├── **test_app.py**         # Tests for the CLI using CliRunner
│   └── ...
└── pyproject.toml              # Will be updated with an entry point
```

**Component Blueprint: `app.py`**

This file will contain all the code for the Command Line Interface.

-   **`app = typer.Typer()`**: An instance of the main Typer application.
-   **`@app.command()`**: The decorator to define a CLI command.
-   **`run(config_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input.yaml configuration file."))`**: The main function that executes the workflow.
    -   It uses Typer's built-in validation to ensure the `config_file` argument is a path that exists and is readable, providing automatic, user-friendly errors if these conditions are not met.
    -   **Step 1: Load and Parse Config.** The function will contain a `try...except` block to handle YAML parsing errors and Pydantic validation errors. It will load the YAML file and attempt to parse it into the `UserInputConfig` model.
    -   **Step 2: Initialize Workflow.** If parsing is successful, it will print a confirmation message to the console (e.g., using `rich` for formatted output) and instantiate the `WorkflowManager`.
    -   **Step 3: Launch Workflow.** It will call `workflow_manager.run()`. This call will be wrapped in another `try...except` block to catch high-level `WorkflowError` exceptions that might be raised from the backend.
    -   **Error Handling:** The `except` blocks will print clean, user-facing error messages and exit the application with a non-zero status code to indicate failure.

**Project Entry Point:**

To make the CLI runnable via a simple command like `mlip-auto`, an entry point will be added to `pyproject.toml`:

```toml
[project.scripts]
mlip-auto = "mlip_autopipec.app:app"
```
This tells the package manager to create an executable script named `mlip-auto` that calls the `app` object in our `app.py` module.

## 3. Design Architecture

The design of the CLI is centered on the principle of a **thin, non-blocking interface**. The CLI's only job is to translate a user's command-line action into a call to the core application logic residing in the `WorkflowManager`.

**CLI Command Structure:**

-   **`mlip-auto`**: The main application command.
    -   **`run`**: The primary command to execute a workflow.
        -   **`CONFIG_FILE` (Argument):** The path to the `input.yaml` file. This is a required argument.

**Data Flow:**

1.  The user executes `$ mlip-auto run /path/to/my_project.yaml` in their terminal.
2.  The `setuptools` entry point invokes the `app` object in `mlip_autopipec.app`.
3.  Typer parses the command and arguments, identifying the `run` command and the value for `config_file`.
4.  Typer performs its built-in checks (`exists=True`, etc.) on the `config_file` argument.
5.  If checks pass, Typer calls the `run()` Python function, passing the `Path` object as an argument.
6.  The `run()` function reads the YAML file.
7.  The YAML data is passed to `UserInputConfig.model_validate()`. Pydantic performs the deep validation defined in Cycle 5.
8.  If validation passes, the `UserInputConfig` object is used to instantiate the `WorkflowManager`.
9.  The `workflow_manager.run()` method is called. At this point, control is handed over to the backend, and the CLI layer's job is essentially done, aside from waiting for the process to complete and handling any top-level exceptions.

This design ensures a strong separation of concerns. The `app.py` module knows nothing about DFT, Dask, or machine learning. It only knows how to parse a config file and start the workflow. This makes the core logic highly testable, independent of the CLI.

**User Feedback:**

The CLI will use a library like `rich` to provide pleasant and informative console output.
-   **On start:** A title and a summary of the validated configuration.
-   **On error:** A clearly marked "ERROR" message, printed in red, with a helpful description of what went wrong.
-   **On success:** A confirmation message printed in green.

## 4. Implementation Approach

1.  **Add Dependencies:** Add `typer`, `rich`, and `pyyaml` to the project dependencies in `pyproject.toml`.
2.  **Create `app.py`:** Create the `mlip_autopipec/app.py` file.
3.  **Initialize Typer App:** Add `import typer` and `app = typer.Typer()`.
4.  **Implement the `run` Command:**
    -   Define the `run` function with the `@app.command()` decorator.
    -   Add the `config_file` argument with Typer's `Argument` class for validation.
    -   Implement the logic to read the file, validate it with `UserInputConfig`, instantiate `WorkflowManager`, and call `run()`.
    -   Add the `try...except` blocks for `ValidationError` and a custom `WorkflowError`, printing formatted messages to the console using `rich`.
5.  **Add Entry Point to `pyproject.toml`:** Add the `[project.scripts]` table to `pyproject.toml` to make the `mlip-auto` command available after installation.
6.  **Install for Testing:** Run `pip install -e .` in the terminal. This will install the package in editable mode and create the `mlip-auto` executable, making it available for testing.

## 5. Test Strategy

Testing the CLI involves using a dedicated test runner that can invoke the command line application and capture its output and exit codes. `typer.testing.CliRunner` is the standard tool for this.

**Unit Testing Approach (Min 300 words):**

The unit tests in `tests/test_app.py` will test the behavior of the CLI in isolation by mocking the `WorkflowManager` so that the actual complex backend logic is not run.

-   **`test_run_success`:** This test will validate the "happy path."
    1.  It will create a temporary, valid `config.yaml` file.
    2.  It will use `unittest.mock.patch` to replace the `WorkflowManager` class with a mock.
    3.  It will use `CliRunner.invoke` to run the `run` command with the path to the temporary config file.
    4.  The assertions will be:
        -   The exit code of the process is 0.
        -   The mock `WorkflowManager` was instantiated exactly once with the correct configuration.
        -   The `run` method on the mock `WorkflowManager` instance was called exactly once.
        -   The console output contains a success message like "Workflow started".

-   **`test_run_config_not_found`:** This test validates Typer's built-in file checking.
    1.  It will use `CliRunner.invoke` to run the `run` command with a path that does not exist.
    2.  It will assert that the exit code is non-zero (e.g., 2 for CLI argument errors).
    3.  It will assert that the captured `stderr` contains the helpful message from Typer, like "File 'no_such_file.yaml' not found."

-   **`test_run_invalid_config`:** This test validates our custom error handling for Pydantic.
    1.  It will create a temporary `config.yaml` file with a known validation error (e.g., composition does not sum to 1.0).
    2.  It will invoke the `run` command.
    3.  It will assert that the exit code is non-zero (e.g., 1 for application errors).
    4.  It will assert that the captured `stderr` contains the specific Pydantic validation message, proving that our `try...except` block is working correctly.

**Integration Testing Approach (Min 300 words):**

The integration test provides confidence that the CLI can successfully launch the *real* workflow.

-   **`test_cli_launches_short_mock_workflow`:** This test will run the CLI from a subprocess and have it execute a minimal, fast, and predictable workflow.
    1.  **Setup:** Create a valid `config.yaml`. The test will patch the `WorkflowManager.run` method to replace the real, complex workflow with a simple mock function that, for example, prints "Workflow running...", sleeps for 1 second, and then prints "Workflow finished."
    2.  **Execution:** The test will use `subprocess.run` to execute the CLI command: `['mlip-auto', 'run', 'test_config.yaml']`. It will capture `stdout` and `stderr`.
    3.  **Assertion:** The assertions will confirm that the end-to-end process was launched successfully.
        -   Assert that the subprocess exit code is 0.
        -   Assert that `stdout` contains both "Workflow running..." and "Workflow finished."
        -   Assert that `stderr` is empty.

This test proves that the `mlip-auto` command is correctly installed, that it can parse a real configuration file, instantiate the real `WorkflowManager`, and that a call to its `run` method is successfully made. It validates the entire application wiring, from the command line to the core logic.
