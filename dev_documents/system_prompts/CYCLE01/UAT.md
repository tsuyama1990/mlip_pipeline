# Cycle 01 UAT: Core Framework Verification

## 1. Summary

This document describes the User Acceptance Tests (UAT) for the Core Framework of the PyAceMaker system. The goal is to verify that the system can load configuration, initialize components, and log activity correctly.

## 2. Test Scenarios

### 2.1. Basic Configuration Loading
*   **Goal**: Verify the system can read a valid `config.yaml` file.
*   **Input**: A minimal `config.yaml` with essential settings.
*   **Expected Output**:
    *   System starts without errors.
    *   Logs "Configuration loaded successfully".
    *   Creates directories specified in `config.orchestrator.work_dir`.

### 2.2. Invalid Configuration Handling
*   **Goal**: Verify the system rejects invalid configuration.
*   **Input**: A `config.yaml` missing required fields (e.g., `orchestrator.work_dir`).
*   **Expected Output**:
    *   System exits with a clear error message (ValidationError).
    *   Does NOT start the Orchestrator.

### 2.3. CLI Entry Point
*   **Goal**: Verify the CLI command `mlip-run config.yaml` works.
*   **Input**: Command line execution.
*   **Expected Output**:
    *   Same as 2.1.

## 3. Execution Steps

1.  **Prepare**: Create a temporary directory.
2.  **Config**: Create `config_valid.yaml` and `config_invalid.yaml`.
3.  **Execute**: Run the `mlip-run` command (simulated via python script).
4.  **Verify**: Check exit codes and log files.

## 4. Acceptance Criteria
*   The system must parse `config.yaml` using Pydantic V2.
*   The system must create the directory structure: `active_learning/`, `potentials/`, `data/`.
*   The system must log to both console and file.
