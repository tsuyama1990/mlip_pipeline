# User Acceptance Test (UAT): Cycle 01

## 1. Test Scenarios

### Scenario 01-01: The First Breath (Priority: High)
**Objective**: Verify that the system can initialize, read configuration, and complete a single "Skeleton Cycle" without crashing.

**Description**:
In this scenario, we simulate a user starting a new project. They have prepared a configuration file and a dataset. They expect the system to run, process the configuration, and "pretend" to train a potential (or actually train one if the binary is available). This corresponds to the "Zero-Config" promise—if the config is valid, the system just works.

**User Journey**:
1.  The user creates a folder `project_test`.
2.  The user places a `config.yaml` and a `data.pckl` (can be empty/dummy) in the folder.
3.  The user runs the command `python -m mlip_autopipec.main config.yaml`.
4.  The terminal displays structured logs showing the progress: "Initializing...", "Loading Config...", "Starting Cycle 0...".
5.  After a few seconds, the command finishes with "Workflow completed successfully."
6.  The user checks the folder and sees a new `state.json` file and an `output.yace` file.

**Success Criteria**:
*   Process exit code is 0.
*   Log output contains no "ERROR" or "CRITICAL" lines.
*   `state.json` reflects that iteration has incremented.

### Scenario 01-02: The Guard Rails (Priority: Medium)
**Objective**: Verify that the system provides helpful feedback when the configuration is invalid.

**Description**:
Users make mistakes. They might typo a filename or forget a mandatory setting. The system must catch this immediately.

**User Journey**:
1.  The user creates a `bad_config.yaml` where the `dataset_path` points to a non-existent file.
2.  The user runs the command.
3.  The system **immediately** stops.
4.  The terminal displays a clear error message: "Validation Error: file 'missing_data.pckl' does not exist."

**Success Criteria**:
*   Process exit code is non-zero (usually 1).
*   The error message is human-readable (not just a raw Python stack trace).

## 2. Behavior Definitions (Gherkin)

### Feature: Configuration Loading

```gherkin
Feature: Load and Validate Configuration

  Scenario: User provides a valid configuration file
    GIVEN a file named "config.yaml" exists
    AND the file contains valid YAML syntax
    AND the "dataset_path" field points to an existing file
    WHEN the system is started with "config.yaml"
    THEN the configuration should be loaded into memory
    AND no validation errors should be raised
    AND the logging level should be set according to the config

  Scenario: User provides a configuration with missing file
    GIVEN a file named "bad_config.yaml" exists
    AND the "dataset_path" field is "ghost.pckl"
    AND the file "ghost.pckl" does not exist
    WHEN the system is started with "bad_config.yaml"
    THEN the system should exit with an error code
    AND the output should contain "file not found" or "does not exist"
```

### Feature: Workflow Execution

```gherkin
Feature: Basic Active Learning Loop (Skeleton)

  Scenario: Execute a single cycle with mock trainer
    GIVEN the configuration "max_iterations" is set to 1
    AND the environment variable "PYACEMAKER_MOCK_MODE" is "1"
    WHEN the Orchestrator run method is called
    THEN the system should log "Starting Cycle 0"
    AND the system should call the Trainer module
    AND the Trainer should produce a dummy "output.yace" file
    AND the Workflow State should be saved to "state.json"
    AND the system should log "Cycle 0 completed"
```
