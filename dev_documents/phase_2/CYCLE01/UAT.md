# Cycle 01 UAT: Core Framework & User Interface

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-01-01** | High | **Valid Project Initialization** | Verify that a user can initialize a new project by providing a valid `input.yaml`. The system should create the directory structure, database, and log files without error. The user should see a confirmation message. |
| **UAT-01-02** | High | **Invalid Configuration Handling** | Verify that the system provides clear, human-readable error messages when the user provides an invalid configuration (e.g., composition sums to 0.9 instead of 1.0, or invalid chemical element like 'Xy'). The process must exit with a non-zero code. |
| **UAT-01-03** | Medium | **Database & Provenance Check** | Verify that the initialized database contains the configuration settings in its metadata. This ensures that every run is traceable back to its input parameters. The user should be able to query this metadata using standard ASE tools. |
| **UAT-01-04** | Low | **Idempotency** | Verify that running the initialization command twice on the same directory does not destroy existing data (or warns the user). |

### Recommended Demo
Create a Jupyter Notebook `demo_01_initialization.ipynb` to demonstrate these capabilities.
1.  **Block 1**: Write a valid `input.yaml` file to disk using Python file I/O.
2.  **Block 2**: Run the system initialization command via `!mlip-auto run input.yaml`.
3.  **Block 3**: Use `ls -R` or Python's `os.walk` to display the created folder hierarchy.
4.  **Block 4**: Use `ase.db` to connect to the generated `project.db` and print the `metadata` dictionary to prove the config was saved.
5.  **Block 5**: Write an *invalid* yaml file and run the command again, showing the error message.

## 2. Behavior Definitions

### Scenario: Valid Project Initialization
**GIVEN** a clean working directory (no existing project files).
**AND** a file named `input.yaml` containing:
```yaml
project_name: "AlCu_Alloy"
target_system:
  elements: ["Al", "Cu"]
  composition: {"Al": 0.5, "Cu": 0.5}
resources:
  dft_code: "quantum_espresso"
  parallel_cores: 4
```
**WHEN** the user executes the command `mlip-auto run input.yaml`
**THEN** a directory named `AlCu_Alloy` should be created in the current path.
**AND** a file `AlCu_Alloy/project.db` should exist and be a valid SQLite file.
**AND** a file `AlCu_Alloy/system.log` should exist and contain text.
**AND** the standard output should display a success message like "System initialized successfully".
**AND** the log file should contain the timestamp of initialization.

### Scenario: Invalid Configuration (Composition Error)
**GIVEN** a file named `bad_input.yaml` containing a composition `{"Fe": 0.5}` (sum is not 1.0).
**WHEN** the user executes the command `mlip-auto run bad_input.yaml`
**THEN** the system should exit with a non-zero error code (e.g., 1).
**AND** the standard output (or stderr) should contain a `ValidationError`.
**AND** the error message should explicitly state: "Composition must sum to 1.0".
**AND** no project directory `AlCu_Alloy` should be created (or at least no corrupt state should persist).
**AND** the user should not see a raw Python traceback, but a formatted error message.

### Scenario: Database Metadata Persistence
**GIVEN** that the project `AlCu_Alloy` has been successfully initialized.
**WHEN** a user or script connects to `AlCu_Alloy/project.db` using the ASE database interface:
```python
import ase.db
db = ase.db.connect('AlCu_Alloy/project.db')
meta = db.metadata
```
**THEN** the `meta` dictionary should be non-empty.
**AND** `meta['minimal_config']['target_system']['elements']` should equal `['Al', 'Cu']`.
**AND** `meta['system_config']['working_dir']` should match the absolute path of the created directory.
**THIS** ensures data provenance is established from the very first step.

### Scenario: Idempotency check
**GIVEN** an already initialized project.
**WHEN** the user runs `mlip-auto run input.yaml` again.
**THEN** the system should log "Project already exists".
**AND** it should either exit gracefully or proceed without overwriting the existing database file (unless a `--force` flag is provided).
