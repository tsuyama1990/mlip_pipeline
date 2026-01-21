# Cycle 01 UAT: Foundation & Configuration

## 1. Test Scenarios

### Scenario 1.1: System Initialization
-   **Priority**: Critical
-   **Description**: A new user installs the package and wants to set up a new project. They should be able to generate a default configuration file template without needing to consult external documentation. This tests the "Zero-Human" onboarding experience. The goal is to ensure the barrier to entry is as low as possible.
-   **Pre-conditions**:
    -   The user is in a clean, empty directory.
    -   The `mlip-auto` package is installed and available in the `$PATH`.
    -   Python 3.11+ is the active environment.
-   **Detailed Steps**:
    1.  User opens a terminal window.
    2.  User executes the command `mlip-auto init`.
    3.  System checks for existing config files (e.g., `input.yaml`).
    4.  System determines the file does not exist.
    5.  System generates `input.yaml` from an internal template.
    6.  System prints a success message: "Initialized new project. Please edit input.yaml."
    7.  User runs `ls -l` to verify file existence and permissions.
    8.  User opens `input.yaml` in a text editor.
    9.  User verifies that the file contains commented-out placeholders for `target_system`, `dft`, `training`, and `inference` sections.
-   **Post-conditions**:
    -   `input.yaml` exists in the current directory.
    -   The file is a valid YAML file (not empty).
-   **Failure Modes**:
    -   Permission denied (cannot write to directory).
    -   File already exists (should prompt overwrite or fail).

### Scenario 1.2: Configuration Validation (Success Path)
-   **Priority**: High
-   **Description**: The user has edited the `input.yaml` file with valid parameters for a real material (FeNi alloy). They want to confirm that the system accepts these parameters before launching any long-running jobs. This verifies the Pydantic schema validation logic.
-   **Pre-conditions**:
    -   A valid `input.yaml` exists in the current directory.
    -   The `pseudopotential_dir` path specified in the config actually exists on the filesystem.
-   **Detailed Steps**:
    1.  User ensures `pseudopotential_dir` points to a real folder containing `.UPF` files.
    2.  User executes `mlip-auto check-config input.yaml`.
    3.  System loads the YAML file into a Python dictionary.
    4.  System instantiates the `MLIPConfig` Pydantic model.
    5.  System validates types (floats, strings) and physical constraints (cutoffs > 0).
    6.  System prints a success message in green (using Rich): "Configuration is valid."
-   **Post-conditions**:
    -   Exit code is 0.
    -   No Python stack traces are visible to the user.
-   **Failure Modes**:
    -   YAML syntax error (indentation).
    -   Missing required fields.

### Scenario 1.3: Configuration Validation (Failure Path - Logical Error)
-   **Priority**: High
-   **Description**: The user makes a semantic error in the configuration, such as setting a negative temperature or providing a non-existent path. The system must catch this *before* execution starts.
-   **Pre-conditions**:
    -   `input.yaml` exists but contains `ecutwfc: -30.0` (physically impossible).
-   **Detailed Steps**:
    1.  User executes `mlip-auto check-config input.yaml`.
    2.  System parses the file.
    3.  Pydantic validator triggers on the `ecutwfc` field.
    4.  System catches the `ValidationError`.
    5.  System prints a formatted error message: "Error in DFT Config: ecutwfc must be greater than 0."
    6.  System exits with a non-zero status code.
-   **Post-conditions**:
    -   Exit code is 1.
    -   The error message clearly points to the specific field `dft.ecutwfc`.
-   **Failure Modes**:
    -   System crashes with a raw traceback (Bad user experience).
    -   System silently accepts the bad value (Dangerous).

### Scenario 1.4: Database Initialization
-   **Priority**: Critical
-   **Description**: The system must be able to create its persistence layer. This tests the integration with `ase.db` and SQLite.
-   **Pre-conditions**:
    -   Valid config file present.
    -   Write permissions in the directory.
-   **Detailed Steps**:
    1.  User executes `mlip-auto db init`.
    2.  System reads config to find `database_path` (default `mlip.db`).
    3.  System initializes the `DatabaseManager`.
    4.  System creates the SQLite file.
    5.  System sets up initial metadata tables (if any schema migration is needed).
    6.  User checks file size > 0 bytes using `ls -l`.
    7.  User attempts to connect to it using `sqlite3 mlip.db ".tables"`.
-   **Post-conditions**:
    -   The `.db` file is created.
    -   It is a valid SQLite database.
-   **Failure Modes**:
    -   Disk full.
    -   Path is a directory, not a file.

## 2. Behaviour Definitions

```gherkin
Feature: Configuration Management
  As a computational scientist
  I want to initialize and validate project configurations
  So that I don't waste time running simulations with bad parameters that will crash later

  Scenario: User initializes a new project in an empty folder
    Given the current directory is empty
    When I run the command "mlip-auto init"
    Then a file named "input.yaml" should be created in the current directory
    And the file should contain a "target_system" section with "elements" and "composition"
    And the file should contain a "dft" section with "ecutwfc"

  Scenario: User validates a syntactically correct configuration
    Given a file "good_config.yaml" exists
    And the file contains "elements: ['Fe', 'Ni']"
    And the file contains "ecutwfc: 50.0"
    And the file contains "pseudopotential_dir: /tmp"
    When I run the command "mlip-auto check-config good_config.yaml"
    Then the system exit code should be 0
    And the standard output should contain "Validation Successful" in green text

  Scenario: User validates a configuration with logical errors (Negative Cutoff)
    Given a file "bad_config.yaml" exists
    And the file contains "ecutwfc: -100.0"
    When I run the command "mlip-auto check-config bad_config.yaml"
    Then the system exit code should be 1
    And the standard output should contain "ensure this value is greater than 0"
    And the error message should specify the field "dft.ecutwfc"

Feature: Database Management
  As a system administrator
  I need to persist atomic structures in a structured format
  So that I can train models later and query the data provenance

  Scenario: Initialize Database
    Given a valid configuration file specifying "mlip.db"
    When I run the command "mlip-auto db init"
    Then a SQLite database file named "mlip.db" should be created
    And the database should have 0 rows initially
    And the database should allow connections via ase.db
```
