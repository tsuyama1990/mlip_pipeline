# Cycle 01 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario ID: UAT-C01-001 - Project Initialization and Configuration Validation

**Priority:** High
**Description:**
This scenario verifies that a new user, potentially a materials scientist with limited programming experience, can successfully initialise a new MLIP-AutoPipe project and validate their configuration file. The "Zero-Human" philosophy begins here: the system must guide the user through setup and prevent invalid configurations from ever starting a job.

**User Story:**
As a Researcher, I want to create a new project directory and generate a template configuration file so that I can easily specify my material system (e.g., Fe-Ni alloy) and simulation parameters. Once I have edited the file, I want to run a validation command to ensure I haven't made any typos or logical errors (like setting a negative cutoff energy), saving me from discovering these errors days later.

**Step-by-Step Walkthrough:**
1.  **Installation**: The user installs the package (simulated via `pip install .` in the test environment).
2.  **Directory Creation**: The user creates a folder `my_new_alloy`.
3.  **Initialization**: The user runs `mlip-auto init`.
    -   *Expectation*: The system creates a default `input.yaml` file. This file is not empty but contains commented-out examples and sensible defaults (e.g., standard QE flags).
4.  **Configuration Editing**: The user edits `input.yaml`.
    -   *Action*: They set `project_name` to "SuperAlloy".
    -   *Action*: They intentionally make a mistake, setting `ecutwfc` to -50.0 to test the safety net.
5.  **Validation (Fail)**: The user runs `mlip-auto validate`.
    -   *Expectation*: The command fails with a clear, red-coloured error message: "Validation Error: 'ecutwfc' must be greater than or equal to 20.0 Ry."
6.  **Correction**: The user fixes the value to 60.0.
7.  **Validation (Success)**: The user runs `mlip-auto validate`.
    -   *Expectation*: The system prints a green "Configuration Validated Successfully" message. It also prints a summary of the interpreted config (e.g., "DFT Engine: Quantum Espresso, Parallel Cores: 64") to confirm the parser understood the intent.

**Success Criteria:**
-   The `init` command must generate a syntactically valid YAML file.
-   The `validate` command must catch Type Errors (string vs float).
-   The `validate` command must catch Value Errors (constraints like > 0).
-   The error messages must be human-readable, not raw Python stack traces.

### Scenario ID: UAT-C01-002 - Database Creation and Atomic Structure Persistence

**Priority:** High
**Description:**
This scenario verifies the backend data management capability. While the user might not directly interact with the database via SQL, they need confidence that the system can store and retrieve atomic structures without data loss. This acts as a "sanity check" for the persistence layer.

**User Story:**
As a Power User or Developer, I want to verify that the underlying database correctly handles my atomic structures. I want to manually insert a test structure (e.g., a Copper crystal) into the system's database and verify that it is assigned a unique ID and the correct 'Pending' status. This ensures that when the massive generation loop starts in Cycle 02, the foundation is solid.

**Step-by-Step Walkthrough:**
1.  **Setup**: The user opens a Jupyter Notebook or a Python script in the project directory.
2.  **Connection**: They import the `DatabaseManager` and point it to `mlip.db`.
3.  **Structure Creation**: They use ASE to build a simple structure: `atoms = bulk('Cu', 'fcc', a=3.6)`.
4.  **Insertion**: They call `db_manager.add_structure(atoms)`.
    -   *Expectation*: The method returns an ID (integer). No errors are raised.
5.  **Verification (Query)**: They query the database using `ase.db` or the manager's `get_structure(id)` method.
    -   *Expectation*: The retrieved object matches the original `atoms` object (positions, cell, numbers).
    -   *Expectation*: The `status` field in the database is automatically set to "pending".
    -   *Expectation*: The `config_type` (if provided) is stored correctly.
6.  **Concurrency Check (Optional)**: The user tries to write to the DB from two different notebook cells/terminals.
    -   *Expectation*: The SQLite database handles the lock gracefully (waiting or succeeding), ensuring no corruption.

**Success Criteria:**
-   The `mlip.db` file is created on disk if it doesn't exist.
-   Data round-trip (Write -> Read) preserves floating-point precision of atomic positions.
-   Metadata columns (`status`, `created_at`) are populated automatically.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Configuration Management
  As a User
  I want to configure the pipeline using a simple YAML file
  So that I can define my scientific goals without writing Python code

  Background:
    Given I have installed the "mlip_autopipec" package
    And I am in a clean project directory

  Scenario: Initialize a new project configuration
    When I run the command "mlip-auto init"
    Then a file named "input.yaml" should be created in the current directory
    And the file "input.yaml" should contain the key "dft"
    And the file "input.yaml" should contain the key "training"

  Scenario: Validate a correct configuration
    Given a file "input.yaml" exists with the following content:
      """
      project_name: "ValidProject"
      dft:
        command: "mpirun -np 4 pw.x"
        pseudopotential_dir: "./sssp"
        ecutwfc: 50.0
      training:
        potential_name: "my_pot"
      """
    When I run the command "mlip-auto validate"
    Then the exit code should be 0
    And the output should contain "Configuration Validated Successfully"

  Scenario: Reject configuration with invalid types
    Given a file "input.yaml" exists with the following content:
      """
      project_name: "BadTypeProject"
      dft:
        ecutwfc: "fifty"  # String instead of float
      """
    When I run the command "mlip-auto validate"
    Then the exit code should be 1
    And the output should contain "Input should be a valid number"

  Scenario: Reject configuration with physical constraints violation
    Given a file "input.yaml" exists with the following content:
      """
      project_name: "NegativeCutoff"
      dft:
        ecutwfc: -10.0  # Physically impossible
      """
    When I run the command "mlip-auto validate"
    Then the exit code should be 1
    And the output should contain "Input should be greater than or equal to 20"

Feature: Database Persistence
  As a System Component
  I want to store atomic structures reliably
  So that they can be processed by downstream workers

  Scenario: Auto-initialization of Database
    Given the file "mlip.db" does not exist
    When I initialize the DatabaseManager with path "mlip.db"
    And I perform a write operation
    Then the file "mlip.db" should be created on the disk

  Scenario: Storing an Atom with Status
    Given I have a valid ASE Atoms object representing "NaCl"
    When I add the structure to the database using DatabaseManager
    Then the entry should be retrievable by its ID
    And the stored "status" should be "pending"
    And the stored "positions" should match the original atoms within 1e-8 tolerance
```
