# Cycle 01: User Acceptance Testing (UAT)

## 1. Test Scenarios

The goal of Cycle 01 is to establish a solid foundation. If the configuration cannot be parsed or the database cannot be written to, the rest of the project is moot. These tests simulate the very first steps a user would take.

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-01-01** | High | Configuration Loading | Verify that the system can load a user-provided `input.yaml` and validate its contents against the defined schema. This checks the "Zero-Human" requirement by ensuring invalid inputs are caught early. |
| **UAT-01-02** | High | Database Operations | Verify that atomic structures can be saved to and retrieved from the persistent storage (ASE database). This checks the "Provenance" requirement. |
| **UAT-01-03** | Medium | Logging Verification | Confirm that system activities are logged to both the console and a log file with correct formatting. This is critical for "Robustness" and debugging. |

### Recommended Notebooks
*   `notebooks/UAT_01_Config_and_DB.ipynb`: A step-by-step notebook that:
    1.  Defines a sample YAML configuration string.
    2.  Parses it using the `MinimalConfig` model.
    3.  Initializes a `DatabaseManager`.
    4.  Creates a dummy atom (e.g., Fe crystal).
    5.  Saves it to the DB and queries it back.

## 2. Behavior Definitions

### UAT-01-01: Configuration Loading

**Narrative**:
The user, a materials scientist, wants to start a new project to study Iron-Nickel alloys. They prepare a simple text file `input.yaml` specifying the elements and the goal ("melt_quench"). They intentionally make a mistake in one version (empty elements list) to verify the system warns them. They expect the system to accept the valid file and populate the internal settings with sensible defaults (like using 4 cores).

```gherkin
Feature: Configuration Parsing

  Scenario: Loading a valid configuration
    GIVEN a valid YAML configuration file "input.yaml" containing:
      """
      project_name: "TestProject"
      target_system:
        elements: ["Fe", "Ni"]
        composition: {"Fe": 0.7, "Ni": 0.3}
      simulation_goal:
        type: "melt_quench"
        temperature_range: [300, 1500]
      """
    WHEN the configuration is parsed by the MinimalConfig model
    THEN no validation errors should be raised
    AND the "elements" field should contain ["Fe", "Ni"]
    AND the "project_name" should be "TestProject"
    AND the SystemConfig should be automatically populated with default resources
    AND "dft_command" in SystemConfig should strictly match the default MPI command template

  Scenario: Loading an invalid configuration (Empty Elements)
    GIVEN an invalid YAML configuration where "elements" is an empty list
    WHEN the configuration is parsed
    THEN a ValidationError should be raised
    AND the error message should explicitly mention "elements" cannot be empty
    AND the application should exit with a non-zero status code

  Scenario: Loading an invalid configuration (Negative Temperature)
    GIVEN an invalid YAML configuration where "temperature_range" contains -100
    WHEN the configuration is parsed
    THEN a ValidationError should be raised
    AND the error message should mention physical constraints
```

### UAT-01-02: Database Persistence

**Narrative**:
The system has generated a candidate structure (a Silicon crystal). It needs to save this structure to the database so that the DFT worker can pick it up later. The system also needs to tag this structure with "source=test" so we know where it came from. Later, the system queries the database to count how many "test" structures exist.

```gherkin
Feature: Database Persistence

  Scenario: Saving and Retrieving Atoms
    GIVEN a DatabaseManager initialized with "test.db"
    AND an ASE Atoms object representing a Silicon crystal
    AND a metadata dictionary {"source": "manual_test", "uuid": "12345"}
    WHEN the atom is added to the database with the metadata
    THEN the database count should increase by 1
    AND querying the database for source="manual_test" should return exactly 1 row
    AND the retrieved structure should match the original Silicon crystal
    AND the "uuid" key should be present in the key-value pairs
    AND the file "test.db" should exist on the filesystem

  Scenario: Concurrent Writes
    GIVEN a DatabaseManager initialized with "concurrent.db"
    WHEN 5 different threads attempt to write a structure simultaneously
    THEN the database should not be corrupted
    AND the final count of structures should be 5
```

### UAT-01-03: Logging Verification

**Narrative**:
The system is running in the background. The user wants to know what's happening. They check the `system.log` file. They expect to see a timestamped entry saying "Database initialized". They also check the console output and expect to see a simpler message.

```gherkin
Feature: System Observability

  Scenario: Logging to File and Console
    GIVEN the logging system is configured with file level DEBUG and console level INFO
    WHEN an INFO level message "System Start" is logged
    THEN the console should display "System Start"
    AND the log file should contain "System Start" with a timestamp prefix

    WHEN a DEBUG level message "Connecting to DB..." is logged
    THEN the console should NOT display the message
    BUT the log file SHOULD contain the message
```
