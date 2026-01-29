# User Acceptance Testing (UAT): Cycle 02

## 1. Test Scenarios

In Cycle 02, we are testing the "Muscle" of the system: the ability to exert force (run simulations). The UAT focuses on the robustness of this execution.

### Scenario 2.1: The "One-Shot" MD Run
-   **ID**: UAT-C02-01
-   **Priority**: Critical
-   **Description**: The user wants to verify that the system can actually run a simulation.
-   **Success Criteria**:
    -   Running `mlip-auto run-one-shot` (or similar CLI command) completes successfully.
    -   A working directory is created (e.g., `_work/job_...`).
    -   The directory contains `in.lammps`, `data.lammps`, and `dump.lammpstrj`.
    -   The CLI outputs "Simulation Completed: Status DONE".

### Scenario 2.2: Missing Executable Handling
-   **ID**: UAT-C02-02
-   **Priority**: High
-   **Description**: The user forgot to install LAMMPS or specified the wrong path in `config.yaml`.
-   **Success Criteria**:
    -   The system should NOT hang or print a generic Python Traceback.
    -   It should print a friendly error: "Executable 'lmp_serial' not found at /usr/bin/lmp. Please check your config."

### Scenario 2.3: Simulation Instability (The "Explosion")
-   **ID**: UAT-C02-03
-   **Priority**: Medium
-   **Description**: The simulation crashes due to bad physics (simulated by a very high timestep or bad potential parameters).
-   **Success Criteria**:
    -   The system detects the non-zero exit code of LAMMPS.
    -   The job status is reported as `FAILED`.
    -   The tail of `log.lammps` is printed to the console to help the user debug.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Basic Molecular Dynamics Execution

  Background:
    Given a valid "config.yaml"
    And the LAMMPS executable path is correctly configured

  Scenario: Run a standard NVT simulation
    When I run the command "mlip-auto run-cycle-02"
    Then the system should generate a structure for "Si"
    And it should create a job directory "_work_md/"
    And the process should exit with code 0
    And the job status should be "COMPLETED"
    And a file "dump.lammpstrj" should exist

  Scenario: Handle invalid executable path
    Given the config "lammps.command" is set to "/path/to/nothing"
    When I run the command "mlip-auto run-cycle-02"
    Then the exit code should be 1
    And the output should contain "Executable not found"

  Scenario: Handle simulation timeout
    Given the config "lammps.timeout" is set to "1s"
    And I run a simulation that takes 5s
    When I execute the run command
    Then the job status should be "TIMEOUT"
    And the system should gracefully terminate the subprocess
```
