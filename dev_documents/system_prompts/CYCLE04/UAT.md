# Cycle 04 UAT: Automated DFT Factory

## 1. Test Scenarios

### Scenario 4.1: Successful Static Calculation
-   **Priority**: Critical
-   **Description**: Run a standard SCF calculation on a clean structure. This validates the happy path of the DFT factory.
-   **Pre-conditions**:
    -   DB has 1 structure with `status="selected"`.
    -   Quantum Espresso (`pw.x`) is installed or a valid mock is on the `$PATH`.
    -   Pseudopotential files are present.
-   **Detailed Steps**:
    1.  User executes `mlip-auto calculate`.
    2.  System queries DB for pending tasks. Finds one.
    3.  System creates a temporary directory `_work/calc_{uuid}`.
    4.  System writes `pw.in` with correct atomic positions and species.
    5.  System executes `pw.x < pw.in > pw.out`.
    6.  Process returns exit code 0.
    7.  System parses `pw.out`. Finds "JOB DONE". Extracts Energy (-1234.5 eV).
    8.  System updates DB with `status="completed"`.
-   **Post-conditions**:
    -   The structure in DB has valid float values for Energy and Forces.
    -   The working directory is cleaned up (optional, depending on config).
-   **Failure Modes**:
    -   `pw.x` not found.
    -   Pseudopotential missing.

### Scenario 4.2: Convergence Failure Recovery (The "Ladder")
-   **Priority**: High
-   **Description**: Simulate a hard-to-converge system to verify the self-healing logic. This is the core value proposition of the "Factory".
-   **Pre-conditions**:
    -   A structure known to be difficult (e.g., a slab with high spin).
    -   Alternatively, use a Mock Runner that is programmed to fail the first 2 times.
-   **Detailed Steps**:
    1.  System launches Attempt 1 (Default Params: `mixing_beta=0.7`).
    2.  Mock Runner waits 1 second and returns success, but writes "convergence not achieved" to the output file.
    3.  Parser reads output, raises `DFTConvergenceError`.
    4.  System catches error. Logs "Convergence Error. Retrying with Strategy: REDUCE_MIXING".
    5.  System launches Attempt 2 (Params: `mixing_beta=0.3`).
    6.  Mock Runner writes "convergence not achieved" again.
    7.  System catches error. Logs "Retrying with Strategy: CG_DIAGONALIZATION".
    8.  System launches Attempt 3.
    9.  Mock Runner writes "JOB DONE".
    10. System marks task as successful.
-   **Post-conditions**:
    -   The final status is `completed`.
    -   The logs show the history of retries.
-   **Failure Modes**:
    -   System gives up too early.
    -   System gets stuck in an infinite loop.

### Scenario 4.3: Fatal Error Handling
-   **Priority**: Medium
-   **Description**: Some errors are unrecoverable (e.g., Segmentation Fault, Disk Full). The system must recognize these and stop wasting resources.
-   **Pre-conditions**:
    -   Mock `pw.x` to return exit code 139 (Segfault).
-   **Detailed Steps**:
    1.  System launches Attempt 1.
    2.  Process crashes immediately with exit code 139.
    3.  System determines this is not a convergence error.
    4.  System marks structure as `failed`.
    5.  System logs the stderr output for debugging.
-   **Post-conditions**:
    -   Structure status is `failed`.
    -   Pipeline proceeds to the next structure in the queue.
    -   System does not hang.

## 2. Behaviour Definitions

```gherkin
Feature: DFT Execution Factory
  As an automated system
  I want to run quantum mechanical calculations robustly
  So that I can build a training dataset without human babysitting

  Scenario: Standard SCF Execution
    Given a selected candidate structure
    When the DFT Runner executes
    Then a Quantum Espresso input file should be generated with "tprnfor=.true."
    And the calculation should produce an output file
    And the energy and forces should be extracted to the database

  Scenario: Recovering from Convergence Failure
    Given a calculation that fails to converge within 100 steps
    When the runner detects the "convergence not achieved" message
    Then it should NOT mark the task as failed
    But it should generate a new input file with "mixing_beta" reduced
    And it should restart the calculation

  Scenario: Handling Garbage Output
    Given a calculation that finishes but prints NaN for forces
    When the parser reads the output
    Then it should raise a runtime error
    And the result should be discarded to prevent poisoning the training set
```
