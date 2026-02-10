# Cycle 06: The OTF Loop Integration - UAT

## 1. Test Scenarios

### Scenario 1: Basic Halt and Retrain (Mock)
*   **ID**: UAT-06-001
*   **Objective**: Verify the core "Halt & Diagnose" loop execution.
*   **Pre-conditions**: Mock dynamics configured to halt at step 10.
*   **Steps**:
    1.  Run the OTF loop.
    2.  Check the log file.
*   **Expected Result**:
    *   "Dynamics Halted at step 10" is logged.
    *   "Extracting local candidates..." is logged.
    *   "Generating DFT data..." is logged.
    *   "Updating potential..." is logged.
    *   "Resuming simulation from restart_10.mpi" is logged.

### Scenario 2: Local Candidate Generation
*   **ID**: UAT-06-002
*   **Objective**: Ensure candidates are generated around the high-uncertainty atom.
*   **Pre-conditions**: A structure with a high $\gamma$ atom (index 5) exists.
*   **Steps**:
    1.  Call `generator.generate_local_candidates(structure, index=5)`.
    2.  Inspect the output structures.
*   **Expected Result**:
    *   The central atom (index 5) is perturbed significantly more than others.
    *   The total number of atoms remains the same (if not adding defects).
    *   Candidates are physically reasonable (no overlap).

### Scenario 3: Loop Termination (Convergence)
*   **ID**: UAT-06-003
*   **Objective**: Ensure the loop stops when the simulation completes without halts.
*   **Pre-conditions**: Mock dynamics configured to run successfully.
*   **Steps**:
    1.  Run the OTF loop.
    2.  Check the result.
*   **Expected Result**:
    *   The loop runs once.
    *   "Dynamics Completed Successfully" is logged.
    *   Orchestrator exits with code 0.

## 2. Behavior Definitions

```gherkin
Feature: Active Learning Loop

  Scenario: OTF Loop Execution
    Given the current potential is outdated
    When the dynamics engine detects high uncertainty
    Then the system should halt the simulation
    And it should identify the problematic atomic configuration
    And it should generate local candidates around the high-uncertainty region
    And it should run DFT calculations on these candidates
    And it should update the potential
    And it should resume the simulation with the new potential

  Scenario: Convergence
    Given the current potential is accurate
    When the dynamics engine runs the simulation
    Then the uncertainty should remain below the threshold
    And the simulation should complete successfully
    And the loop should terminate
```
