# Cycle 06 UAT: Orchestrator & Validation

## 1. Test Scenarios

### SCENARIO 01: Full Active Learning Loop (Mocked)
**Priority**: Critical
**Description**: Verify that the Orchestrator correctly manages the lifecycle of the active learning process.
**Pre-conditions**: All components configured as Mock. Max cycles = 3.
**Steps**:
1.  Run `mlip-pipeline run config_mock.yaml`.
2.  Inspect the working directory.
**Expected Result**:
-   Folders `iter_001`, `iter_002`, `iter_003` are created.
-   Each folder contains `md_run`, `dft_calc`, `training` subdirectories.
-   Logs show the transition "Exploration -> Halt -> Retraining -> Validation".

### SCENARIO 02: Validation Gate
**Priority**: High
**Description**: Verify that the Validator prevents the deployment of physically unstable potentials.
**Pre-conditions**: A MockValidator configured to always fail (return `passed=False`).
**Steps**:
1.  Run the loop.
2.  Check the Orchestrator logic.
**Expected Result**:
-   The system logs "Validation Failed".
-   The potential is *not* promoted to "production".
-   (Optionally) The system attempts to recover or stops, depending on policy.

## 2. Behaviour Definitions

```gherkin
Feature: Orchestration

  Scenario: Convergence detection
    Given an Orchestrator running an exploration
    When the Dynamics engine reports "CONVERGED" (no halts)
    Then the Orchestrator should stop the loop
    And it should mark the project as "Complete"

  Scenario: Physical Validation
    Given a newly trained potential
    When I run the validation suite
    Then it should calculate Phonon spectrum
    And it should calculate Elastic constants
    And it should only pass if there are no imaginary frequencies (stable crystal)
```
