# CYCLE 06 UAT: Validation & Scale-Up

## 1. Test Scenarios

### Scenario 06.1: Quality Gate (Validation)
*   **Priority**: High
*   **Objective**: Verify that bad potentials are rejected.
*   **Description**: Train a potential on very little data (expected to be bad). Run validation.
*   **Success Criteria**:
    *   Validation Report shows "FAILED".
    *   Reason: "Phonon instability detected" or "Force RMSE > Threshold".
    *   Potential is NOT deployed to `production/`.

### Scenario 06.2: kMC Ordering
*   **Priority**: Critical
*   **Objective**: Verify the "Ordering" phase of the Fe/Pt scenario.
*   **Description**: Run EON with the trained potential on a disordered FePt cluster.
*   **Success Criteria**:
    *   EON runs multiple steps.
    *   The system energy decreases over time (finding more stable configurations).
    *   Visual inspection shows formation of L10-like order (layers of Fe and Pt).

## 2. Behavior Definitions

```gherkin
Feature: Quality Assurance

  Scenario: Phonon Stability Check
    GIVEN a trained potential
    WHEN the Validator runs the phonon dispersion calculation
    THEN it should check for imaginary frequencies
    AND if found, mark the potential as "Unstable"

  Scenario: kMC Driver Generation
    GIVEN an EON simulation request
    WHEN the Orchestrator prepares the run
    THEN it should generate a "pace_driver.py" script
    AND this script should correctly wrap the "potential.yace" for EON
```
