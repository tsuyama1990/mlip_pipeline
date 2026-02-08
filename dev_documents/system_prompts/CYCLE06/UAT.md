# Cycle 06 UAT: Validation & Full Orchestration

## 1. Test Scenarios

### Scenario 6.1: Elastic Constants Validation
*   **ID**: S06-01
*   **Priority**: High
*   **Description**: Verify the Validator correctly calculates elastic constants.
*   **Steps**:
    1.  Use an LJ potential for Argon.
    2.  Run `validator.validate(potential)`.
    3.  Check the computed $C_{11}, C_{12}, C_{44}$.
*   **Expected Result**:
    *   Values should be close to literature values (or analytical solution for LJ FCC).
    *   Born stability criteria should PASS.

### Scenario 6.2: Full End-to-End Run (Mocked)
*   **ID**: S06-02
*   **Priority**: Critical
*   **Description**: Verify the entire pipeline runs from start to finish.
*   **Steps**:
    1.  Create a comprehensive `config.yaml` enabling all components (Mock versions).
    2.  Set `max_cycles: 3`.
    3.  Run `mlip-pipeline run config.yaml`.
*   **Expected Result**:
    *   Cycle 1: Initial generation -> Oracle -> Train -> Validate.
    *   Cycle 2: MD (Halt) -> Oracle (Halt Structure) -> Train -> Validate.
    *   Cycle 3: MD (Converged) -> Finish.
    *   Report generated in `workdir/report.html`.

### Scenario 6.3: Validation Failure Handling
*   **ID**: S06-03
*   **Priority**: Medium
*   **Description**: Verify that a bad potential is rejected.
*   **Steps**:
    1.  Create a Mock Trainer that produces a "Bad Potential" (e.g., one that returns random forces).
    2.  Run the validation step.
*   **Expected Result**:
    *   Validator reports FAIL on Elastic/Phonon tests.
    *   Orchestrator logs "Validation failed".
    *   Pipeline continues to next cycle (hoping for improvement) or aborts (depending on config).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Full Pipeline Orchestration

  Scenario: Successful Active Learning Loop
    GIVEN a complete system configuration
    WHEN I start the pipeline
    THEN it should generate an initial dataset
    AND train a potential
    AND run MD simulations
    AND if MD fails, it should automatically collect new data and retraining
    AND finally output a validated potential file

  Scenario: Validation Gatekeeper
    GIVEN a trained potential that predicts unstable phonons
    WHEN the validator runs
    THEN it should flag the potential as "UNSTABLE"
    AND the system should NOT deploy this potential to production
```
