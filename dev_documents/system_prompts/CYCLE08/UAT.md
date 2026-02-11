# Cycle 08 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 8.1: Stability Check (Born Criteria)
*   **Priority**: High
*   **Goal**: Verify physical validation.
*   **Action**:
    1.  Provide a stable cubic crystal (e.g., LJ Argon).
    2.  Invoke `Validator.check_elastic_stability()`.
*   **Expectation**:
    *   Returns `True`.
    *   Calculated $C_{11}, C_{12}, C_{44}$ are positive and satisfy $C_{11} - C_{12} > 0$.

### Scenario 8.2: Equation of State (EOS) Fitting
*   **Priority**: High
*   **Goal**: Verify bulk modulus calculation.
*   **Action**:
    1.  Provide a crystal.
    2.  Invoke `Validator.check_eos()`.
*   **Expectation**:
    *   Returns `True` (has minimum).
    *   Fitted Bulk Modulus $B_0$ is within 10% of reference value.
    *   Plot `eos.png` is generated.

### Scenario 8.3: Report Generation
*   **Priority**: Medium
*   **Goal**: Verify user feedback.
*   **Action**:
    1.  Mock a completed workflow state (5 iterations).
    2.  Invoke `Reporter.generate_report()`.
*   **Expectation**:
    *   File `report.html` is created.
    *   It contains "RMSE vs Cycle" plot.
    *   It contains a table of validation metrics.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Physical Validation

  Scenario: Potential passes stability tests
    Given a trained potential
    When the Validator runs elastic stability checks
    Then the Born criteria must be satisfied
    And the bulk modulus must be positive

  Scenario: Reporter summarizes results
    Given a completed active learning campaign
    When the report is generated
    Then it must include learning curves
    And it must include physical property validation status
```
