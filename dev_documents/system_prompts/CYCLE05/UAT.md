# User Acceptance Test (UAT): Cycle 05

## 1. Test Scenarios

### Scenario 05-01: The Report Card (Priority: High)
**Objective**: Verify that the system generates a comprehensive HTML report after a cycle.

**Description**:
The user wants to see "what happened" during the night. They open the report to check the convergence and physical properties.

**User Journey**:
1.  User navigates to `active_learning/iter_005/`.
2.  User opens `report.html` in their browser.
3.  User sees a summary table: "Status: PASSED".
4.  User sees a Phonon Band Structure plot.
5.  User sees a table of Elastic Constants ($C_{11}, C_{12}, ...$) compared to DFT reference values.

**Success Criteria**:
*   The HTML file exists and renders correctly in Chrome/Firefox.
*   Plots are present (images or interactive).

### Scenario 05-02: The Gatekeeper (Priority: Medium)
**Objective**: Verify that the system correctly rejects a physically invalid potential.

**Description**:
The system trained a potential that is numerically accurate (low RMSE) but physically unstable (soft mode). The validator must catch this.

**User Journey**:
1.  The system completes training.
2.  The `PhononValidator` detects an imaginary frequency of -2.0 THz at the X point.
3.  The Orchestrator logs "Validation FAILED: Unstable phonon mode detected."
4.  The system marks the potential as "REJECTED" in the state.
5.  (Optional) The system attempts to add more data or stop.

**Success Criteria**:
*   The `WorkflowState` shows the iteration status as "FAILED" or "WARNING".
*   The report clearly highlights the failure in red.

## 2. Behavior Definitions (Gherkin)

### Feature: Validation Suite

```gherkin
Feature: Physics Validation

  Scenario: Phonon Stability Check
    GIVEN a trained potential
    AND a crystal structure
    WHEN the PhononValidator runs
    THEN it should calculate the band structure
    AND if the minimum frequency is negative (imaginary), it should fail
    AND it should generate a plot of the dispersion

  Scenario: Elastic Stability Check
    GIVEN a trained potential
    WHEN the ElasticValidator runs
    THEN it should calculate the stiffness tensor Cij
    AND it should verify the Born stability criteria for the crystal class
    AND if criteria are violated, it should fail
```
