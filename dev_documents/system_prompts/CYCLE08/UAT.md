# Cycle 08 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: The "Zero-Config" Experience
*   **ID**: UAT-08-01
*   **Priority**: Critical
*   **Description**: A user with no coding knowledge installs the package and runs it.
*   **Success Criteria**:
    *   Installation via `uv sync` or `pip install .` works.
    *   `mlip-auto run config.yaml` starts the process.
    *   No Python exceptions occur during a 24-hour run.

### Scenario 2: Dashboard Visualization
*   **ID**: UAT-08-02
*   **Priority**: High
*   **Description**: Verify that the generated HTML report gives actionable insights.
*   **Success Criteria**:
    *   Open `report.html` in a browser.
    *   The RMSE plot clearly shows the error decreasing over generations.
    *   The Phonon plot shows the band structure (or failure message).
    *   The layout is responsive and readable.

### Scenario 3: Manual Validation Command
*   **ID**: UAT-08-03
*   **Priority**: Medium
*   **Description**: A user wants to check an existing potential without running the full loop.
*   **Success Criteria**:
    *   Run `mlip-auto validate my_potential.yace --config config.yaml`.
    *   The system runs only the validation suite and prints a summary.

## 2. Behavior Definitions

```gherkin
Feature: Full Loop Integration

  As a user
  I want a simple command to run the complex active learning loop
  And a visual dashboard to monitor it

  Scenario: Monitoring Progress
    GIVEN a running PyAcemaker instance
    WHEN I check the "report.html"
    THEN I should see the current cycle number
    AND a graph of the validation error decreasing over time

  Scenario: Manual Check
    GIVEN I have a potential file from a colleague
    WHEN I run "mlip-auto validate"
    THEN I should get a Pass/Fail report on its physical stability
```
