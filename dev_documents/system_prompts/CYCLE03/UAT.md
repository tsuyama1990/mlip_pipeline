# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 3.1: Successful DFT Calculation
*   **ID**: UAT-03-01
*   **Priority**: High
*   **Description**: The system runs a DFT calculation for a valid structure and retrieves results.
*   **Steps**:
    1.  Provide a simple structure (e.g., Al atom) in the candidate list.
    2.  Configure `DFTConfig` to point to a mock QE script (or actual QE if available).
    3.  Run the loop (Oracle Phase).
    4.  Verify that `DFTResult` is stored with valid energy and forces.

### Scenario 3.2: Self-Correction of Convergence Error
*   **ID**: UAT-03-02
*   **Priority**: Medium
*   **Description**: The system detects a convergence failure and retries with adjusted parameters.
*   **Steps**:
    1.  Configure a mock QE script that fails the first time with "convergence not achieved" but succeeds the second time.
    2.  Run the loop.
    3.  Verify logs show "Convergence error detected. Retrying with mixing_beta=0.3".
    4.  Verify the final status is `converged=True`.

## 2. Behavior Definitions

```gherkin
Feature: DFT Oracle

  Scenario: Running a standard calculation
    GIVEN a candidate structure
    AND a valid QE configuration
    WHEN the Oracle processes the structure
    THEN it should generate a standard input file
    AND execute the QE command
    AND parse the output to extract Energy, Forces, and Stress

  Scenario: Handling convergence failure
    GIVEN a calculation that fails to converge
    WHEN the Error Handler analyzes the log
    THEN it should identify the error type
    AND propose a new set of parameters (e.g., reduced mixing beta)
    AND the Runner should retry with the new parameters
```
