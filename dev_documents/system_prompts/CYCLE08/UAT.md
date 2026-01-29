# Cycle 08 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 8.1: kMC Simulation Setup
*   **ID**: UAT-08-01
*   **Priority**: Low (Advanced)
*   **Description**: The system correctly sets up an EON kMC calculation.
*   **Steps**:
    1.  Configure `DynamicsConfig` to use kMC.
    2.  Run the Dynamics Phase.
    3.  Inspect the run directory.
    4.  Verify `config.ini` exists and `potential` is set to use the external script.

### Scenario 8.2: Full Active Learning Cycle (E2E)
*   **ID**: UAT-08-02
*   **Priority**: High
*   **Description**: A complete run-through of the system (Dry Run).
*   **Steps**:
    1.  Initialize project.
    2.  Run `mlip-auto run-loop`.
    3.  Simulate 1 cycle of: Generation -> Oracle (Mock) -> Training (Mock) -> Dynamics (Mock Halt) -> Extraction.
    4.  Verify the cycle index increments.
    5.  Verify the final report is generated.

## 2. Behavior Definitions

```gherkin
Feature: kMC and System Expansion

  Scenario: kMC Encountering Saddle Point Uncertainty
    GIVEN a kMC simulation finding a saddle point
    WHEN the uncertainty at the saddle point is high
    THEN the driver script should exit with a "Halt" code
    AND the Orchestrator should capture the saddle point structure
```
