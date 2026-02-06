# Cycle 07 UAT: The Long Haul

## 1. Test Scenarios

### Scenario 7: Running kMC
**Priority**: Medium
**Objective**: Verify that the system can execute a kMC simulation to find saddle points.

**Steps**:
1.  **Preparation**:
    *   Config: `explorer.type = "eon"`.
    *   Mock: Ensure `eonclient` is available (or mocked).
2.  **Execution**:
    *   Run the pipeline.
3.  **Verification**:
    *   Check that `config.ini` was created.
    *   Check that `pace_driver.py` was copied.
    *   Check logs: "EON simulation started."

## 2. Behavior Definitions

```gherkin
Feature: kMC Integration

  Scenario: Detecting uncertainty in saddle point search
    GIVEN an EON simulation searching for a transition state
    WHEN the "pace_driver.py" calculates a gamma value above threshold
    THEN the driver should exit with code 100
    AND the EonClient should report a "Halt" to the Orchestrator
```
