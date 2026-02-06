# Cycle 06 UAT: The Rescue Mission

## 1. Test Scenarios

### Scenario 6: Handling High Uncertainty
**Priority**: Critical
**Objective**: Verify that the system correctly interrupts the simulation when uncertainty spikes and triggers the retraining workflow.

**Steps**:
1.  **Preparation**:
    *   Config: `uncertainty_threshold = 5.0`.
    *   Use `MockExplorer` configured to return `halted=True` on the first run.
2.  **Execution**:
    *   Run the pipeline.
3.  **Verification**:
    *   Logs should show: "Simulation halted due to high uncertainty."
    *   Logs should show: "Extracting structure..."
    *   Logs should show: "Retraining potential..."
    *   Logs should show: "Resuming simulation..."

## 2. Behavior Definitions

```gherkin
Feature: Active Learning Interrupt

  Scenario: Uncertainty exceeds threshold
    GIVEN a running MD simulation
    WHEN the extrapolation grade (gamma) exceeds the configured threshold
    THEN the simulation must stop immediately
    AND the Orchestrator must be notified of the halt
    AND the failing structure must be added to the labeling queue
```
