# CYCLE 04 UAT: MD & Uncertainty Watchdog

## 1. Test Scenarios

### Scenario 04.1: Watchdog Triggering
*   **Priority**: Critical
*   **Objective**: Verify that the simulation stops when the potential enters an extrapolation region.
*   **Description**: Run MD with a deliberately poor potential (or low threshold).
*   **Input**:
    *   `uncertainty_threshold: 0.01` (Force trigger)
*   **Success Criteria**:
    *   Simulation terminates before `steps` is reached.
    *   Logs indicate "Halt triggered by uncertainty watchdog".
    *   A structure file `bad_structure.xyz` is returned/saved.

### Scenario 04.2: Closed-Loop Active Learning
*   **Priority**: Critical
*   **Objective**: Verify the full cycle: MD -> Halt -> DFT -> Retrain -> Resume.
*   **Description**: Start with an empty dataset. Run the pipeline.
*   **Success Criteria**:
    *   Initial MD halts.
    *   Oracle adds new data.
    *   Trainer produces `generation_001.yace`.
    *   Subsequent MD runs longer or completes (if stability improves).

## 2. Behavior Definitions

```gherkin
Feature: Dynamics Engine

  Scenario: Uncertainty Halt
    GIVEN a running MD simulation
    WHEN the extrapolation grade (gamma) of any atom exceeds the threshold
    THEN the simulation should immediately stop
    AND the current atomic configuration should be saved for analysis

  Scenario: Hybrid Potential Safety
    GIVEN an MD simulation config
    WHEN the input script is generated
    THEN it must use "pair_style hybrid/overlay"
    AND include a physical baseline (LJ/ZBL) to prevent atomic overlap
```
