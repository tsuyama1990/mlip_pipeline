# Cycle 05 UAT: The Active Learning Loop

## 1. Test Scenarios

### Scenario 01: The Watchdog Trigger
**Priority**: High
**Description**: Run MD and trigger a halt.
**Objective**: Verify that high uncertainty stops the simulation.

**Steps**:
1.  (Requires real LAMMPS + Pace).
2.  Run an MD simulation on a high-temperature liquid using a very sparse potential (trained on only 1 structure).
3.  **Expected Result**:
    -   The simulation does NOT reach the target 1000 steps.
    -   It stops early (e.g., at step 50).
    -   The log file says `Fix halt condition met`.
    -   The `Orchestrator` logs: `[INFO] MD Halted. Max Gamma: 12.5. Extracting structure...`

### Scenario 02: The Resume Capability
**Priority**: Medium
**Description**: Restart the loop.
**Objective**: Verify continuity.

**Steps**:
1.  Force an interrupt (Ctrl+C) during the "Training" phase of the OTF loop.
2.  Restart the application: `pyacemaker run config.yaml`.
3.  **Expected Result**:
    -   The system detects `active_learning/state.json`.
    -   It skips "Deployment" (as it was done) and resumes "Training" or the next step.
    -   It does NOT restart the MD from step 0.

### Scenario 03: Hybrid Potential Safety
**Priority**: High
**Description**: Ensure atoms don't collapse.
**Objective**: Verify ZBL/LJ baseline.

**Steps**:
1.  Create a configuration with two atoms very close ($0.5 \AA$).
2.  Run a single point energy calculation using the Dynamics Engine.
3.  **Expected Result**:
    -   The energy should be extremely high (positive), dominated by the ZBL repulsive term.
    -   If the ACE part was dominant (and untrained), it might predict a non-physical attractive well. The Hybrid setup prevents this.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: On-the-Fly Active Learning

  Scenario: Handling High Uncertainty
    Given a running MD simulation
    When the extrapolation grade "gamma" exceeds 5.0
    Then the simulation should halt immediately
    And the problematic atomic configuration should be saved
    And the Orchestrator should trigger the Refinement cycle

  Scenario: Continuous Improvement
    Given an initial potential with high error
    When the OTF loop runs for several iterations
    Then the frequency of Halts should decrease over time
    And the potential's validation metrics should improve
```
