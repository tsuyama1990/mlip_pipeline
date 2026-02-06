# Cycle 07 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 7.1: The Watchdog Trigger
**Priority**: Critical
**Description**: Verify that the system stops the simulation when it encounters unknown structures.
**Steps**:
1.  Configure `uncertainty_threshold = 2.0`.
2.  Run MD on a system known to be unstable/unknown (e.g., high T liquid).
3.  **Expectation**:
    *   LAMMPS stops early (before `n_steps`).
    *   Log message: "Simulation halted due to high uncertainty (gamma=2.5)".
    *   The structure causing the halt is added to the candidate list.

### Scenario 7.2: Adaptive Temperature Ramp
**Priority**: High
**Description**: Verify that the system automatically increases difficulty as it learns.
**Steps**:
1.  Run Cycle 1 (High uncertainty).
2.  **Check**: Cycle 2 config uses Low Temperature (e.g., 300K).
3.  Run Cycle 2 (Low uncertainty).
4.  **Check**: Cycle 3 config uses High Temperature (e.g., 600K).
5.  **Check**: The Policy Engine logs "Increasing temperature due to stable performance."

### Scenario 7.3: Recovery from Divergence
**Priority**: Medium
**Description**: Verify that the system recovers if a potential explodes.
**Steps**:
1.  Simulate a cycle where the Validator reports `passed=False` (RMSE spike).
2.  **Expectation**:
    *   The Orchestrator does *not* deploy this bad potential.
    *   It triggers a "Recovery Strategy" (e.g., reverts to previous potential, adds more robust data (ZBL), and retrains).
    *   The loop continues instead of crashing.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Robust Exploration

  Scenario: Halting on Uncertainty
    Given an uncertainty threshold of 5.0
    When the simulation encounters a structure with gamma = 6.0
    Then the simulation should stop immediately
    And the high-uncertainty structure should be saved for labeling

  Scenario: Adaptive Policy
    Given a model that failed validation in the previous cycle
    When the policy engine generates the next configuration
    Then it should select more conservative parameters (lower T)
```
