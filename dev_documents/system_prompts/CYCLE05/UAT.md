# Cycle 05: Dynamics Engine (MD) & Hybrid Potential - UAT

## 1. Test Scenarios

### Scenario 1: Hybrid Potential Setup
*   **ID**: UAT-05-001
*   **Objective**: Ensure LAMMPS runs with `pair_style hybrid/overlay`.
*   **Pre-conditions**: Mock or Real LAMMPS binary.
*   **Steps**:
    1.  Configure `DynamicsConfig` with `hybrid_potential="zbl"`.
    2.  Provide a `.yace` potential and an initial structure.
    3.  Call `dynamics.explore()`.
*   **Expected Result**:
    *   The `in.lammps` file contains `pair_style hybrid/overlay pace zbl`.
    *   The `log.lammps` file shows the hybrid pair style was initialized without error.

### Scenario 2: Uncertainty Watchdog Trigger
*   **ID**: UAT-05-002
*   **Objective**: Verify that MD stops when $\gamma$ exceeds the threshold.
*   **Pre-conditions**: A potential that predicts high uncertainty (or a mock that injects high $\gamma$).
*   **Steps**:
    1.  Configure `DynamicsConfig` with `uncertainty_threshold=2.0`.
    2.  Run MD on a structure that is far from the training set (e.g., highly compressed).
    3.  Check the run status.
*   **Expected Result**:
    *   The simulation stops before completing all steps.
    *   The log contains "Halted by fix halt".
    *   The `dynamics.explore()` method returns `halted=True` and the snapshot at the halt step.

### Scenario 3: Clean Exit on Convergence
*   **ID**: UAT-05-003
*   **Objective**: Verify that MD completes if uncertainty is low.
*   **Pre-conditions**: A good potential and a stable structure.
*   **Steps**:
    1.  Configure `DynamicsConfig` with `steps=1000`.
    2.  Run MD.
    3.  Check the run status.
*   **Expected Result**:
    *   The simulation runs for 1000 steps.
    *   The log contains "Loop time of ...".
    *   The `dynamics.explore()` method returns `halted=False`.

## 2. Behavior Definitions

```gherkin
Feature: Safe Molecular Dynamics

  Scenario: Running with Hybrid Potential
    Given I have a machine-learned potential (.yace)
    And I want to prevent nuclear fusion
    When I start the MD simulation
    Then the system should use a hybrid potential (ACE + ZBL)
    And the ZBL potential should be active at short distances (< 1.0 A)

  Scenario: Automatic Halt on High Uncertainty
    Given the uncertainty threshold is set to 5.0
    When the extrapolation grade ($\gamma$) of any atom exceeds 5.0
    Then the simulation should stop immediately
    And the current configuration should be saved for analysis
    And the Orchestrator should be notified of the "Halt" event
```
