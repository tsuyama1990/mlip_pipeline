# Cycle 06: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 01: kMC Exploration
*   **ID**: UAT-CY06-01
*   **Priority**: Medium
*   **Description**: Verify that the system can use Kinetic Monte Carlo to find rare events and saddle points.
*   **Pre-conditions**: A valid potential. EON installed.
*   **Steps**:
    1.  Config: `dynamics_engine: eon`.
    2.  Run `mlip-auto run --config kmc_config.yaml`.
*   **Expected Result**:
    *   System starts EON client.
    *   Logs show "Saddle point search initiated".
    *   System halts if a transition state has high uncertainty.

### Scenario 02: Adaptive Policy Behavior
*   **ID**: UAT-CY06-02
*   **Priority**: High
*   **Description**: Verify that the system changes its strategy based on feedback.
*   **Steps**:
    1.  Run the system with a mock that always fails Phonon validation.
    2.  Observe the logs for the next cycle.
*   **Expected Result**:
    *   The system logs "Phonon instability detected. Switching policy to Low-Temperature Sampling".
    *   The generated MD tasks have `temperature` lower than previous cycles.

### Scenario 03: The "Zero-Config" Experience
*   **ID**: UAT-CY06-03
*   **Priority**: Critical
*   **Description**: The ultimate acceptance test. A user provides ONLY the composition.
*   **Steps**:
    1.  Create `simple.yaml`:
        ```yaml
        system:
          elements: ["Al"]
        ```
    2.  Run `mlip-auto start simple.yaml`.
*   **Expected Result**:
    *   The system infers reasonable defaults (e.g., cutoff=5.0, max_cycles=10).
    *   The workflow starts without asking for more input.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Adaptive Autonomous Learning

  Scenario: Policy switching on Validation Failure
    GIVEN a completed training cycle
    WHEN the validation step fails due to 'Imaginary Phonons'
    THEN the Orchestrator should query the Policy engine
    AND the Policy engine should recommend 'Low Temperature MD'
    AND the next Exploration phase should use T=300K instead of T=2000K

  Scenario: kMC-driven Active Learning
    GIVEN a stable potential for solids
    WHEN kMC is used for exploration
    AND a high-energy saddle point is encountered with high uncertainty
    THEN the system should Halt
    AND the saddle point structure should be added to the training set
```
