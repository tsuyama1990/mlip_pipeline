# Cycle 07 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Defect Generation
*   **ID**: UAT-07-01
*   **Priority**: High
*   **Description**: Verify that the system can automatically create training data for point defects.
*   **Success Criteria**:
    *   Input: Perfect crystal.
    *   Action: Run `DefectStrategy`.
    *   Output: A list of structures including single vacancies and interstitials.

### Scenario 2: EON Driver & Uncertainty Halt
*   **ID**: UAT-07-02
*   **Priority**: Critical
*   **Description**: Verify that EON stops when the potential becomes uncertain during a saddle point search.
*   **Success Criteria**:
    *   Start EON exploration.
    *   The potential driver calculates $\gamma$.
    *   Mock the driver to see a high $\gamma$.
    *   Driver exits with code 100.
    *   EON stops.
    *   Orchestrator catches the halt and extracts the structure.

### Scenario 3: Adaptive Policy Switching
*   **ID**: UAT-07-03
*   **Priority**: Medium
*   **Description**: Verify that the system changes strategy based on the cycle count or material state.
*   **Success Criteria**:
    *   Cycle 1: System chooses "Strain + Defect" generation.
    *   Cycle 5 (Low T): System chooses "EON".
    *   Cycle 5 (High T): System chooses "LAMMPS MD".

## 2. Behavior Definitions

```gherkin
Feature: Advanced Exploration

  As a researcher
  I want the system to find rare events and defects automatically
  So that the potential is robust for real-world materials problems

  Scenario: Learning Diffusion Barriers
    GIVEN a trained potential
    WHEN I run the EON exploration module
    THEN it should search for saddle points (transition states)
    AND if the barrier is uncertain, it should trigger a retraining loop

  Scenario: Defect Coverage
    GIVEN a new element pair
    WHEN the system initializes (Cycle 1-2)
    THEN it should automatically generate vacancy and interstitial structures
    AND add them to the training set to ensure core repulsion and defect physics
```
