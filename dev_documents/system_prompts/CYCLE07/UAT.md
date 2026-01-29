# User Acceptance Testing (UAT): Cycle 07

## 1. Test Scenarios

Cycle 07 makes the robot smart.

### Scenario 7.1: Defect Sampling
-   **ID**: UAT-C07-01
-   **Priority**: High
-   **Description**: We want to ensure the potential learns about vacancies.
-   **Success Criteria**:
    -   Configure the system to use `DefectStrategy`.
    -   Run the Exploration phase.
    -   The generated structures should contain vacancies (N-1 atoms).
    -   These structures are sent to DFT and Training.

### Scenario 7.2: Adaptive Policy (Metal vs Ceramic)
-   **ID**: UAT-C07-02
-   **Priority**: Medium
-   **Description**: Different materials need different exploration.
-   **Success Criteria**:
    -   **Case A (Alloy)**: Set config to `is_metal: true`. Policy suggests `HybridMDMC` (Atom swap).
    -   **Case B (Oxide)**: Set config to `is_metal: false`. Policy suggests `VariableT_MD` (Temperature ramping).

### Scenario 7.3: High-Pressure Exploration
-   **ID**: UAT-C07-03
-   **Priority**: Low
-   **Description**: Explore dense phases.
-   **Success Criteria**:
    -   Policy schedules a pressure ramp (0 -> 10 GPa).
    -   LAMMPS runs NPT with increasing pressure.
    -   Resulting structures have smaller volumes.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Adaptive Exploration Policy

  Scenario: Policy chooses defects for insulators
    Given the material is "SiO2" (Insulator)
    When the policy is queried
    Then it should recommend "DefectSampling"
    And it should recommend "HighTemperatureMD"

  Scenario: Policy chooses swapping for alloys
    Given the material is "NiAl" (Alloy)
    When the policy is queried
    Then it should recommend "AtomSwap MonteCarlo"
    And the swap probability should be > 0
```
