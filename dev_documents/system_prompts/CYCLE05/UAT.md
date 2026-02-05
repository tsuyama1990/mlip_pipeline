# CYCLE 05 UAT: Adaptive Exploration

## 1. Test Scenarios

### Scenario 05.1: Alloy Mixing (MC Policy)
*   **Priority**: High
*   **Objective**: Verify that the system tries to mix elements for alloys.
*   **Description**: Define a binary system (e.g., Fe-Pt).
*   **Success Criteria**:
    *   Logs show "Applied High-MC Policy".
    *   Generated structures show different chemical ordering (swapped atoms) compared to the initial structure.

### Scenario 05.2: Insulator Defect Sampling
*   **Priority**: Medium
*   **Objective**: Verify that the system explores non-stoichiometric space for oxides.
*   **Description**: Define an oxide system (e.g., MgO).
*   **Success Criteria**:
    *   Logs show "Applied Defect-Driven Policy".
    *   Generated structures include vacancies (Mg or O missing).

## 2. Behavior Definitions

```gherkin
Feature: Structure Generation

  Scenario: Policy Adaptation
    GIVEN a material defined as "Insulator" (Bandgap > 0)
    WHEN the Generator determines the exploration strategy
    THEN it should prioritize "Defect Engineering"
    AND generate structures with vacancies or interstitials

  Scenario: Cold Start
    GIVEN a new project with no existing data
    WHEN the cycle starts
    THEN the Generator should use a "Cold Start" strategy (e.g., Random or M3GNet)
    AND produce an initial batch of diverse structures
```
