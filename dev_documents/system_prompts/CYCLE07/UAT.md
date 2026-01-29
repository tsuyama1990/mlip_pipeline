# Cycle 07 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 7.1: Policy Decision for Metals
*   **ID**: UAT-07-01
*   **Priority**: Medium
*   **Description**: The system selects MC-heavy sampling for metallic systems.
*   **Steps**:
    1.  Initialize a project with a metallic composition (e.g., Cu-Au).
    2.  Mock the property predictor to return Band Gap = 0.
    3.  Run the Exploration Phase.
    4.  Verify logs show "Metal detected: Activating High-MC Policy".

### Scenario 7.2: Defect Generation
*   **ID**: UAT-07-02
*   **Priority**: Medium
*   **Description**: The system introduces defects when requested by the policy.
*   **Steps**:
    1.  Force the policy to "Defect-Driven" mode.
    2.  Run generation.
    3.  Inspect generated structures.
    4.  Verify that some structures contain vacancies (fewer atoms than perfect supercell).

## 2. Behavior Definitions

```gherkin
Feature: Adaptive Exploration

  Scenario: Adapting to material type
    GIVEN a material identified as an insulator
    WHEN the Adaptive Policy runs
    THEN it should prioritize "Defect" and "Strain" strategies
    AND reduce the probability of random atom swaps (MC)

  Scenario: Adapting to uncertainty
    GIVEN a workflow state with high average uncertainty
    WHEN the Adaptive Policy runs
    THEN it should select "Cautious Exploration"
    AND reduce the maximum simulation temperature
```
