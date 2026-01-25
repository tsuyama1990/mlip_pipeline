# Cycle 07 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 7.1: Vacancy Generation
*   **Priority:** Medium
*   **Description:** The system generates symmetry-irreducible vacancy structures.
*   **Input:** BCC Iron primitive cell.
*   **Expected Output:**
    *   A single unique supercell with one atom missing.
    *   The supercell size respects the configuration (e.g., 50+ atoms).

### Scenario 7.2: Interstitial Generation
*   **Priority:** Medium
*   **Description:** The system inserts an atom into a valid interstitial site.
*   **Input:** FCC Aluminum.
*   **Expected Output:**
    *   Structures with one extra atom.
    *   No atom overlaps (distance > threshold).

### Scenario 7.3: Adaptive Policy
*   **Priority:** High
*   **Description:** The policy engine changes its recommendation based on input parameters.
*   **Input 1:** Material tagged as "Insulator".
*   **Input 2:** Material tagged as "Metal".
*   **Expected Output 1:** Recommendation includes "Charge States" (if supported) or "High Defect Density".
*   **Expected Output 2:** Recommendation includes "Melting Point Search".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Generation

  Scenario: Generate Defects
    Given a primitive crystal structure
    When I request vacancy generation
    Then I should receive a list of unique supercells
    And each supercell should have one missing atom

  Scenario: Policy Decision
    Given the workflow is in the early stages (Cycle 1)
    When I ask the policy for a strategy
    Then it should recommend "EOS/Strain" exploration
    But if the workflow is in late stages (Cycle 8)
    Then it should recommend "Defects" or "Melting"
```
