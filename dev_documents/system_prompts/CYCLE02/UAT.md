# Cycle 02: Structure Generator & Adaptive Policy - UAT

## 1. Test Scenarios

### Scenario 1: Generate Basic Distorted Structures
*   **ID**: UAT-02-001
*   **Objective**: Ensure the generator can create valid, randomly distorted structures.
*   **Pre-conditions**: Cycle 01 complete.
*   **Steps**:
    1.  Configure `StructureGenerator` with `rattle=0.1`.
    2.  Generate 50 MgO unit cells.
    3.  Check the resulting `ASE.Atoms` objects.
*   **Expected Result**: All structures are valid (no NaN coordinates). The average atomic displacement is ~0.1 Angstrom from equilibrium.

### Scenario 2: Apply Volume Strains (EOS)
*   **ID**: UAT-02-002
*   **Objective**: Ensure the generator can create a series of expanded/compressed structures for Equation of State fitting.
*   **Pre-conditions**: Cycle 01 complete.
*   **Steps**:
    1.  Configure `StructureGenerator` with `strain_range=0.1`.
    2.  Generate 10 structures.
    3.  Calculate the volume of each structure.
*   **Expected Result**: Volumes range from 0.73 * V0 to 1.33 * V0 (approx).

### Scenario 3: Introduce Point Defects
*   **ID**: UAT-02-003
*   **Objective**: Ensure the generator can create vacancies and interstitials.
*   **Pre-conditions**: Cycle 01 complete.
*   **Steps**:
    1.  Configure `StructureGenerator` with `defect_density=0.01`.
    2.  Generate a 3x3x3 supercell of MgO (216 atoms).
    3.  Count the atoms in the output.
*   **Expected Result**: Some structures have < 216 atoms (vacancies) or > 216 atoms (interstitials).

## 2. Behavior Definitions

```gherkin
Feature: Adaptive Structure Generation

  Scenario: Generating structures based on uncertainty
    Given the current uncertainty is "High" (> 0.5 extrapolation grade)
    When I request 20 structures from the generator
    Then the AdaptiveExplorationPolicy should favor "Low Temperature MD"
    And the generated structures should be close to equilibrium
    And the displacement magnitude should be small (< 0.05 A)

  Scenario: Generating structures based on material type (Metal)
    Given the target material is "FePt" (Metal)
    When I request 20 structures from the generator
    Then the AdaptiveExplorationPolicy should favor "High Temperature MC"
    And the generated structures should include chemical disorder (swaps)
    And the atomic species should be mixed

  Scenario: Generating structures for EOS fitting
    Given the generator mode is "EOS"
    When I request 11 structures
    Then the AdaptiveExplorationPolicy should set "strain_range" to 0.15
    And the generated structures should have volumes linearly spaced between -15% and +15%
```
