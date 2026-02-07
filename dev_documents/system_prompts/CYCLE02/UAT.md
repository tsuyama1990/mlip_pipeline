# Cycle 02 UAT: Structure Generator

## 1. Test Scenarios

### SCENARIO 01: Random Structure Generation
**Priority**: High
**Description**: Verify that the generator can create valid randomized structures from a bulk template.
**Pre-conditions**: Configured `StructureGenerator` for MgO.
**Steps**:
1.  Initialise `StructureGenerator` with MgO rock-salt settings.
2.  Call `generate(n=5)`.
3.  Inspect the output structures.
**Expected Result**:
-   5 distinct structures are returned.
-   All have the correct stoichiometry (Mg:O = 1:1).
-   Positions are perturbed (not perfect crystal).
-   Cell parameters are within expected range.

### SCENARIO 02: Adaptive Policy Switching
**Priority**: Medium
**Description**: Verify that the generation strategy changes based on the policy.
**Pre-conditions**: `AdaptiveExplorationPolicy` configured.
**Steps**:
1.  Mock the "current iteration" to 0. Call generator -> Expect "Random Rattling".
2.  Mock the "current iteration" to 10. Call generator -> Expect "Mutation" (or configured strategy).
**Expected Result**: The generator logs the switch in strategy and produces structures with different characteristics (e.g., larger displacements or defects).

## 2. Behaviour Definitions

```gherkin
Feature: Structure Generation

  Scenario: Generating initial random structures
    Given a generator configuration for "Al" fcc bulk
    When I request 10 structures
    Then I should receive a list of 10 Structure objects
    And each structure should have valid periodic boundary conditions
    And the minimum interatomic distance should be greater than 1.5 Angstroms (sanity check)

  Scenario: Consistency with ASE
    Given a generated Structure object
    When I convert it to an ASE Atoms object
    Then the positions and cell vectors should be identical
    And the chemical symbols should match
```
