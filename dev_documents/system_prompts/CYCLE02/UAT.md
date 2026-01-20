# Cycle 02: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 2.1: SQS Generation
-   **Priority**: High
-   **Description**: Generate a Special Quasirandom Structure for a binary alloy.
-   **Steps**:
    1.  Define composition: `{"Fe": 0.5, "Ni": 0.5}`.
    2.  Define supercell: `[2, 2, 2]` (based on fcc primitive).
    3.  Call `generate_sqs`.
-   **Success Criteria**:
    -   Result is an `ase.Atoms` object.
    -   Total atoms = $4 \times 2^3 = 32$.
    -   Fe count = 16, Ni count = 16.
    -   Symmetry is broken (spacegroup is P1 or low symmetry).

### Scenario 2.2: Equation of State (EOS) Dataset Creation
-   **Priority**: Medium
-   **Description**: Create a volume-strain dataset for EOS fitting.
-   **Steps**:
    1.  Take an Al fcc primitive cell.
    2.  Apply strains: `[-0.10, -0.05, 0.00, +0.05, +0.10]`.
    3.  Collect resulting 5 structures.
-   **Success Criteria**:
    -   5 structures returned.
    -   Volumes follow the expected trend.
    -   Metadata `config_type` clearly indicates the strain level.

### Scenario 2.3: Rattling & Minimum Distance Check
-   **Priority**: Low
-   **Description**: Verify that heavy rattling acts as expected but doesn't create fusion.
-   **Steps**:
    1.  Take a dense structure.
    2.  Apply large rattle ($\sigma=0.3\text{\AA}$).
    3.  Check nearest neighbor distances.
-   **Success Criteria**:
    -   Positions are significantly displaced from grid.
    -   **Constraint**: No two atoms are closer than $0.5 \times$ bond_length (sanity check).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Generation

  Scenario: Generating a compositionally accurate SQS
    Given a target composition of "Fe:0.75, Ni:0.25"
    And a target supercell size of 32 atoms
    When the SQS generator runs
    Then the resulting structure should have 24 Iron atoms
    And the resulting structure should have 8 Nickel atoms
    And the structure should have valid periodic boundary conditions

  Scenario: Applying Hydrostatic Strain
    Given an equilibrium structure with volume V0
    When I apply a hydrostatic strain of +5%
    Then the new volume should be approximately 1.157 * V0
    And the cell angles should remain unchanged
```
