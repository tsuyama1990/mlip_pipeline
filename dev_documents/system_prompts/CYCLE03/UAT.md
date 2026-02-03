# Cycle 03 UAT: The Structure Generator

## 1. Test Scenarios

### Scenario 01: Creating the "Rattled" Dataset
**Priority**: High
**Description**: Use the Explorer to generate a vibrational dataset.
**Objective**: Verify RandomDisplacement strategy.

**Steps**:
1.  Create a script `test_gen.py`.
2.  Load a bulk Silicon structure.
3.  Initialize `AdaptiveExplorer` with `strategy="random_displacement"` and `magnitude=0.1`.
4.  Generate 100 structures.
5.  **Expected Result**:
    -   A file `candidates.xyz` is created containing 100 frames.
    -   Visual inspection (or code check) shows atoms are slightly off equilibrium positions.
    -   The cell dimensions remain constant.

### Scenario 02: Creating the "Strained" Dataset (EOS)
**Priority**: High
**Description**: Use the Explorer to generate volume expansions/compressions.
**Objective**: Verify Strain strategy for EOS training.

**Steps**:
1.  Initialize `AdaptiveExplorer` with `strategy="strain"` and `range=[-0.1, 0.1]`.
2.  Generate 10 structures.
3.  **Expected Result**:
    -   The volumes of the generated structures vary from 0.9V0 to 1.1V0.
    -   The atomic fractional coordinates remain constant (if affine deformation).

### Scenario 03: Metadata Tracking
**Priority**: Medium
**Description**: Trace the lineage of a structure.
**Objective**: Ensure reproducibility.

**Steps**:
1.  Generate a structure.
2.  Inspect `atoms.info`.
3.  **Expected Result**:
    -   `atoms.info['generation_method']` == "StrainGenerator" (or similar).
    -   `atoms.info['source_id']` is present.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Generation

  Scenario: Generating Vibrational Data
    Given a perfect crystal structure
    When the Explorer applies "RandomDisplacement" with sigma=0.1
    Then the atom positions should deviate from the lattice sites
    And the lattice vectors should remain unchanged

  Scenario: Generating Strain Data
    Given a perfect crystal structure
    When the Explorer applies "UniformStrain" of +10%
    Then the cell volume should increase by approximately 33%
    And the symmetry of the cell should be preserved

  Scenario: Defect Generation
    Given a supercell of 64 atoms
    When the Explorer applies "VacancyGenerator"
    Then the resulting structure should have 63 atoms
```
