# Cycle 02 UAT: Physics-Informed Generator

## 1. Test Scenarios

### Scenario 2.1: Generate SQS Alloy Structures
-   **Priority**: Critical
-   **Description**: The user needs to generate chemically disordered structures for an Fe-Ni alloy to train the model on mixing energies. This ensures that the potential is not biased towards ordered phases (like L1_0 or L1_2) if the real material is a random solid solution.
-   **Pre-conditions**:
    -   A valid config with `target_system: {elements: [Fe, Ni], composition: {Fe: 0.5, Ni: 0.5}}`.
    -   Config enables SQS: `generator: {sqs_enabled: true}`.
    -   DB is initialized and writeable.
-   **Detailed Steps**:
    1.  User executes `mlip-auto generate --n_structures 10`.
    2.  System reads the config and identifies the target composition (50/50).
    3.  System calculates the supercell size (e.g., 2x2x2 FCC -> 32 atoms).
    4.  System attempts to import `icet`. If present, it runs the Monte Carlo SQS generation.
    5.  System generates 10 unique SQS realizations (or copies if uniqueness is hard to guarantee for small cells).
    6.  System saves the structures to the DB with `config_type='sqs'`.
    7.  User inspects the DB using `mlip-auto db list`.
-   **Post-conditions**:
    -   The database contains 10 new entries.
    -   Each entry contains exactly 16 "Fe" and 16 "Ni" atoms.
    -   The metadata field `sqs` is present in the database record.
-   **Failure Modes**:
    -   Impossible stoichiometry (e.g., 50% of 31 atoms). System should round or raise error.
    -   `icet` import failure (should fallback to random).

### Scenario 2.2: Apply Elastic Strain (Equation of State)
-   **Priority**: High
-   **Description**: The user wants to cover the Pressure-Volume curve (Equation of State). The generator should produce compressed and expanded versions of the same structure. This allows the potential to learn the bulk modulus.
-   **Pre-conditions**:
    -   Config has `strain_range: [-0.10, 0.10]` (compression to expansion).
-   **Detailed Steps**:
    1.  User executes `mlip-auto generate --n_structures 100`.
    2.  System creates base structures.
    3.  For each structure, System generates a random strain tensor $\epsilon$.
    4.  System applies the transformation $v' = (I + \epsilon) v$.
    5.  System saves the result.
    6.  User queries the database for volumes: `ase db mlip.db --limit 0`.
    7.  User plots a histogram of cell volumes.
-   **Post-conditions**:
    -   The volumes are distributed roughly uniformly between $V_0 * 0.9^3$ and $V_0 * 1.1^3$.
    -   The lattice angles may deviate from 90 degrees if shear was included.
-   **Failure Modes**:
    -   Extreme strain leads to atom overlap (caught later by Surrogate).

### Scenario 2.3: Defect Generation (Vacancy)
-   **Priority**: Medium
-   **Description**: The user wants to train the potential to recognize missing atoms (Vacancies). This is crucial for diffusion studies.
-   **Pre-conditions**:
    -   Config has `defects: {vacancy_probability: 1.0, count: 1}`.
-   **Detailed Steps**:
    1.  User executes `mlip-auto generate --n_structures 5`.
    2.  System generates a 32-atom supercell.
    3.  System randomly selects 1 index to delete.
    4.  System deletes the atom.
    5.  System saves the 31-atom structure to DB.
    6.  User checks the number of atoms in each structure.
-   **Post-conditions**:
    -   Generated structures have 31 atoms (assuming 1 vacancy).
    -   `config_type` contains "vacancy".
-   **Failure Modes**:
    -   Removing too many atoms (collapsing the cell).

## 2. Behaviour Definitions

```gherkin
Feature: Structure Generation
  As a materials scientist
  I want to generate physically diverse atomic structures
  So that my machine learning model learns valid physics across the phase diagram

  Scenario: Generating a 50-50 Alloy SQS
    Given a target system of Fe and Ni with 50-50 composition
    And a supercell size of 32 atoms
    When I run the generator command with "sqs_enabled=true"
    Then the resulting structure should contain exactly 16 Fe atoms and 16 Ni atoms
    And the structure config_type should be recorded as "sqs"

  Scenario: Applying Hydrostatic Strain
    Given a base unit cell of volume 100 A^3
    And a requested strain range of +/- 10%
    When the generator applies random strain
    Then the resulting volume should be strictly between 72.9 and 133.1 A^3
    And the lattice vectors should remain orthogonal (for pure hydrostatic strain)

  Scenario: Thermal Rattling
    Given a perfect crystal structure
    And a rattle standard deviation of 0.1 Angstrom
    When the generator applies rattling
    Then the atomic positions should deviate from the perfect lattice sites
    But the cell vectors should remain unchanged (Volume is constant)

  Scenario: Database Persistence and Provenance
    Given an empty database
    When I generate 100 structures
    Then the database count should increase by 100
    And each new record should have a unique UUID
    And the generation tag should be 0 (Initial set)
```
