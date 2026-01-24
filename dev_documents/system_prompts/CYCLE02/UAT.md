# Cycle 02 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario ID: UAT-C02-001 - "The Cold Start": Generating an Initial Dataset for Aluminum

**Priority:** High
**Description:**
This scenario validates the core value proposition of Cycle 02: the ability to generate a diverse, physically relevant dataset from scratch. The user acts as a materials scientist starting a new project on Aluminum. They want to generate a "Cold Start" dataset containing strained cells, thermal snapshots, and vacancies to train the initial version of the potential. This test ensures that the `Generator` module works as expected and integrates with the `Database` (mocked or real).

**User Story:**
As a Computational Materials Scientist, I want to automatically generate 100+ diverse configurations of Aluminum, including expanded/compressed cells and vacancy defects, so that I can start training my MLIP without having to manually construct each geometry in a GUI tool like VESTA. I expect the system to handle the supercell creation and randomisation deterministically.

**Step-by-Step Walkthrough:**
1.  **Configuration**: The user modifies `input.yaml` to target "Al" (FCC) and enables the generator module.
    -   `target_system`: Al, fcc, a=4.05.
    -   `generator.supercell`: [3, 3, 3] (108 atoms).
    -   `generator.strain_variants`: 10.
    -   `generator.rattle_variants`: 10.
    -   `generator.include_defects`: True.
2.  **Execution**: The user runs `mlip-auto generate`.
    -   *Expectation*: The CLI displays a progress bar: "Generating structures...".
3.  **Output Verification (CLI)**: The CLI reports: "Successfully generated 52 structures (1 Base + 10 Strain + 10 Rattle + 1 Vacancy + 30 Combinations)." (Exact number depends on logic).
4.  **Visual Verification**: The user opens the database (or exported .xyz file) in a visualiser (Ovito).
    -   *Action*: They inspect a "Rattle" structure.
    -   *Observation*: Atoms are slightly displaced from perfect lattice sites.
    -   *Action*: They inspect a "Vacancy" structure.
    -   *Observation*: One atom is missing from the 108-atom grid.
    -   *Action*: They inspect a "Strain" structure.
    -   *Observation*: The cubic box is now slightly triclinic or rectangular.
5.  **Clash Check**: The user runs a provided utility script to check minimal distances.
    -   *Expectation*: No interatomic distance is less than 2.0 Angstroms (Al-Al bond is ~2.86). This confirms the generator didn't create "nuclear fusion" configurations.

**Success Criteria:**
-   The generator runs without crashing.
-   The output structures span the requested parameter space (strain $\pm 5\%$, rattle $\sigma=0.1$).
-   Defects are correctly identified (107 atoms instead of 108).
-   All structures are tagged with correct `config_type` metadata.

### Scenario ID: UAT-C02-002 - Deterministic Reproduction of Datasets

**Priority:** Medium
**Description:**
Science requires reproducibility. This scenario ensures that if a user shares their `input.yaml` and `seed` with a colleague, the colleague generates the *exact same* atomic structures. This is critical for debugging and for validating results in publications.

**User Story:**
As a Researcher publishing a paper, I want to ensure that my dataset generation is deterministic. If I re-run the generation command with the same random seed, I must get bitwise-identical coordinates. This allows me to distribute my methodology without distributing gigabytes of structure files.

**Step-by-Step Walkthrough:**
1.  **Run A**: The user sets `seed: 12345` in `input.yaml` and runs `mlip-auto generate`.
2.  **Snapshot A**: The user exports the result to `run_a.xyz`.
3.  **Run B**: The user clears the database, ensures the config is identical, and runs `mlip-auto generate` again.
4.  **Snapshot B**: The user exports the result to `run_b.xyz`.
5.  **Comparison**: The user runs `diff run_a.xyz run_b.xyz`.
    -   *Expectation*: The diff returns nothing (or only timestamp differences in headers). The coordinates and cell vectors must be identical to the last decimal place.
6.  **Run C (Control)**: The user changes `seed: 67890` and runs `mlip-auto generate`.
    -   *Expectation*: The resulting structures are statistically similar (same distribution) but the specific coordinates are different.

**Success Criteria:**
-   Run A and Run B produce identical outputs.
-   Run C produces different outputs.
-   This validates the correct propagation of the Random Number Generator (RNG) state throughout the `StructureBuilder`, `StrainGenerator`, and `RattleGenerator` classes.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Physics-Informed Structure Generation
  As a User
  I want to generate diverse atomic configurations
  So that I can train a robust machine learning potential

  Background:
    Given I have a configured "Al" system
    And the "StructureBuilder" is initialized with a seed "42"

  Scenario: Generate Rattle Variants
    When I request 5 rattle variants with sigma=0.1
    Then I should receive 5 distinct Atoms objects
    And each object should have the same cell dimensions as the primitive
    And the maximum displacement of any atom should be approx 0.3 Angstroms (3 sigma)
    And the `config_type` metadata should start with "rattle"

  Scenario: Generate Strain Variants
    When I request a strain variant with 5% expansion
    Then the volume of the new cell should be roughly 1.15 times the original volume
    And the fractional coordinates of atoms should remain constant
    And the `config_type` metadata should be "strain_vol_expansion"

  Scenario: Generate Vacancy Defect
    Given a supercell with 32 atoms
    When I request vacancy generation
    Then I should receive a list of structures
    And each structure should have exactly 31 atoms
    And the structure should preserve the cell dimensions of the supercell

  Scenario: Prevent Atomic Clashes
    Given I request a high-amplitude rattle (sigma=1.0)
    And I have enabled the "minimal_distance_filter" with cutoff 1.5A
    When the generator produces a structure with atoms closer than 1.0A
    Then that structure should be discarded or regenerated
    And the final list should only contain physically reasonable structures
```
