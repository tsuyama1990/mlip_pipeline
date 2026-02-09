# Cycle 02 UAT: Structure Generator

## 1. Test Scenarios

### Scenario 2.1: Basic Structure Generation (Cold Start)
*   **Goal:** Verify that the system can generate initial structures without a prior potential.
*   **Steps:**
    1.  Create a `config.yaml` with `generator: { mode: "COLD_START", composition: "MgO" }`.
    2.  Run the Orchestrator with this config.
    3.  Inspect the output list of `Structure` objects.
*   **Expected Behavior:**
    *   10-20 structures generated (e.g., perturbed FCC/BCC/HCP).
    *   Metadata tags contain `provenance: COLD_START`.
    *   Positions are valid (no overlap < 1.0 Å).

### Scenario 2.2: Defect Injection (Vacancy)
*   **Goal:** Verify that the system correctly introduces vacancies into a supercell.
*   **Steps:**
    1.  Create a `config.yaml` with `generator: { defects: { vacancy: 0.05 } }` (5% vacancy).
    2.  Run the Generator.
    3.  Inspect the output structures.
*   **Expected Behavior:**
    *   Output structures have fewer atoms than the pristine supercell.
    *   Example: A 2x2x2 MgO supercell (64 atoms) should have ~3 atoms removed (61 atoms).
    *   Positions remain valid (no collapse).

### Scenario 2.3: Adaptive Policy Switching
*   **Goal:** Verify that the Policy Engine changes strategy based on material properties.
*   **Steps:**
    1.  Mock a `MaterialFeatures` input with `band_gap = 0.0` (Metal).
    2.  Check the `SamplingConfig` output.
    3.  Mock a `MaterialFeatures` input with `band_gap = 5.0` (Insulator).
    4.  Check the `SamplingConfig` output.
*   **Expected Behavior:**
    *   Metal case: `mc_swap_ratio` should be > 0 (e.g., 0.5).
    *   Insulator case: `mc_swap_ratio` should be 0.0.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Generation

  Scenario: Generate initial random structures
    Given a target composition "MgO"
    And a generation mode "COLD_START"
    When I request 10 structures
    Then I should receive 10 valid atomic configurations
    And each structure should have valid cell dimensions
    And the minimum interatomic distance should be > 1.0 Å

  Scenario: Inject vacancies
    Given a perfect supercell of 64 atoms
    When I request a vacancy concentration of 0.05
    Then the resulting structure should have approximately 61 atoms
    And the structure should retain its crystal symmetry (mostly)

  Scenario: Apply strain
    Given a cubic unit cell
    When I apply a strain of 0.1 (10%)
    Then the cell volume should increase by approximately 33%
    And the atom fractional coordinates should remain constant
```
