# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 2.1: Cold Start Generation
*   **ID**: UAT-02-01
*   **Priority**: High
*   **Description**: The system must generate initial structures for a given composition when no prior data exists.
*   **Steps**:
    1.  Configure `config.yaml` with target composition "Si" and `num_candidates: 5`.
    2.  Run `mlip-auto run-loop`.
    3.  Inspect the output log. It should say "Generated 5 initial structures".
    4.  Verify the generated structures are saved (e.g., in `debug/` or state).

### Scenario 2.2: Structure Validity Check
*   **ID**: UAT-02-02
*   **Priority**: Medium
*   **Description**: Generated structures must not have overlapping atoms.
*   **Steps**:
    1.  Configure `config.yaml` with a large rattle amplitude.
    2.  Run the generator.
    3.  Load the generated structures in a notebook or viewer (ASE GUI).
    4.  Verify no atoms are closer than 0.5 Angstrom (or physically reasonable limit).

## 2. Behavior Definitions

```gherkin
Feature: Structure Generation

  Scenario: Cold Start with valid composition
    GIVEN an empty dataset
    AND a configuration for composition "Al"
    WHEN the Exploration Phase executes
    THEN it should produce a list of "Candidate" objects
    AND each candidate should contain "Al" atoms
    AND the number of candidates should match the configuration

  Scenario: Generating distorted structures
    GIVEN a base structure
    WHEN the Random Perturbation strategy is applied
    THEN the resulting structure should have different lattice constants
    AND the atomic positions should be slightly displaced
```
