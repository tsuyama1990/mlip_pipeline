# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Random Generation (Cold Start)
**Priority**: High
**Goal**: Verify that the system can generate valid random structures from a seed.
**Procedure**:
1.  Configure `config.yaml` with `generator: type: random`.
2.  Provide `seed_structure.xyz` (e.g., a simple bulk crystal).
3.  Run the generator module (standalone or via main).
4.  Inspect output structures in `candidates/`.
**Expected Result**:
*   Generated structures are valid ASE atoms (readable by `ase gui`).
*   Positions are perturbed (not identical to seed).
*   Atom counts match (or vary if scaling is used).

### Scenario 2: Defect Generation
**Priority**: Medium
**Goal**: Verify that vacancies and interstitials are created correctly.
**Procedure**:
1.  Configure `generator: type: defect`.
2.  Run generation.
3.  Inspect output.
**Expected Result**:
*   Some structures have $N-1$ atoms (Vacancy).
*   Some have $N+1$ atoms (Interstitial).
*   The system logs "Generated 5 vacancies, 5 interstitials".

### Scenario 3: Adaptive Policy
**Priority**: Low (Advanced)
**Goal**: Verify that the strategy changes over cycles.
**Procedure**:
1.  Mock the cycle number to 0, then 5.
2.  Check logs for the "Generation Strategy".
**Expected Result**:
*   Cycle 0 log: "Strategy: Random Exploration (100%)".
*   Cycle 5 log: "Strategy: Refinement (80% Defect, 20% Random)".

## 2. Behavior Definitions

```gherkin
Feature: Structure Generation

  Scenario: Generating random structures
    GIVEN a seed structure "Al.cif"
    AND a configuration "count: 10, rattle: 0.1"
    WHEN the "RandomGenerator" is executed
    THEN it should produce 10 structures
    AND the positions should differ from the seed by approx 0.1 Angstrom

  Scenario: Generating defects
    GIVEN a seed structure with 32 atoms
    WHEN the "DefectGenerator" is executed with "vacancy_mode: true"
    THEN it should produce structures with 31 atoms
```
