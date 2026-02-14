# Cycle 04 UAT: Structure Generator & Adaptive Policy

## 1. Test Scenarios

### Scenario 01: Strain & Rattle Generation
**Priority**: High
**Description**: Verify that the `StructureGenerator` (using `RandomStrategy`) correctly generates strained and rattled structures from a seed.
**Steps**:
1.  Create a python script `test_rattle.py`.
2.  Define a perfect FCC crystal (Seed).
3.  Use `RandomStrategy` with `strain_range=0.1` and `rattle=0.1`.
4.  Generate 10 candidates.
5.  Check that cell volumes vary within ±30% and atomic positions are displaced.
**Expected Result**:
-   All 10 candidates are different from seed.
-   Volume changes are observed.

### Scenario 02: Defect Introduction
**Priority**: Medium
**Description**: Verify that `DefectStrategy` correctly introduces vacancies and interstitials into a supercell.
**Steps**:
1.  Create a python script `test_defect.py`.
2.  Define a $2 \times 2 \times 2$ supercell (Seed, $N=32$ atoms).
3.  Use `DefectStrategy` with `vacancy_concentration=1/32`.
4.  Generate a candidate.
**Expected Result**:
-   Candidate has 31 atoms.
-   Candidate cell parameters match seed (or slightly relaxed if M3GNet used, but here just geometric).

### Scenario 03: Cold Start via M3GNet (Mock)
**Priority**: Low (Optional dependency)
**Description**: Verify that the system attempts to use M3GNet for initial relaxation if configured.
**Steps**:
1.  Create a python script `test_cold_start.py`.
2.  Mock `m3gnet.models.Relaxer` (or check for `ImportError` handling).
3.  Run `M3GNetStrategy.generate(seed)`.
**Expected Result**:
-   If mocked, returns "relaxed" structure.
-   If not installed, logs warning and returns perturbed structure.

### Scenario 04: Adaptive Policy Logic
**Priority**: High
**Description**: Verify that the `AdaptivePolicy` selects the correct strategy based on input context (e.g., initial cycle vs. refinement cycle).
**Steps**:
1.  Create a python script `test_policy.py`.
2.  Create a `Context` object with `cycle=0` (Cold Start).
3.  Ask `AdaptivePolicy` for strategy. Assert it is `M3GNetStrategy` (or Random if M3GNet unavailable).
4.  Update `Context` with `cycle=1` and `uncertainty=high`.
5.  Ask `AdaptivePolicy` for strategy. Assert it is `CautiousStrategy` (e.g., small rattle).
**Expected Result**:
-   Correct strategy class returned for each context.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Generator

  Scenario: Generate Strained Candidates
    Given a perfect crystal structure
    When I request 10 random candidates with strain
    Then I should receive 10 structures with different cell volumes
    And the atomic positions should be perturbed

  Scenario: Generate Vacancy Defect
    Given a supercell with 32 atoms
    When I request a structure with 1 vacancy
    Then I should receive a structure with 31 atoms
    And the cell parameters should remain similar

  Scenario: Adaptive Policy Selection (Cold Start)
    Given the system is in Cycle 0 (Cold Start)
    When the policy engine decides on a strategy
    Then it should select the M3GNet (or Robust Random) strategy

  Scenario: Adaptive Policy Selection (High Uncertainty)
    Given the system detects high uncertainty in Cycle N
    When the policy engine decides on a strategy
    Then it should select a Cautious strategy (small perturbations) to explore the local basin
```
