# Cycle 03 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario ID: UAT-C03-001 - The "Garbage Collector" Test (Filtering Bad Structures)

**Priority:** High
**Description:**
The generator (Module A) is sometimes too enthusiastic, creating structures with atoms dangerously close together (e.g., during aggressive random rattling or high-density liquid simulation). If these structures reach Quantum Espresso, they will cause SCF convergence failures or "segmentation faults", wasting hours of compute time. This scenario verifies that the Surrogate Module effectively identifies and discards these "garbage" structures based on high MACE predicted forces/energies.

**User Story:**
As a System Operator, I want to ensure that "exploded" structures never enter the DFT queue. I want the Surrogate model to quickly scan thousands of candidates and automatically discard those with unphysical energies (>10 eV/atom above hull) or massive forces, so that my DFT resources are focused on solvable problems.

**Step-by-Step Walkthrough:**
1.  **Preparation**: The user manually creates a "Bad" structure (two atoms 0.5 Å apart) and a "Good" structure (equilibrium bond distance).
2.  **Configuration**: The user sets `surrogate.energy_threshold: 5.0` and `surrogate.force_threshold: 100.0`.
3.  **Execution**: The user runs `mlip-auto filter --input candidate_list.xyz`.
    -   *Expectation*: The system loads the MACE model (or a mock that simulates high energy for short bonds).
4.  **Result**: The CLI reports: "Processed 2 structures. 1 discarded (High Energy/Force). 1 retained."
5.  **Verification**: The user inspects the output file. Only the "Good" structure remains.

**Success Criteria:**
-   The "Bad" structure is rejected.
-   The "Good" structure is kept.
-   The rejection reason is logged (e.g., "Energy 50.2 eV > Threshold 5.0 eV").

### Scenario ID: UAT-C03-002 - The "Diversity" Test (Farthest Point Sampling)

**Priority:** Medium
**Description:**
Generating 1,000 structures that are nearly identical is useless for machine learning. We need diversity. This test verifies that the FPS algorithm successfully selects a diverse subset from a larger pool of similar candidates.

**User Story:**
As a Data Scientist, I want to downsample my dataset from 10,000 generated structures to 1,000 for DFT calculation. I expect the system to pick the 1,000 most distinct structures, covering the edges of the phase space (e.g., extreme strains, unique defect configurations) rather than just picking the first 1,000 or a random subset.

**Step-by-Step Walkthrough:**
1.  **Preparation**: The user generates a "Cluster" of 90 very similar structures (small rattle) and 10 "Outliers" (large strain/different crystal phase). Total = 100.
2.  **Configuration**: The user sets `surrogate.selection_method: fps` and `surrogate.selection_ratio: 0.2` (Select 20 structures).
3.  **Execution**: The user runs `mlip-auto filter --input cluster_and_outliers.xyz`.
4.  **Result**: The system calculates descriptors and selects 20 points.
5.  **Analysis**: The user checks which structures were selected.
    -   *Expectation*: All 10 "Outliers" should be selected because they are far from the cluster in descriptor space.
    -   *Expectation*: The remaining 10 selections should be from the "Cluster", spread out as much as possible.
    -   *Contrast*: If random sampling were used, only ~2 outliers would statistically be selected. FPS guarantees the outliers are prioritised.

**Success Criteria:**
-   The selection includes the structurally distinct outliers.
-   The visual distribution of the selected set (e.g., on a PCA plot of descriptors) is more uniform than the original set.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Surrogate Model Integration
  As a Pipeline Manager
  I want to screen candidate structures using a fast ML model
  So that I only spend DFT resources on high-value targets

  Background:
    Given the MACE model is loaded (or mocked)
    And I have a list of candidate atoms

  Scenario: Filter High Energy Structures
    Given a candidate structure with predicted energy 20.0 eV/atom
    And the energy threshold is set to 5.0 eV/atom
    When I run the surrogate filter
    Then the structure should be marked as "rejected"
    And it should not be passed to the FPS stage

  Scenario: Calculate Descriptors
    Given a list of valid structures
    When I request descriptor calculation
    Then I should receive a numpy array of shape (N_structures, N_features)
    And the values should be deterministic for a given structure

  Scenario: Farthest Point Sampling
    Given a set of 3 points in descriptor space: A(0,0), B(0.1, 0), C(10, 10)
    When I request to select 2 points using FPS
    Then the selected set should contain A and C (or B and C)
    And the selected set should NOT contain just A and B (as they are too close)
```
