# Cycle 03: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 3.1: MACE Pre-screening (Filter "Bad" Structures)
-   **Priority**: High
-   **Description**: Ensure the surrogate correctly identifies and rejects unphysical structures.
-   **Steps**:
    1.  Generate a "Good" structure (Equilibrium Al).
    2.  Generate a "Bad" structure (Two Al atoms at 0.5 Angstrom distance).
    3.  Run `MaceClient` on both.
    4.  Apply `filter_unphysical(threshold=50.0)`.
-   **Success Criteria**:
    -   The "Good" structure is retained (Max Force < 50).
    -   The "Bad" structure is rejected (Max Force > 50, likely thousands).
    -   Rejected structure is logged with reason.

### Scenario 3.2: Diversity Selection (FPS)
-   **Priority**: Medium
-   **Description**: Verify that FPS selects a diverse set of structures from a generated pool.
-   **Steps**:
    1.  Generate 100 SQS structures with varying strain (-10% to +10%).
    2.  Use MACE to get descriptors.
    3.  Select 10 structures using FPS.
    4.  Select 10 structures Randomly.
    5.  Compare the volume distribution of both sets.
-   **Success Criteria**:
    -   FPS set covers the extremes (min volume, max volume) consistently.
    -   Random set might cluster in the middle.
    -   (Visual check in Notebook): PCA plot of descriptors shows FPS points spread out.

### Scenario 3.3: Throughput Benchmark
-   **Priority**: Low
-   **Description**: Ensure the surrogate is significantly faster than DFT.
-   **Steps**:
    1.  Prepare 100 structures.
    2.  Time the `MaceClient` evaluation (on CPU or GPU).
-   **Success Criteria**:
    -   Processing 100 structures should take seconds (or < 1 min on CPU).
    -   Equivalent DFT would take hours. Validating the speedup justifies the architecture.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Surrogate Exploration

  Scenario: Filtering high-energy collisions
    Given a batch of candidate structures
    And a force threshold of 100 eV/A
    When the MACE surrogate evaluates the batch
    Then any structure with max force > 100 should be flagged "rejected"
    And valid structures should proceed to selection

  Scenario: Selecting diverse candidates
    Given a pool of 1000 valid candidates
    And an existing dataset of 50 structures
    When FPS selection is run for 10 new samples
    Then the selected 10 samples should have the maximum distance from the existing 50
    And the selected samples should be diverse among themselves
```
