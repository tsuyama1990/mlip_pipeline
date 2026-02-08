# Cycle 02 UAT: Data Management & Structure Generation

## 1. Test Scenarios

These scenarios verify the data pipeline and structure generation logic.

### Scenario 02-01: "Data Persistence"
**Priority:** P0 (Critical)
**Description:** Verify that the Dataset component correctly saves generated and labeled structures to disk.
**Success Criteria:**
-   Run a mock pipeline with `generator: random`.
-   Inspect the generated `workdir/dataset.jsonl` (or `.extxyz`).
-   The file must contain valid structure entries (coordinates, cell, types).
-   Running the pipeline again (resume mode) should load the existing data without corruption.

### Scenario 02-02: "Random Alloy Generation"
**Priority:** P1 (High)
**Description:** Verify that the Random Generator creates physically plausible starting structures.
**Success Criteria:**
-   Config: `composition: {Fe: 0.5, Pt: 0.5}`, `lattice: fcc`.
-   Output: A supercell structure where approximately 50% of sites are Fe and 50% are Pt.
-   **Physics Check:** No two atoms should be closer than 1.5 Å (default minimum distance).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Data Management & Structure Generation

  Scenario: Dataset Append and Load
    Given an empty dataset
    When I append 5 random structures
    And I load the dataset from disk
    Then the loaded dataset should contain 5 structures
    And the first structure's position [0,0] should match the original

  Scenario: Random Alloy Generation Logic
    Given a request for an FePt random alloy
    When the generator produces a structure
    Then the structure should have "Fe" and "Pt" atoms
    And the minimum interatomic distance should be > 1.5 Angstroms
```
