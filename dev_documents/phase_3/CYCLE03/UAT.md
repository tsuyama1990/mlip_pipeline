# Cycle 03 UAT: Surrogate Selection

## 1. Test Scenarios

### Scenario 3.1: Pre-screening with Foundation Model
-   **Priority**: High
-   **Description**: Ensure that the system can use MACE (or a mock) to evaluate generated structures. This tests the integration of an external PyTorch model into the pipeline.
-   **Pre-conditions**:
    -   DB contains 50 pending structures generated in Cycle 02.
    -   Internet access is available (to download MACE weights for the first run).
-   **Detailed Steps**:
    1.  User executes `mlip-auto select --method mace --n 10`.
    2.  System initializes the surrogate model. It checks for cached weights. If missing, it downloads them from GitHub/HuggingFace.
    3.  System loads the atoms from the DB.
    4.  System batches atoms (e.g., 2 batches of 25) and passes them to MACE.
    5.  System receives Energy and Force predictions.
    6.  System calculates descriptors.
    7.  System runs FPS selection.
    8.  System updates the database status.
    9.  User queries the DB count of `status='selected'`.
-   **Post-conditions**:
    -   10 rows have `status="selected"`.
    -   40 rows have `status="held"` or `status="rejected"`.
    -   Selected rows contain `mace_energy` and `mace_forces` in their key-value pairs.
-   **Failure Modes**:
    -   Network error (cannot download model).
    -   OOM error (batch size too large for GPU).

### Scenario 3.2: Filtering Bad Structures (Clash Detection)
-   **Priority**: Medium
-   **Description**: The generator might create atoms that are overlapping. We must not send these to DFT. This tests the "Fail-Fast" logic.
-   **Pre-conditions**:
    -   User manually inserts a structure with two atoms at distance 0.1 A (unphysical).
-   **Detailed Steps**:
    1.  User runs `mlip-auto select`.
    2.  System evaluates forces using MACE.
    3.  The clash yields a force magnitude > 1000 eV/A.
    4.  System compares this to the threshold (default 50 eV/A).
    5.  System logs "Structure rejected due to high forces > 50 eV/A".
    6.  The bad structure is marked `REJECTED` in the database.
-   **Post-conditions**:
    -   The bad structure is NEVER selected for DFT.
    -   It is preserved in the DB for audit purposes but flagged as bad.
-   **Failure Modes**:
    -   Threshold too low (rejecting valid structures).

### Scenario 3.3: Diversity Selection (FPS)
-   **Priority**: High
-   **Description**: Verify that the selection algorithm actually maximizes diversity. This is crucial for active learning efficiency.
-   **Pre-conditions**:
    -   The pool contains 40 FCC crystals (very similar) and 10 Liquid structures (very different).
-   **Detailed Steps**:
    1.  User runs `mlip-auto select --n 5`.
    2.  FPS calculates descriptors. The liquid structures will be far from the crystals in descriptor space.
    3.  The algorithm picks the first point (e.g., a crystal).
    4.  The next point selected will be the farthest possible point (likely a liquid).
    5.  The next points will fill the gaps.
    6.  User inspects the `config_type` of the selected items.
-   **Post-conditions**:
    -   The selection includes a mix of the crystals and the liquids (e.g., 2 crystals, 3 liquids), rather than just the first 5 entries in the database.
-   **Failure Modes**:
    -   Descriptor calculation fails or produces NaNs.

## 2. Behaviour Definitions

```gherkin
Feature: Surrogate-Based Selection
  As a resource-constrained scientist
  I want to filter candidate structures using a cheap model
  So that I only run expensive DFT calculations on the most valuable candidates

  Scenario: Basic Selection Flow
    Given a database with 100 "pending" structures
    And a target selection count of 10
    When I run the surrogate selection command
    Then exactly 10 structures should be updated to "selected" status
    And the remaining 90 should be "held" or "rejected"
    And the selected structures should have MACE energy predictions stored

  Scenario: Rejection of Unphysical Structures
    Given a structure with atoms overlapping (distance < 0.5 A)
    When the surrogate model evaluates it
    Then the predicted force should exceed the safety threshold
    And the structure status should be set to "rejected"
    And a warning should be logged

  Scenario: Idempotency
    Given the selection has already run and 10 items are selected
    When I run the command again
    Then it should not re-select or duplicate the work
    And it should output "No pending structures found" or similar message
```
