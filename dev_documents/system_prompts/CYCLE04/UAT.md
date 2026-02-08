# Cycle 04 UAT: Trainer (Pacemaker Integration)

## 1. Test Scenarios

These scenarios verify the integration with the machine learning backend (Pacemaker).

### Scenario 04-01: "Active Set Generation"
**Priority:** P1 (High)
**Description:** Verify that the system can filter a large dataset into a smaller, representative training set using D-Optimality.
**Success Criteria:**
-   **Input:** A mock dataset with 1000 structures.
-   **Config:** `activeset_max_size: 50`.
-   **Operation:** The system calls `pace_activeset`.
-   **Result:** A `train_set.pckl` is created containing exactly 50 structures.

### Scenario 04-02: "Delta Learning Config"
**Priority:** P2 (Medium)
**Description:** Verify that the system correctly configures Pacemaker to use a physical baseline (ZBL/LJ).
**Success Criteria:**
-   **Config:** `trainer: pacemaker`, `reference_potential: zbl`.
-   **Operation:** Run the pipeline (mock mode).
-   **Check:** Inspect the generated `input.yaml` in the work directory.
-   **Result:** It must contain a `potential:` section with `type: zbl` (or equivalent definition as per Pacemaker docs).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Trainer Integration

  Scenario: Active Set Optimization
    Given a dataset with 1000 structures
    When I request an active set of size 50
    Then the trainer should produce a training file with exactly 50 structures
    And the structures should be the most diverse ones (mock check)

  Scenario: Delta Learning Configuration
    Given a configuration requesting ZBL baseline
    When the trainer generates the input file
    Then the input file should contain the ZBL parameters
    And the training command should execute without error
```
