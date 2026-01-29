# User Acceptance Testing (UAT): Cycle 04

## 1. Test Scenarios

Cycle 04 gives the system the ability to "Learn".

### Scenario 4.1: Full Training Loop
-   **ID**: UAT-C04-01
-   **Priority**: Critical
-   **Description**: Given a set of labelled structures (from Cycle 03), train a potential.
-   **Success Criteria**:
    -   `pace_train` runs to completion.
    -   A `potential.yace` file is created.
    -   A training report is generated showing Energy RMSE and Force RMSE.
    -   The RMSE should be "reasonable" (e.g., < 10 meV/atom for a simple test case).

### Scenario 4.2: Active Set Selection
-   **ID**: UAT-C04-02
-   **Priority**: High
-   **Description**: We have 1000 structures, but many are similar. We want to train on only the most important ones.
-   **Success Criteria**:
    -   User configures `active_set_selection: true`.
    -   System runs `pace_activeset`.
    -   Logs show "Reduced dataset from 1000 to X structures".
    -   Training proceeds using the reduced dataset.

### Scenario 4.3: Delta Learning (Robustness)
-   **ID**: UAT-C04-03
-   **Priority**: Medium
-   **Description**: Ensure that the reference potential (ZBL/LJ) is correctly included.
-   **Success Criteria**:
    -   Inspect the generated `potential.yace` (or the input YAML).
    -   It should contain a definition for the reference potential (e.g., `type: ZBL`).
    -   This confirms that the ACE model is learning the *difference*, not the absolute energy.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Machine Learning Potential Training

  Background:
    Given a valid "train.pckl.gzip" dataset exists
    And the config specifies "Titanium" and "Oxygen"

  Scenario: Train a new potential
    When I run the training command
    Then the process should exit with code 0
    And a file "potential.yace" should be created
    And the log should contain "RMSE Energy"
    And the log should contain "RMSE Force"

  Scenario: Resume training from previous generation
    Given an existing "old_potential.yace"
    When I run the training command with "initial_potential=old_potential.yace"
    Then the system should use the old potential as a starting point (fine-tuning)
    And the training time should be shorter than a fresh start
```
