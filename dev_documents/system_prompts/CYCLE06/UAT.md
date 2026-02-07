# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 6.1: CLI User Experience
**Goal**: Verify that the CLI is intuitive and robust.
**Priority**: High (P1) - First impression matters.
**Steps**:
1.  Run `pyacemaker init my_project`.
2.  Assert `my_project/config.yaml` is created.
3.  Modify config for "Mock" backend.
4.  Run `pyacemaker run --config my_project/config.yaml`.
5.  In another terminal, run `pyacemaker status`.
**Success Criteria**:
*   `init` creates a valid template.
*   `run` executes without errors.
*   `status` shows the current iteration and dataset sise.

### Scenario 6.2: End-to-End Fe/Pt on MgO
**Goal**: Verify the full scientific workflow (Tutorial 03).
**Priority**: Critical (P0) - Core value proposition.
**Steps**:
1.  Execute `tutorials/03_Deposition_and_Ordering.ipynb`.
2.  Use "CI Mode" (tiny system, mock Oracle/Trainer).
3.  Verify that:
    *   MD deposition runs.
    *   Uncertainty is detected.
    *   New potential is trained (simulated).
    *   kMC runs.
    *   Final structure is visualised.
**Success Criteria**:
*   The notebook runs from top to bottom without errors.
*   Plots are generated.
*   The narrative makes sense to a user.

## 2. Behaviour Definitions (Gherkin)

```gherkin
Feature: Production CLI

  Scenario: Initialise Project
    When I run "pyacemaker init new_proj"
    Then a directory "new_proj" should be created
    And "new_proj/config.yaml" should exist

  Scenario: Check Status
    Given a running project
    When I run "pyacemaker status"
    Then it should display the current iteration
    And the number of structures in the dataset
    And the RMSE of the latest potential
```
