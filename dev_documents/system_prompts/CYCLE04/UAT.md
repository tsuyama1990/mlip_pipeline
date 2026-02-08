# Cycle 04: Trainer UAT

## 1. Test Scenarios

### Scenario 04-01: Train from Scratch
**Priority**: High
**Goal**: Verify Pacemaker integration.
**Description**:
Train a potential on a small dataset of 100 LJ Argon structures.
**Expected Outcome**:
-   The training completes successfully (exit code 0).
-   `potential.yace` is created.
-   `metrics.json` contains training RMSE (Energy < 0.1 meV/atom, Force < 0.05 eV/A).

### Scenario 04-02: Fine-Tuning (Resume)
**Priority**: Medium
**Goal**: Verify delta learning or incremental training.
**Description**:
1.  Take the potential from Scenario 04-01.
2.  Add 10 more high-energy structures (e.g., dimers).
3.  Train *starting from* the previous potential (`--initial_potential`).
**Expected Outcome**:
-   The training finishes much faster (fewer epochs needed).
-   The new potential has lower error on the *new* structures compared to the old one.

### Scenario 04-03: Active Set Reduction
**Priority**: Critical
**Goal**: Verify D-Optimality.
**Description**:
1.  Create a dataset with 1000 identical bulk structures (redundant).
2.  Run `Trainer.train(dataset, active_set_size=10)`.
**Expected Outcome**:
-   The Trainer selects only ~10 structures.
-   The log says "Selected 10 structures from 1000 candidates".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Training Loop

  Scenario: Train on Argon
    Given a dataset of Argon atoms (LJ potential)
    When I request training with max_epochs=100
    Then a file "potential.yace" should be generated
    And the training RMSE for energy should be < 0.001 eV/atom

  Scenario: Active Set Selection
    Given a redundant dataset of 1000 structures
    When I request training with active_set_limit=50
    Then the internal dataset size should be <= 50
    And the training time should be significantly reduced
```

## 3. Jupyter Notebook Validation (`tutorials/03_Trainer_Test.ipynb`)
-   **Load Data**: Create a simple LJ dataset using `ase`.
-   **Train**: Call `trainer.train(dataset)`.
-   **Predict**: Load the resulting potential with `pyace` or `lammps` and predict forces on a test set.
-   **Plot**: Parity plot of $F_{pred}$ vs $F_{true}$.
