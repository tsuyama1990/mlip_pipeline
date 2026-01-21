# Cycle 05 UAT: Active Learning Trainer

## 1. Test Scenarios

### Scenario 5.1: Data Preparation for Training
-   **Priority**: High
-   **Description**: Convert the database content into a format Pacemaker can understand. This involves serializing ASE Atoms objects into the Extended XYZ format, which includes lattice vectors, species, positions, energies, forces, and stresses.
-   **Pre-conditions**:
    -   DB has 100 labeled structures (status='completed').
    -   User has permissions to write to the `data/` directory.
-   **Detailed Steps**:
    1.  User executes `mlip-auto train --prepare-only`.
    2.  System queries the DB for `status='completed'`.
    3.  System splits the data into Training (90 atoms) and Validation (10 atoms) sets using a random seed.
    4.  System writes `data/train.xyz` and `data/test.xyz`.
    5.  User inspects the files.
    6.  User verifies the header contains `Lattice="..." Properties=species:S:1:pos:R:3:forces:R:3 energy=...`.
-   **Post-conditions**:
    -   Files exist and are valid ExtXYZ.
    -   Data count matches DB count.
-   **Failure Modes**:
    -   DB empty (should warn user).
    -   Disk full.

### Scenario 5.2: Training Execution
-   **Priority**: Critical
-   **Description**: Run the training loop to produce a potential. This is the core learning step.
-   **Pre-conditions**:
    -   Data files exist (`train.xyz`, `test.xyz`).
    -   Pacemaker is installed (or mocked for test).
-   **Detailed Steps**:
    1.  User executes `mlip-auto train`.
    2.  System reads `TrainingConfig` (cutoff, order).
    3.  System generates `input.yaml` for Pacemaker.
    4.  System launches Pacemaker as a subprocess.
    5.  Console shows a progress bar or epoch log (Epoch 1..10..).
    6.  Process exits with code 0.
    7.  System checks for `output.yace`.
    8.  System renames it to `potential_gen_X.yace` and moves it to `potentials/`.
-   **Post-conditions**:
    -   A valid `.yace` potential file exists.
    -   RMSE values are printed to stdout.
-   **Failure Modes**:
    -   Divergence (RMSE goes to infinity).
    -   Pacemaker crash (segfault).

### Scenario 5.3: Metric Analysis
-   **Priority**: Medium
-   **Description**: Verify that we can track the quality of the fit. We need to know if the model is overfitting.
-   **Pre-conditions**:
    -   A completed training run log exists (`log.txt`).
-   **Detailed Steps**:
    1.  System calls the Metric Parser.
    2.  System reads `log.txt`.
    3.  System extracts the final Validation RMSE for Energy and Forces.
    4.  System checks if `Validation RMSE > 1.5 * Training RMSE` (Overfitting warning).
    5.  System saves these metrics to a JSON report `training_metrics.json`.
-   **Post-conditions**:
    -   Report contains keys `rmse_energy_meV_atom` and `rmse_force_eV_A`.
    -   User can plot learning curves.
-   **Failure Modes**:
    -   Log file format changed (parser breaks).

## 2. Behaviour Definitions

```gherkin
Feature: Model Training
  As a user
  I want to fit a potential to my data
  So that I can run MD simulations

  Scenario: Generating Training Input
    Given a database with 100 labeled DFT structures
    When I trigger the data export command
    Then the data should be split into training and validation sets (90/10)
    And the output files should be in ExtXYZ format compatible with Pacemaker
    And the files should contain energy, forces, and stress data

  Scenario: Configuring the Trainer
    Given a training configuration with cutoff=5.0 and force_weight=100.0
    When the system generates the Pacemaker input file
    Then the YAML file should contain "cutoff: 5.0"
    And the loss function weights should reflect the high priority of forces (kappa_f > kappa_e)

  Scenario: Successful Training Run
    Given valid input data and configuration
    When the training process completes successfully
    Then a .yace potential file should be produced in the output directory
    And the final validation error should be logged to a JSON file
```
