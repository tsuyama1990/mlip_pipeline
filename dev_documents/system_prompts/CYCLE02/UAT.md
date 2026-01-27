# Cycle 02: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 01: Initial Structure Generation
*   **ID**: UAT-CY02-01
*   **Priority**: High
*   **Description**: Verify that the system can generate a set of perturbed structures from a single input crystal file, creating a diverse initial dataset.
*   **Pre-conditions**: A `cif` or `xyz` file of a primitive unit cell (e.g., Al.cif).
*   **Steps**:
    1.  Run `mlip-auto generate --input Al.cif --count 10 --strain 0.1 --rattle 0.05 --output initial_set.xyz`.
    2.  Open the output file `initial_set.xyz` (using a visualizer or text editor).
*   **Expected Result**:
    *   The file contains 10 frames.
    *   The lattice constants of the frames are different (due to strain).
    *   The atomic positions are slightly disordered (due to rattling).

### Scenario 02: Database Storage and Retrieval
*   **ID**: UAT-CY02-02
*   **Priority**: Medium
*   **Description**: Verify that generated structures can be saved to the internal database format and retrieved without data loss.
*   **Steps**:
    1.  Generate structures as in Scenario 01.
    2.  Run `mlip-auto db-import --input initial_set.xyz --output database.pckl.gzip`.
    3.  Run `mlip-auto db-inspect --input database.pckl.gzip`.
*   **Expected Result**:
    *   Step 2 completes without error.
    *   Step 3 outputs a summary table showing "10 structures" and lists their composition.

### Scenario 03: Mock Training Run
*   **ID**: UAT-CY02-03
*   **Priority**: High
*   **Description**: Verify that the Trainer module correctly interfaces with the Pacemaker software (or its mock) to initiate a training job.
*   **Pre-conditions**: `pace_train` executable is available (or mocked). A dummy `dataset.pckl.gzip` exists.
*   **Steps**:
    1.  Create a `train_config.yaml` specifying `batch_size: 10` and `epochs: 5`.
    2.  Run `mlip-auto train --config train_config.yaml --dataset dataset.pckl.gzip`.
*   **Expected Result**:
    *   The system generates an `input.yaml` for Pacemaker.
    *   The system executes the training command.
    *   (If real execution) Output logs show the loss decreasing.
    *   The system reports "Training Completed" and points to the `potential.yace` file.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Generation and Management

  Scenario: Generate rattled structures
    GIVEN a primitive unit cell of Copper
    WHEN I request 20 generated structures with 5% strain
    THEN the system should produce 20 Atom objects
    AND the volumes of these objects should vary within +/- 15% (approx) of the original
    AND the atomic positions should not be identical to the input

  Scenario: Train a potential from database
    GIVEN a database file containing 50 DFT-calculated structures
    AND a training configuration with max_epochs=10
    WHEN I invoke the training command
    THEN the system should create a temporary build directory
    AND the system should generate a valid 'input.yaml' for Pacemaker
    AND the 'pace_train' process should be started
    AND upon completion, a 'potential.yace' file should exist
```
