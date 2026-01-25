# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Dataset Conversion with Delta Learning
*   **ID**: UAT-03-01
*   **Priority**: High
*   **Description**: Verify that the system correctly subtracts the baseline energy from the DFT energy when preparing the dataset.
*   **Success Criteria**:
    *   Input: Atom pair with $E_{DFT} = -10.0$ eV.
    *   Baseline: LJ potential gives $E_{LJ} = 1.0$ eV at this distance.
    *   Output: The training data stored in the file should have $E_{target} = -11.0$ eV.

### Scenario 2: Active Set Selection
*   **ID**: UAT-03-02
*   **Priority**: Medium
*   **Description**: Verify that `pace_activeset` is called correctly to reduce dataset size.
*   **Success Criteria**:
    *   Mock input: 1000 candidate structures.
    *   Mock output from `pace_activeset`: 50 selected structures.
    *   The system should log "Selected 50 structures from 1000 candidates".

### Scenario 3: Training Execution
*   **ID**: UAT-03-03
*   **Priority**: Critical
*   **Description**: Verify that the training command is constructed and executed correctly.
*   **Success Criteria**:
    *   The command line must include `--dataset`, `--output_dir`, and `--test_size`.
    *   If `initial_potential` is provided (fine-tuning mode), the command must include it.

## 2. Behavior Definitions

```gherkin
Feature: Trainer (Pacemaker)

  As a researcher
  I want the system to handle the complexity of delta-learning
  So that I can get physically robust potentials without manual data manipulation

  Scenario: Delta Learning Data Prep
    GIVEN a DFT dataset with total energies
    AND a configured Lennard-Jones baseline
    WHEN I convert the dataset for Pacemaker
    THEN the stored energies should be the difference (DFT - LJ)

  Scenario: Fine-tuning
    GIVEN an existing potential "gen_001.yace"
    AND a new batch of data
    WHEN I request a training update
    THEN the trainer should run in "fine-tuning" mode (fewer epochs)
    AND the trainer should initialize from "gen_001.yace"
```
