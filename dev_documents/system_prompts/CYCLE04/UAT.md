# Cycle 04 UAT: The Efficient Learner

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-04-01** | High | **Active Set Reduction** | Verify that the system reduces the number of training structures without losing information diversity. |
| **UAT-04-02** | Medium | **Potential Generation** | Verify that the training process successfully produces a valid `.yace` potential file. |

## 2. Behavior Definitions

### UAT-04-01: Active Set Reduction

**GIVEN** a pool of 100 candidate structures, where 90 are identical perturbations of a crystal and 10 are unique
**WHEN** the `ActiveSetSelector` is run with a target count of 20
**THEN** the output should contain the 10 unique structures
**AND** a subset of the perturbed ones
**AND** the total count should be <= 20

### UAT-04-02: Potential Generation

**GIVEN** a valid dataset of 5 structures (with Energy and Forces)
**AND** a valid `TrainingConfig`
**WHEN** `PacemakerTrainer.train()` is called
**THEN** the system should execute `pace_train`
**AND** a file named `output_potential.yace` should exist in the output directory
**AND** the log should report "Training finished"
