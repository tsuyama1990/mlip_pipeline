# CYCLE 02 UAT: Trainer & Baseline Integration

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-02-01** | High | Dataset Serialization | Verify that the system can save and load training data in the `.pckl.gzip` format required by Pacemaker. |
| **UAT-02-02** | High | Baseline Config Generation | Verify that the system automatically generates a valid `input.yaml` with ZBL/LJ baseline settings for a given element set. |
| **UAT-02-03** | Medium | Trainer Execution (Mocked Binary) | Verify that the `PacemakerTrainer` constructs the correct CLI command and handles the output file path. |

## 2. Behavior Definitions

### Scenario: Baseline Configuration
**GIVEN** a training configuration for elements ["Fe", "Pt"]
**AND** `baseline_model` is set to "ZBL"
**WHEN** the Trainer prepares the Pacemaker input
**THEN** the generated `input.yaml` should contain a `potential` section
**AND** it should define `pair_style: hybrid/overlay`
**AND** it should include `zbl` parameters for Fe-Fe, Fe-Pt, and Pt-Pt.

### Scenario: Dataset Validation
**GIVEN** a set of `ase.Atoms` where one atom is missing "forces" array
**WHEN** the `DatasetManager` attempts to save this set
**THEN** it should raise a `ValidationError`
**AND** protect the user from training on invalid data.
