# Cycle 03 UAT: MACE Fine-tuning and Surrogate Generation

## 1. Test Scenarios

### Scenario 01: Successful MACE Fine-tuning
-   **Priority**: Critical
-   **Description**: Verify that the system can fine-tune the MACE model using the DFT data collected in Step 2.
-   **Execution**:
    1.  Create `tests/uat/configs/cycle03_finetune.yaml` specifying `step3_mace_finetune.base_model: "MACE-MP-0"`.
    2.  Provide a pre-existing `dft_dataset.pckl` (e.g., from Cycle 02 output).
    3.  Run the pipeline for Step 3.
    4.  Check output for "Fine-tuning complete" and verify `fine_tuned_mace.model` is created.

### Scenario 02: Diverse Surrogate Sampling
-   **Priority**: High
-   **Description**: Verify that the MD engine generates a diverse set of surrogate structures using the fine-tuned model.
-   **Execution**:
    1.  Run Step 4 with `step4_surrogate_sampling.target_points: 100` and `temperature: 1000K`.
    2.  Check output for "Step 4: Generated 100 structures via MD".
    3.  Load the generated structures and verify that they are not all identical (variance in energy/volume > 0).

## 2. Behavior Definitions (Gherkin)

### Feature: MACE Model Adaptation

**Scenario: Fine-tune MACE on DFT data**
  GIVEN a `dft_dataset` with labeled structures
  AND a base `MACE-MP-0` model configuration
  WHEN the Orchestrator executes Step 3
  THEN it should invoke the `MaceTrainer`
  AND the `MaceTrainer` should produce a `fine_tuned_mace.model` file
  AND the system should log the training loss (mocked)

**Scenario: Generate surrogate data via MD**
  GIVEN a valid `fine_tuned_mace.model`
  AND an initial structure from the `dft_dataset`
  WHEN the Orchestrator executes Step 4
  THEN it should initialize the `DynamicsEngine` with the MACE calculator
  AND it should run an MD simulation for the configured steps
  AND it should collect the trajectory frames into `surrogate_candidates`
  AND the generated structures should be physically reasonable (no core overlaps)
