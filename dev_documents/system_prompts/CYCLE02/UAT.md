# Cycle 02 UAT: DIRECT Sampling and Active Learning

## 1. Test Scenarios

### Scenario 01: Intelligent Structure Generation
-   **Priority**: High
-   **Description**: Verify that the DIRECT sampling algorithm produces a diverse set of initial structures, not just random noise.
-   **Execution**:
    1.  Create `tests/uat/configs/cycle02_direct.yaml` specifying `step1_direct_sampling.target_points: 50`.
    2.  Run `python -m pyacemaker.main config.yaml`.
    3.  Check output for "Step 1: Generated 50 structures".
    4.  Verify that the pairwise distance matrix of descriptors is computed and mean distance > 0.1.

### Scenario 02: Active Learning Selection
-   **Priority**: Critical
-   **Description**: Verify that the system correctly identifies and selects high-uncertainty structures for DFT calculation.
-   **Execution**:
    1.  Run the pipeline with `step2_active_learning.uncertainty_threshold: 0.8`.
    2.  Check output for "Step 2: Selected X structures for DFT".
    3.  Verify that the selected structures have `uncertainty > 0.8`.
    4.  Verify that the final dataset contains structures with labels from the `DFTOracle` (Mock).

## 2. Behavior Definitions (Gherkin)

### Feature: Structure Generation and Active Learning

**Scenario: Generate diverse structures with DIRECT**
  GIVEN a valid config with `step1_direct_sampling` enabled
  WHEN the Orchestrator executes Step 1
  THEN it should generate 50 structures
  AND the structures should be marked as `source="DIRECT"`
  AND the structures should have diverse descriptors (Mean Pairwise Distance > Threshold)

**Scenario: Filter structures via Active Learning**
  GIVEN a pool of 50 generated structures
  AND a `MaceSurrogateOracle` that reports uncertainty
  WHEN the Orchestrator executes Step 2
  THEN it should calculate uncertainty for all 50 structures
  AND it should select the top 10 most uncertain ones (mocked behavior)
  AND it should send these 10 structures to the `DFTOracle`
  AND the `DFTOracle` should label them with "True" Energy/Forces
  AND the Orchestrator should store them in the `dft_dataset`
