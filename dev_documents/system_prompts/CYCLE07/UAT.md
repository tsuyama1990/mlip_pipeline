# Cycle 07 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario ID: UAT-C07-001 - The Discovery of the Unknown (Active Learning Trigger)

**Priority:** High
**Description:**
This test verifies the system's ability to detect when it enters "unknown territory". We simulate an MD run where the potential becomes unstable (high uncertainty). The system must catch this, stop the simulation, and extract the problematic configuration.

**User Story:**
As a Scientist, I want my simulation to auto-stop when the atoms enter a configuration that the potential hasn't seen before (Extrapolation Grade > 5.0), so that I don't generate junk physical data. Instead, I want that configuration to be automatically queued for DFT refinement.

**Step-by-Step Walkthrough:**
1.  **Preparation**: The user provides a trained potential and a starting structure (liquid).
2.  **Configuration**: `inference.uncertainty_threshold: 5.0`.
3.  **Execution**: `mlip-auto run-md`.
    -   *Mock*: The LAMMPS runner is mocked to simulate $\gamma$ rising from 1.0 to 6.0 at step 500.
4.  **Observation**:
    -   CLI: "Step 100: Gamma=1.2"
    -   CLI: "Step 500: Gamma=6.1 -> HALT TRIGGERED."
5.  **Extraction**:
    -   CLI: "Extracting cluster around atom 42..."
    -   CLI: "Candidate structure added to DB (ID: 101)."
6.  **Verification**:
    -   User checks DB ID 101.
    -   It is a small cell (e.g., 50 atoms).
    -   It has `force_mask` array.
    -   Status is `PENDING`.

**Success Criteria:**
-   Simulation stops early.
-   The correct atom (center of uncertainty) is identified.
-   The candidate is stored for the next cycle.

### Scenario ID: UAT-C07-002 - Force Masking Verification

**Priority:** High
**Description:**
If we train on the buffer atoms, we will ruin the potential. We must verify that the masking logic is correct: 1 for core, 0 for buffer.

**User Story:**
As a Developer, I want to verify that the extracted clusters have the correct `force_mask`, so that the Training Engine (Cycle 06) knows which atoms to trust.

**Step-by-Step Walkthrough:**
1.  **Action**: Use `EmbeddingExtractor` to cut a cluster with $r_{core}=3.0, r_{buffer}=2.0$.
2.  **Analysis**:
    -   Measure distance from center to Atom A (2.5 Å). -> Mask should be 1.0.
    -   Measure distance from center to Atom B (4.0 Å). -> Mask should be 0.0.
3.  **Verification**: Check the `atoms.arrays['force_mask']` values.

**Success Criteria:**
-   Mask values correspond strictly to the geometric distance logic.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Inference
  As an Explorer
  I want to run simulations and detect uncertainty
  So that I can improve the potential autonomously

  Scenario: Stop on High Uncertainty
    Given the uncertainty threshold is 5.0
    When the simulation reports a max extrapolation grade of 5.1
    Then the simulation should halt immediately
    And the current snapshot should be captured

  Scenario: Cluster Extraction with Masking
    Given a captured snapshot with high uncertainty at Atom X
    When I extract a cluster with core radius 4.0A and buffer 3.0A
    Then the new structure should have a bounding box of approx 14.0A
    And atoms within 4.0A of X should have force_mask = 1.0
    And atoms between 4.0A and 7.0A should have force_mask = 0.0
```
