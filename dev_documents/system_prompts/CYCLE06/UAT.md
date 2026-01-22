# Cycle 06 UAT: Orchestration & Inference

## 1. Test Scenarios

### Scenario 6.1: Run Inference MD
-   **Priority**: High
-   **Description**: Run a molecular dynamics simulation using the trained potential. This validates that the potential can actually drive atoms.
-   **Pre-conditions**:
    -   A `.yace` file exists (from Cycle 05).
    -   LAMMPS is installed with the MLIP plugin.
-   **Detailed Steps**:
    1.  User executes `mlip-auto simulate --potential current.yace --structure start.xyz`.
    2.  System writes `in.lammps`. It sets `pair_style pace`.
    3.  System executes `lmp_serial < in.lammps`.
    4.  Console logs progress (Step 100, Temp 300K, Gamma 1.2).
    5.  Simulation finishes (e.g., 1000 steps).
    6.  System parses `log.lammps` to ensure energy conservation (if NVE).
-   **Post-conditions**:
    -   `dump.lammpstrj` exists and contains valid atomic trajectories.
    -   Exit code is 0.
-   **Failure Modes**:
    -   LAMMPS segfault (potential file corrupted).
    -   Atoms fly apart (bad potential).

### Scenario 6.2: Uncertainty Detection & Stop
-   **Priority**: Critical
-   **Description**: Ensure the simulation aborts when it encounters unknown physics. This is the "Active Learning" trigger.
-   **Pre-conditions**:
    -   Use a potential trained only on Low-Temp data.
    -   Run MD at High-Temp (Mock or Real).
    -   Config sets `uncertainty_threshold: 10.0`.
-   **Detailed Steps**:
    1.  System generates LAMMPS input with `compute g all extrapolation_grade ...` and `fix halt ... variable v_g > 10.0 ...`.
    2.  Run MD.
    3.  At step 500, the atoms enter a high-energy config. $\gamma$ spikes to 12.0.
    4.  LAMMPS `fix halt` triggers. Simulation stops.
    5.  System detects early exit.
    6.  System logs "High Uncertainty Detected".
-   **Post-conditions**:
    -   The simulation did not run to completion (e.g., 10000 steps).
    -   The final frame in the dump file corresponds to the high-uncertainty state.
-   **Failure Modes**:
    -   LAMMPS ignores fix halt.

### Scenario 6.3: Cluster Extraction
-   **Priority**: High
-   **Description**: Verify that we can cut out a cluster for re-training. This prepares the feedback loop.
-   **Pre-conditions**:
    -   A structure with high uncertainty exists (result of Scenario 6.2).
-   **Detailed Steps**:
    1.  System calls `EmbeddingExtractor`.
    2.  System reads the dump file.
    3.  System identifies atom ID 42 has max $\gamma$.
    4.  System creates a new structure centered on atom 42, including all neighbors within 6.0A.
    5.  System calculates the `force_mask`.
    6.  System saves to DB with `status="pending"` and `config_type="active_learning"`.
-   **Post-conditions**:
    -   DB contains a new record.
    -   The structure size is small (e.g., 30-60 atoms).
    -   The structure is not periodic (or is a small periodic supercell).
-   **Failure Modes**:
    -   Extractor fails on PBC.

### Scenario 6.4: Full Autonomous Loop (Zero-Human)
-   **Priority**: Critical
-   **Description**: The ultimate test. Can it run by itself?
-   **Pre-conditions**:
    -   Empty DB.
    -   Valid Config.
    -   Mocks for heavy binaries (to speed up test).
-   **Detailed Steps**:
    1.  User executes `mlip-auto run loop`.
    2.  System enters `GENERATION` state -> Creates 10 structures.
    3.  System enters `SELECTION` state -> Picks 5.
    4.  System enters `DFT` state -> Runs "DFT" on 5.
    5.  System enters `TRAINING` state -> Trains Potential V1.
    6.  System enters `INFERENCE` state -> Runs "MD" with V1.
    7.  Mock MD fails at step 50.
    8.  System extracts 1 structure. Adds to DB.
    9.  System detects new pending DFT. Transitions to `DFT`.
    10. System runs DFT on 1 structure.
    11. System trains Potential V2.
-   **Post-conditions**:
    -   The loop cycles at least once.
    -   The generation counter increments.
    -   Dashboard shows the progress.
-   **Failure Modes**:
    -   Deadlock (waiting for tasks that never finish).

## 2. Behaviour Definitions

```gherkin
Feature: Autonomous Active Learning
  As a principal investigator
  I want the system to improve the potential automatically
  So that I can sleep while it works

  Scenario: Inference Interruption
    Given a running MD simulation with a threshold of 5.0
    When the extrapolation grade of any atom reaches 5.1
    Then the simulation should be terminated immediately
    And the offending configuration should be extracted
    And the new candidate should be added to the database with status "pending"

  Scenario: Workflow State Machine
    Given the workflow manager is in IDLE state
    When new DFT results arrive in the database
    Then the state should transition to TRAINING
    And a new training job should be submitted

  Scenario: Cluster Extraction
    Given a large simulation box (1000 atoms)
    And a target atom index corresponding to high error
    When the extractor runs with a cutoff of 6.0 Angstrom
    Then a new structure containing only the neighbors should be created
    And the boundary atoms should be masked (weight=0)
    And the center atoms should be unmasked (weight=1)
```
