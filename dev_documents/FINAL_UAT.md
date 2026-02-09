# Final User Acceptance Test (UAT) Plan

## 1. Tutorial Strategy

The User Acceptance Testing (UAT) strategy for PyAceMaker is centered around a "Grand Challenge" scenario: **Fe/Pt Deposition on MgO**. This scenario was chosen because it exercises every component of the system—from multi-species potential generation (Structure Generator, Oracle, Trainer) to dynamic deposition (LAMMPS) and long-timescale ordering (EON/kMC).

To ensure that these tutorials are both educational for users and executable in a Continuous Integration (CI) environment, we will employ a **Dual-Mode Execution Strategy**.

### 1.1 Dual-Mode Execution (Mock vs Real)

Every Jupyter Notebook in the tutorial series will implement a `IS_CI_MODE` flag at the beginning.

*   **Real Mode (`IS_CI_MODE = False`)**:
    *   **Goal**: Demonstrate the full scientific capability.
    *   **Compute**: Runs actual DFT (Quantum Espresso), Molecular Dynamics (LAMMPS), and Training (Pacemaker).
    *   **Scale**: Uses realistic system sizes (e.g., 100+ atoms), extensive sampling (1000s of steps), and tight convergence criteria.
    *   **Time**: May take hours to run.

*   **Mock/CI Mode (`IS_CI_MODE = True`)**:
    *   **Goal**: Verify the *workflow logic* and *API integrity*.
    *   **Compute**:
        *   **DFT**: Uses a `MockOracle` that returns pre-calculated energies/forces or simple Lennard-Jones approximations instead of running QE.
        *   **Dynamics**: Runs very short MD (e.g., 10 steps) or uses `MockDynamics` that instantly returns a "Halt" signal to test the active learning loop.
        *   **Training**: Runs Pacemaker for 1 epoch on a tiny dataset.
    *   **Scale**: Uses minimal unit cells (e.g., 2 atoms), loose thresholds.
    *   **Time**: Must finish within 5-10 minutes.

### 1.2 "Divide and Conquer" Philosophy

Scientific validity is paramount. We will not attempt to learn a complex ternary system (Mg-O-Fe-Pt) in one go. The tutorials will guide the user through a "Divide and Conquer" approach:
1.  **Substrate**: Learn MgO bulk and surface.
2.  **Cluster**: Learn Fe-Pt bulk and clusters.
3.  **Interface**: Learn the interaction between the Cluster and Substrate (Adhesion).
4.  **Production**: Run the deposition simulation.

## 2. Notebook Plan

We will generate four (4) sequential Jupyter Notebooks in the `tutorials/` directory.

### `tutorials/01_MgO_FePt_Training.ipynb`
**Title**: Foundations: Training Component Potentials
**Objective**: Generate and train separate potentials for the substrate (MgO) and the deposit (FePt).
**Key Concepts**:
*   `StructureGenerator`: Creating bulk and surface structures.
*   `AdaptiveExplorationPolicy`: How the system decides to sample high-temperature liquids vs. low-temperature distorted crystals.
*   `Trainer`: Active Learning loop (Generation -> Uncertainty Check -> DFT -> Train).
**Validation**:
*   Show learning curves (RMSE Energy/Force).
*   Verify MgO lattice parameter matches experiment.

### `tutorials/02_Interface_Learning.ipynb`
**Title**: The Interface: Adhesion and Interaction
**Objective**: Learn the interaction between the metal cluster and the oxide support.
**Key Concepts**:
*   **Structure Construction**: Placing Fe/Pt clusters on MgO(001) slabs.
*   **Active Learning for Interfaces**: Focusing sampling on the interface region.
*   **Hybrid Potentials**: The importance of ZBL/LJ baselines for preventing fusion at the interface.
**Validation**:
*   Calculate the **Adhesion Energy** ($E_{adh} = E_{total} - (E_{slab} + E_{cluster})$).
*   Ensure no "holes" in the potential surface (no unphysical attraction).

### `tutorials/03_Deposition_MD.ipynb`
**Title**: Dynamics: Atom-by-Atom Deposition
**Objective**: Simulate the growth of FePt nanoparticles using Molecular Dynamics.
**Key Concepts**:
*   `LAMMPS` Integration: Using `fix deposit`.
*   **Hybrid/Overlay**: `pair_style hybrid/overlay pace zbl`.
*   **On-the-Fly (OTF) Monitoring**: The system watches for "high uncertainty" events during deposition and triggers retraining if necessary (simulated in CI mode).
**Validation**:
*   Visualisation of the deposition process (atoms landing and diffusing).
*   Check that atoms do not penetrate the substrate (Core Repulsion check).

### `tutorials/04_Ordering_aKMC.ipynb`
**Title**: Long-Term Evolution: Ordering via aKMC
**Objective**: Observe the chemical ordering (A1 -> L10 phase transformation) which occurs on timescales inaccessible to MD.
**Key Concepts**:
*   `EON` Integration: Bridging the gap between MD and kMC.
*   **State Search**: Finding saddle points using the Dimer method with ACE potentials.
*   **Time Scale**: Comparing MD time (ns) vs. kMC time (s/hours).
**Validation**:
*   **Order Parameter**: Calculate the number of Fe-Pt bonds vs. Fe-Fe/Pt-Pt bonds.
*   Observe the formation of an L10-like layered structure.

## 3. Validation Steps for QA

The QA Agent (and the human user) should verify the following points when running these notebooks:

### 3.1 Workflow Integrity
1.  **Dependency Handling**: If `lammps` or `eon` executables are missing (common in generic Python environments), the notebook must **not crash**. It should catch the `FileNotFoundError` and print a friendly warning (e.g., "Skipping actual simulation step due to missing binary"), while still showing the pre-calculated results or logic flow.
2.  **File Outputs**: Verify that each notebook produces the expected artifacts:
    *   `data/potential_mgo.yace`
    *   `data/potential_fept.yace`
    *   `simulation_logs/deposition.log`
    *   `visualization/final_structure.xyz`

### 3.2 Scientific Validity
1.  **Physical Sanity Checks**:
    *   **Energy**: Total energy should be negative and finite.
    *   **Forces**: Maximum force in relaxed structures should be small (< 0.1 eV/Å).
    *   **Geometry**: Bond lengths should be physical (e.g., Mg-O distance approx 2.1 Å). No atoms should be closer than 1.5 Å (except H).
2.  **Visual Confirmation**:
    *   The deposited cluster must sit *on top* of the surface, not inside it.
    *   The Fe and Pt atoms should show some degree of mixing or ordering, not remain as two separate blobs (unless temperature is very low).

### 3.3 CI Performance
1.  **Execution Time**: The entire suite (in `IS_CI_MODE=True`) must complete within **10 minutes** on a standard GitHub Actions runner (2-core).
2.  **Determinism**: The Mock Oracle must return deterministic values so that regression tests are stable.

## 4. Gherkin Behavior Definitions

### Scenario: High Uncertainty Halt
**GIVEN** a running MD simulation of Fe deposition
**AND** the active potential `potential_v1.yace`
**WHEN** an Fe atom approaches the MgO surface with high velocity
**AND** the extrapolation grade $\gamma$ exceeds the threshold (5.0)
**THEN** the simulation should Halt immediately
**AND** the `Orchestrator` should identify the "high-gamma" atoms
**AND** a new DFT calculation job should be triggered for that configuration.

### Scenario: Hybrid Potential Safety
**GIVEN** a structure with an unphysically short distance ($r < 1.0 \AA$) between Fe and O
**WHEN** the energy is calculated using `pair_style hybrid/overlay pace zbl`
**THEN** the energy should be extremely high (positive) dominated by the ZBL term
**AND** the force should be strongly repulsive
**SO THAT** the atoms are pushed apart, preventing simulation collapse.
