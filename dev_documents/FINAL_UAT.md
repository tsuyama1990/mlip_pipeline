# Final User Acceptance Testing (UAT) Plan

## 1. Tutorial Strategy: The "Divide & Conquer" Approach

To ensure a smooth learning curve and robust testing, the complex scientific problem of "Fe/Pt Deposition on MgO" is broken down into manageable phases. This strategy allows users to verify each component independently before attempting the full-scale simulation.

### 1.1. Core Philosophy
*   **Scientific Validity**: The tutorial must follow a logically sound scientific workflow (Bulk -> Surface -> Interface -> Dynamic Process).
*   **CI/CD Compatibility**: All tutorials must be executable in a "Mock Mode" within a standard CI environment (GitHub Actions) without requiring heavy computational resources or specialized licenses (VASP/QE).
*   **Visual Confirmation**: The output must provide immediate visual feedback (plots, structure visualizations) to confirm success.

### 1.2. Mock Mode vs. Real Mode
To support automated testing, every notebook will implement a `IS_CI_MODE` flag.

*   **Mock Mode (Default in CI)**:
    *   **Goal**: Verify the *workflow logic* and *API integrity*.
    *   **Implementation**:
        *   Uses tiny supercells (e.g., 2x2x1 MgO).
        *   Deposits a minimal number of atoms (e.g., 5 atoms).
        *   Mocks expensive calls (DFT, specific MD steps) using pre-calculated data or simple analytical functions if the external binary is missing.
        *   Runs in < 5 minutes.
*   **Real Mode (User)**:
    *   **Goal**: Verify the *scientific accuracy*.
    *   **Implementation**:
        *   Uses production-scale systems (e.g., 10x10x4 slab).
        *   Deposits 500+ atoms.
        *   Runs full DFT and lengthy MD/kMC simulations.
        *   May take hours/days.

## 2. Notebook Plan

We will deliver two primary Jupyter Notebooks that cover the entire UAT scenario.

### 2.1. `tutorials/01_MgO_FePt_Training.ipynb`
**Phase 1: Component Training**
*   **Objective**: Train the foundational potentials required for the deposition simulation.
*   **Steps**:
    1.  **MgO Bulk & Surface**:
        *   Generate random structures of MgO.
        *   Run Active Learning to train a potential for the substrate.
        *   *Check*: Validate lattice constant and surface energy.
    2.  **Fe-Pt Alloy**:
        *   Generate Fe-Pt binary structures (focusing on L10 phase).
        *   Run Active Learning.
        *   *Check*: Verify the formation energy of L10 phase.
*   **Success Criteria**:
    *   Notebook completes without error.
    *   Generated `.yace` files are saved to `potentials/`.
    *   Validation plots (Energy vs Volume) show correct trends.

### 2.2. `tutorials/02_Deposition_and_Ordering.ipynb`
**Phase 2 & 3: Interface Learning, Deposition, and Ordering**
*   **Objective**: Simulate the dynamic growth of Fe/Pt nanoparticles on MgO and their subsequent ordering.
*   **Steps**:
    1.  **Interface Learning**:
        *   Place Fe/Pt clusters on the MgO slab (trained in NB01).
        *   Run a short Active Learning loop to capture adhesion forces.
        *   *Check*: Adhesion energy is negative (attractive).
    2.  **Dynamic Deposition (MD)**:
        *   Use LAMMPS `fix deposit` to drop Fe and Pt atoms alternately.
        *   *Observation*: Atoms should adsorb and diffuse, forming islands.
    3.  **Long-Term Ordering (aKMC)**:
        *   Bridge the final MD state to EON.
        *   Run aKMC to simulate long-timescale ordering (L10 formation).
        *   *Observation*: Chemical ordering parameter (Fe-Pt bonds) increases.
*   **Success Criteria**:
    *   **Hybrid Potential**: Successfully combines `pace` (ML) with `zbl` (Physics) using `pair_style hybrid/overlay`.
    *   **Stability**: The simulation does not crash (segmentation fault) during deposition.
    *   **Visuals**: A final snapshot showing distinct Fe/Pt atoms on the MgO surface.

## 3. Validation Steps for QA

The QA Agent (and automated tests) should look for the following indicators of success:

### 3.1. Execution Integrity
*   **Exit Code 0**: All cells in the notebook must execute without raising unhandled exceptions.
*   **Dependency Handling**: If `lammps` or `eon` are missing, the notebook should print a warning and skip the specific cell (or use a mock), rather than crashing.

### 3.2. Physics Sanity Checks
Inside the notebooks, we will embed assertions to catch non-physical results:
*   `assert potential_energy < 0`: The system should be bound.
*   `assert min_distance > 1.5`: Atoms should not overlap (nuclear fusion check).
*   `assert ordering_parameter_final >= ordering_parameter_initial`: For the aKMC step, ordering should generally improve or stay stable.

### 3.3. Artifact Generation
*   **Files**:
    *   `potentials/mgo_fept.yace`: The final trained potential.
    *   `trajectory.lammpstrj`: The deposition movie.
    *   `final_structure.xyz`: The ordered structure.
*   **Images**:
    *   `plots/learning_curve.png`: RMSE vs Generation.
    *   `plots/deposition_snapshot.png`: Visualization of the final state.
