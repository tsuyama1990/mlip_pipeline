# Final UAT: Fe/Pt on MgO Deposition & Ordering

## 1. Tutorial Strategy

The User Acceptance Testing (UAT) for PYACEMAKER is designed around a scientifically relevant "Grand Challenge": **Hetero-epitaxial Growth of FePt magnetic nanoparticles on an MgO substrate**. This scenario covers all aspects of the system: multi-species interactions, surface physics, deposition (non-equilibrium), and ordering (long-timescale evolution).

### 1.1. "Divide & Conquer" Workflow
Instead of a single monolithic script, the tutorial is split into logical scientific phases. This teaches the user the "best practice" of building MLIPs:
1.  **Component Training**: Learn Bulk MgO and Bulk FePt separately.
2.  **Interface Training**: Learn the interaction between FePt and MgO.
3.  **Application**: Run the complex deposition simulation.

### 1.2. Execution Modes (Mock vs Real)
To ensure the UAT is verifiable in a Continuous Integration (CI) environment without requiring a supercomputer, every notebook implements a dual-mode strategy:

*   **CI Mode (`IS_CI = True`)**:
    *   **Data**: Uses tiny supercells (e.g., 2 atoms) or pre-calculated datasets.
    *   **Compute**: Runs 1 training epoch, 10 MD steps.
    *   **Components**: May use `MockOracle` if DFT binaries are missing.
    *   **Goal**: Verify the code paths and API calls are correct.
*   **Real Mode (`IS_CI = False`)**:
    *   **Data**: Uses realistic supercells (100+ atoms).
    *   **Compute**: Runs full convergence (1000 epochs), 1ns MD.
    *   **Components**: Uses real `pw.x`, `pace_train`, `lmp`.
    *   **Goal**: Produce scientifically publishable results.

## 2. Notebook Plan

The UAT consists of two primary Jupyter Notebooks located in `tutorials/`.

### 2.1. `tutorials/01_MgO_FePt_Training.ipynb`
**Title**: "From Zero to Potential: Training the Fe-Pt-Mg-O System"

**Objective**: Demonstrate the **Active Learning Loop**.
*   **Step 1: Initialization**: Define the `GlobalConfig` for a multi-component system.
*   **Step 2: Exploration**: Use `StructureGenerator` to create random bulk and surface structures for MgO and FePt.
*   **Step 3: Labeling**: Run (or mock) DFT calculations.
*   **Step 4: Training**: Train a preliminary ACE potential.
*   **Step 5: Interface Learning**: Explicitly generate "Slab + Cluster" geometries to learn the adhesion energy.
*   **Validation**: Check the "Adhesion Energy" of Pt on MgO against literature/DFT reference.

### 2.2. `tutorials/02_Deposition_and_Ordering.ipynb`
**Title**: "Simulating Growth: Deposition MD and kMC Ordering"

**Objective**: Demonstrate **Dynamics & Inference** capabilities.
*   **Step 1: Setup**: Load the potential trained in NB01.
*   **Step 2: MD Deposition**: Use LAMMPS `fix deposit` to simulate Fe and Pt atoms landing on the MgO substrate.
    *   *Check*: Verify atoms do not penetrate the substrate (Physics Robustness).
*   **Step 3: Ordering (kMC)**: Take the disordered cluster from MD and run a short Kinetic Monte Carlo (kMC) session (or Mock/Load pre-calc) to observe L10 ordering.
*   **Validation**: Visualize the final structure showing distinct Fe/Pt layers or ordered domains.

## 3. Validation Steps (QA)

The QA Agent (and automated CI) will validate the UAT by running the following checks:

1.  **Dependency Check**: Ensure `uv`, `lammps`, `quantum-espresso` (or mocks) are accessible.
2.  **Execution**: Run `uv run jupyter execute tutorials/01_MgO_FePt_Training.ipynb` (in CI mode).
    *   *Pass Criteria*: Exit code 0, no exceptions.
3.  **Artifact Verification**:
    *   Check for existence of `potential.yace`.
    *   Check for `validation_report.html` or similar output.
    *   Check that generated images (e.g., `structure_snapshot.png`) exist.
4.  **Physics Sanity Check**:
    *   Parse the notebook output for "RMSE Energy". It must be positive and reasonable (or explicitly marked as "Mock Data").
    *   Verify that the "Halt" mechanism was triggered at least once during the "Exploration" phase (simulating active learning), or explicitly tested.
