# Final User Acceptance Testing (UAT) Plan

## 1. Tutorial Strategy: "The Executable Scientific Paper"

The UAT strategy is centred around providing users with **Jupyter Notebooks** that act as "Executable Scientific Papers". These tutorials serve a dual purpose:
1.  **Onboarding**: They guide new users through the "Fe/Pt on MgO" scenario, explaining the *why* (physics) and *how* (code) simultaneously.
2.  **Verification**: They serve as end-to-end integration tests for the system.

### 1.1 The "Divide & Conquer" Workflow
To prevent user frustration with long-running jobs, the tutorials follow a component-based approach:
*   **Step 1: Component Training**: Train potentials for bulk MgO and bulk FePt separately.
*   **Step 2: Interface Training**: Train the specific interaction (adhesion) between the cluster and substrate.
*   **Step 3: Application**: Use the trained potentials for the complex deposition and ordering simulation.

### 1.2 "Mock Mode" vs "Real Mode"
Scientific simulations can take days. To ensure these tutorials are usable in a CI/CD environment or on a laptop:
*   **Mock Mode (CI / Quickstart)**:
    *   Activated by `IS_CI_MODE = True`.
    *   Uses tiny supercells (e.g., 2 atoms).
    *   Runs for very few steps (e.g., 10 MD steps).
    *   **Crucially**: Mocks long-running external calls (like `eonclient`) or uses pre-calculated data to "fast-forward" to the result.
*   **Real Mode (User)**:
    *   Uses production-grade settings (large slabs, long timescales).
    *   Produces publication-quality results.

## 2. Notebook Plan

The following notebooks will be generated in the `tutorials/` directory.

### 2.1 `tutorials/01_MgO_FePt_Training.ipynb`
**Title**: "Foundation: Training Bulk Potentials"
**Goal**: Learn how to use the `Orchestrator` to train potentials for pure materials.
**Scenario**:
1.  Define `GlobalConfig` for MgO (Insulator) and FePt (Alloy).
2.  Run the Active Learning loop.
3.  **Key Observation**: See how the system automatically handles `O` (Oxygen) pseudopotentials and `Fe` spin polarisation.
**Deliverable**: Two potential files: `mgo_bulk.yace` and `fept_bulk.yace`.

### 2.2 `tutorials/02_Interface_Learning.ipynb`
**Title**: "The Interface: Learning Adhesion"
**Goal**: Train the interaction between the metal cluster and the oxide support.
**Scenario**:
1.  Load the pre-trained bulk potentials.
2.  Generate "Cluster on Slab" structures.
3.  Run Active Learning specifically on the interface region.
**Deliverable**: A merged potential `FePt_MgO_interface.yace` that handles the hetero-interface correctly.

### 2.3 `tutorials/03_Deposition_and_Ordering.ipynb`
**Title**: "Grand Challenge: Deposition & Ordering"
**Goal**: Simulate the growth of FePt nanoparticles and their ordering into the L10 phase.
**Scenario**:
1.  **MD Phase**: Use LAMMPS `fix deposit` to drop Fe and Pt atoms onto the MgO substrate at 600K.
    *   *Validation*: Atoms must land and diffuse, not fly away or fuse.
2.  **Halt & Train**: If the deposition creates a high-energy configuration, the system halts and retrains (Demonstrating OTF learning).
3.  **kMC Phase**: Pass the deposited cluster to EON (aKMC) to find the L10 ordered state.
**Deliverable**: A trajectory file showing the transition from disordered alloy to ordered L10 crystal.

## 3. Validation Steps

The QA Agent (and the CI system) will validate the success of these tutorials based on the following criteria.

### 3.1 Automated Checks (CI)
*   **Exit Code**: All cells must execute with exit code 0.
*   **Output Files**:
    *   `*.yace` files must exist in the output directory.
    *   `validation_report.html` must be generated.
*   **Physics Assertions**:
    *   `final_energy < 0` (System is stable).
    *   `min_distance > 1.5 Å` (No core collapse).

### 3.2 Visual Verification (Human/QA)
*   **Adhesion**: Fe/Pt atoms should stick to the MgO surface, not float above it.
*   **Ordering**: The final structure in NB03 should show a checkerboard pattern of Fe and Pt atoms (L10 ordering), or at least a tendency towards it compared to the random deposition.
*   **Graphs**: The `validation_report.html` should show Parity Plots with RMSE < 5 meV/atom for the training set.

### 3.3 Dependencies
The tutorials must gracefully handle missing dependencies (e.g., if `pw.x` or `eonclient` is not in the PATH).
*   **Requirement**: "If tool missing -> Print Warning -> Skip Cell -> Continue".
*   This ensures users can read the code and understand the flow even without a full HPC setup.
