# PYACEMAKER Master UAT Plan

## 1. Tutorial Strategy

The primary goal of the User Acceptance Testing (UAT) is to demonstrate that the PYACEMAKER system can solve real-world materials science problems—specifically the deposition and ordering of Fe/Pt nanoparticles on MgO substrates—without requiring the user to be a computational expert.

To achieve this, we will convert the scientific workflow defined in the `USER_TEST_SCENARIO.md` into a series of executable Jupyter Notebooks. These notebooks will serve a dual purpose: they are the UAT test cases for the developers and the "Getting Started" tutorials for the end-users.

### Divide & Conquer Approach
Training a potential for a complex hetero-epitaxial interface from scratch is computationally expensive. Therefore, we adopt a "Divide & Conquer" strategy:
1.  **Component Training**: First, train robust potentials for the bulk materials (MgO and FePt) separately.
2.  **Interface Learning**: Train the interaction between the two subsystems (Fe/Pt on MgO).
3.  **Application**: Use the combined knowledge to simulate the complex deposition process.

### Mock vs. Real Mode
Scientific simulations can take days to run. To ensure that our Continuous Integration (CI) pipeline and impatient users can verify the workflow quickly, every notebook will implement a `IS_CI_MODE` flag.

*   **Real Mode (`IS_CI_MODE = False`)**: Runs the full physics simulations (large cells, converged DFT, long MD). This is for "Production" runs.
*   **Mock Mode (`IS_CI_MODE = True`)**:
    *   Uses tiny unit cells (e.g., 2 atoms instead of 100).
    *   Uses "Mock Oracles" (pre-calculated values or fast semi-empirical potentials) instead of heavy DFT.
    *   Runs very short MD trajectories (e.g., 100 steps).
    *   Skips long kMC runs and loads a pre-calculated "Success" state to demonstrate the visualization and analysis steps.

## 2. Notebook Plan

We will deliver the following Jupyter Notebooks in the `tutorials/` directory.

### `tutorials/01_MgO_FePt_Training.ipynb` (Phase 1: Foundations)
**Objective**: Train the foundational potentials for the substrate and the nanoparticle alloy.
*   **Scenario ID**: UAT-001
*   **Workflow**:
    1.  **MgO Substrate**:
        *   Generate random distorted MgO structures (bulk and surface).
        *   Run Active Learning loop to train a potential for MgO.
        *   Verify stability (Phonons/EOS).
    2.  **FePt Alloy**:
        *   Generate Fe, Pt, and L10-FePt alloy structures.
        *   Train a potential focusing on phase stability and surface segregation.
*   **Key Outcome**: Two distinct `.yace` potential files (or one combined multi-element potential).

### `tutorials/02_Interface_Learning.ipynb` (Phase 2: The Interface)
**Objective**: Learn the adhesion physics between the metal and the oxide.
*   **Scenario ID**: UAT-002
*   **Workflow**:
    1.  **Interface Builder**: Place small Fe and Pt clusters on an MgO(001) slab.
    2.  **Active Learning**: Run MD at the interface. Detect high forces/uncertainty when atoms land.
    3.  **Refinement**: Trigger DFT calculations on interface clusters to learn the correct adhesion energy.
*   **Key Outcome**: A robust potential that doesn't explode when Fe atoms hit the MgO surface.

### `tutorials/03_Deposition_and_Ordering.ipynb` (Phase 3: The Grand Challenge)
**Objective**: Simulate the full deposition and ordering process.
*   **Scenario ID**: UAT-003
*   **Workflow**:
    1.  **Setup**: Load the trained potential. Configure a large MgO slab.
    2.  **Deposition (MD)**:
        *   Use LAMMPS `fix deposit` to drop Fe and Pt atoms alternately.
        *   Visualize atoms landing and diffusing on the surface (hot deposition).
    3.  **Ordering (aKMC)**:
        *   Take the final disordered cluster from MD.
        *   Hand off to EON (or an internal kMC driver) to simulate long-term ordering.
        *   Observe the formation of the chemically ordered L10 phase.
*   **Key Outcome**: A trajectory showing the formation of an ordered nanoparticle.

## 3. Validation Steps

The QA Agent (or human reviewer) should verify the following when running these notebooks:

### General Checks
*   [ ] **No Crashes**: Notebooks must run from top to bottom without error in `IS_CI_MODE`.
*   [ ] **Dependency Handling**: If `lammps` or `quantum-espresso` binaries are missing, the notebook should gracefully degrade (skip cells or use mocks) rather than crashing with a raw `FileNotFoundError`.

### Scientific Validity Checks
*   [ ] **Energy Conservation**: In MD cells, total energy should be stable (drift < 0.1 meV/atom/ps).
*   [ ] **Geometry**:
    *   MgO lattice constant should be ~4.21 Å.
    *   Fe-Pt bond lengths should be physically reasonable (approx 2.6-2.7 Å).
*   [ ] **Adhesion**: Fe/Pt atoms should stick to the MgO surface, not fly away or sink into the bulk.

### Visualization Checks
*   [ ] **Plots**: Verify that `matplotlib` plots (EOS curves, Parity plots) are generated and visible inline.
*   [ ] **Structure View**: Verify that atomic structures are visualized (using `ase.visualize.plot` or similar).

### Artifacts
*   [ ] **Output Files**: Ensure that `.yace` potential files and `log.lammps` files are created in the working directory.
