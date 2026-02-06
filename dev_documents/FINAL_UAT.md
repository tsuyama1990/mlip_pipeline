# Final User Acceptance Testing (UAT) Plan

## 1. Tutorial Strategy

The ultimate test of the PYACEMAKER system is its ability to guide a user through a complex, scientifically relevant workflow without overwhelming them with implementation details. We will achieve this by delivering "Executable Scientific Papers" in the form of Jupyter Notebooks.

### 1.1. The "Dual-Mode" Execution Strategy
Scientific simulations often require High Performance Computing (HPC) resources and take days to complete. However, our Continuous Integration (CI) and User Acceptance Testing (UAT) must complete in minutes. To resolve this conflict, every tutorial notebook must implement a **Dual-Mode** strategy controlled by an environment variable.

*   **Mock Mode (CI Mode)**:
    *   **Trigger**: `os.environ.get("CI") == "true"` or `IS_CI_MODE = True`.
    *   **Behavior**:
        *   Uses "Tiny" supercells (e.g., 2 atoms instead of 200).
        *   Replaces expensive DFT calls with a "Mock Oracle" (returning pre-defined or random valid forces).
        *   Runs MD for minimal steps (e.g., 10 steps) just to verify the API calls work.
        *   Skips long waits (e.g., kMC) by loading a pre-calculated "Final State" file to demonstrate the visualization.
    *   **Goal**: Verify that the *code* is broken, not that the *physics* is converged.

*   **Real Mode (Production Mode)**:
    *   **Trigger**: Default behavior when user runs locally.
    *   **Behavior**:
        *   Uses full-size simulation cells.
        *   Calls actual binaries (Quantum Espresso, LAMMPS, EON).
        *   Runs until convergence criteria are met.
    *   **Goal**: Produce publication-quality results.

### 1.2. The Scenario: Fe-Pt Nanoparticle Growth on MgO
The core UAT scenario is the hetero-epitaxial growth of L10-ordered FePt alloy nanoparticles on an MgO(001) substrate. This covers all aspects of the system:
*   **Multicomponent**: Fe, Pt, Mg, O.
*   **Interfaces**: Metal on Oxide.
*   **Timescales**: Deposition (MD) and Ordering (kMC).

## 2. Notebook Plan

We will deliver two comprehensive notebooks in the `tutorials/` directory.

### Notebook 01: `tutorials/01_MgO_FePt_Training.ipynb`
**Title**: "From Zero to Potential: Active Learning for MgO and FePt"
**Objective**: Demonstrate the "Zero-Config" training pipeline.
**Key Steps**:
1.  **Initialization**: Define the elements (`Mg`, `O`, `Fe`, `Pt`) in `config.yaml` via the Python API.
2.  **MgO Phase**:
    *   Generate MgO rock-salt structure.
    *   Run the Orchestrator to learn the bulk and (001) surface properties.
    *   *CI Shortcut*: Load a pre-trained `mgo_tiny.yace`.
3.  **FePt Phase**:
    *   Generate Fe, Pt, and L10-FePt structures.
    *   Run Active Learning to capture the alloy formation energy.
    *   *CI Shortcut*: Limit to 1 active learning cycle.
4.  **Interface Phase**:
    *   Place an Fe cluster on an MgO slab.
    *   Demonstrate "Periodic Embedding" (cutting the cluster region for DFT).
    *   Learn the adhesion forces.
5.  **Validation**:
    *   Plot the "Parity Plot" (DFT vs ACE) for the final dataset.
    *   Check phonon stability of MgO.

### Notebook 02: `tutorials/02_Deposition_and_Ordering.ipynb`
**Title**: "Bridging Scales: MD Deposition and kMC Ordering"
**Objective**: Demonstrate the scale-up capabilities (LAMMPS + EON).
**Key Steps**:
1.  **Setup**: Load the potential generated in NB01.
2.  **MD Deposition (LAMMPS)**:
    *   Initialize an MgO substrate at 600K.
    *   Use `fix deposit` to drop Fe and Pt atoms alternately.
    *   Visualize the "Hot" atoms diffusing on the surface.
    *   *Physics Check*: Ensure atoms don't penetrate the substrate (ZBL check).
3.  **Halt & Handover**:
    *   Take the final disordered cluster from MD.
    *   Pass the configuration to the EON wrapper.
4.  **Ordering (aKMC)**:
    *   Run Adaptive kMC to explore long-timescale rearrangement.
    *   Observe the transition from a random alloy to an L10 chemically ordered structure (layered Fe/Pt).
    *   *CI Shortcut*: Mock the EON run; load `ordered_fept.con` and display it.
5.  **Analysis**:
    *   Calculate the "Chemical Order Parameter" (bond counting).
    *   Visualize the final nanoparticle using `ase.visualize.plot`.

## 3. Validation Steps

The QA Agent or Auditor should perform the following checks when reviewing the notebooks.

### 3.1. Automated Checks (CI)
*   [ ] **Execution Time**: The entire notebook must run in under 10 minutes in CI Mode.
*   [ ] **Dependency Handling**: If `lammps` or `pw.x` are not found in the path, the notebook must strictly follow the "Dual-Mode" logic (skip execution or use mocks) and **not crash**.
*   [ ] **Assertions**:
    *   `assert potential.final_rmse_energy < 0.05` (or a lenient value for mock data).
    *   `assert atom_count == expected_count`.

### 3.2. Manual/Visual Checks
*   [ ] **Clarity**: Are the markdown cells explaining *why* we are doing this step?
*   [ ] **Visualization**: Does the final plot clearly show a cluster sitting *on top* of the substrate, not inside it?
*   [ ] **Ordering**: In NB02, is there a visual distinction between the "disordered" initial state and the "ordered" final state (e.g., color-coding atoms by species)?
