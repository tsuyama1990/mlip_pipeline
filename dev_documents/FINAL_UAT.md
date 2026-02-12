# Final UAT Plan: PYACEMAKER

## 1. Tutorial Strategy

The User Acceptance Testing (UAT) for PYACEMAKER is designed to be dual-purpose:
1.  **Verify System Functionality**: Ensure all components (Structure Generation, Active Learning, Dynamics, Validation) work together seamlessly.
2.  **Educate Users**: Provide "Executable Scientific Papers" that guide users through a complete research workflow, from zero knowledge to publication-quality results.

### 1.1 "Mock Mode" vs "Real Mode"
Scientific simulations are computationally expensive and time-consuming. To ensure our UATs can run in Continuous Integration (CI) environments and on standard laptops, we implement a strict **Dual-Mode Execution Strategy**:

*   **`IS_CI_MODE = True` (Mock Mode)**:
    *   **Data**: Uses tiny supercells (e.g., 2 atoms), pre-calculated DFT data, or empirical potentials (LJ/EAM) as a proxy for expensive DFT.
    *   **Compute**: Runs for minimal steps (e.g., 5 MD steps, 1 kMC step).
    *   **Goal**: Verify code paths, error handling, and data flow without waiting for convergence.
    *   **Constraint**: Must finish within 5 minutes.

*   **`IS_CI_MODE = False` (Real Mode)**:
    *   **Data**: Uses realistic supercells (e.g., 100+ atoms), actual DFT calculations (via QE/VASP if available), and full training epochs.
    *   **Compute**: Runs until physical convergence.
    *   **Goal**: Reproduce scientific results (e.g., FePt L10 ordering).
    *   **Constraint**: May take hours or days (runs on HPC/Workstation).

All notebooks must detect the environment variable `CI` or a user flag to switch modes automatically.

## 2. Notebook Plan

We will deliver two core Jupyter Notebooks in the `tutorials/` directory.

### 2.1 `tutorials/01_MgO_FePt_Training.ipynb`
**Title**: "From Scratch to Active Learning: Training Potentials for Interfaces"
**Goal**: The "Aha! Moment" where a user sees the system automatically learn and improve.

**Workflow**:
1.  **Setup**: Define the system (MgO substrate, FePt cluster).
2.  **Phase A (Bulk)**: Train simple bulk potentials for MgO and FePt separately using `StructureGenerator` (Random + MD).
    *   *CI Mode*: Load pre-trained mini-potentials.
3.  **Phase B (Interface)**: Construct an interface (FePt cluster on MgO slab). Run **Active Learning** to capture adhesion forces.
    *   *Action*: The system detects high uncertainty at the interface.
    *   *Action*: It automatically requests DFT (or mock DFT) for interface configurations.
4.  **Validation**: Inspect the parity plots (Energy/Force) and check the "Physics-Informed" baseline (ZBL) effect at short distances.

**Success Criteria**:
*   The notebook runs from top to bottom without errors.
*   The final potential predicts stable MgO and FePt structures.
*   The `uncertainty_watchdog` is triggered at least once (demonstrating safety).

### 2.2 `tutorials/02_Deposition_and_Ordering.ipynb`
**Title**: "Simulating Growth: Deposition MD and Long-Timescale Ordering (aKMC)"
**Goal**: Demonstrate the system's ability to handle complex, multi-scale physics.

**Workflow**:
1.  **Setup**: Load the potential trained in Tutorial 01.
2.  **Phase A (Deposition MD)**:
    *   Use LAMMPS `fix deposit` to drop Fe and Pt atoms onto the MgO substrate at 600K.
    *   *Observation*: Atoms land, diffuse, and form a disordered cluster.
    *   *CI Mode*: Deposit 5 atoms. *Real Mode*: Deposit 500 atoms.
3.  **Phase B (Ordering aKMC)**:
    *   Take the final MD structure.
    *   Run **Adaptive Kinetic Monte Carlo (aKMC)** via EON to simulate long-term evolution (seconds to hours).
    *   *Observation*: Fe and Pt atoms rearrange into the chemically ordered L10 phase (alternating layers).
4.  **Analysis**:
    *   Calculate the **Order Parameter** (number of Fe-Pt bonds vs Fe-Fe/Pt-Pt).
    *   Visualize the structure using `ase.visualize`.

**Success Criteria**:
*   Seamless handover from LAMMPS (MD) to EON (kMC).
*   Visual proof of clustering (MD) and ordering (kMC).
*   Demonstration of the "Hybrid Potential" preventing nuclear fusion during high-energy deposition.

## 3. Validation Steps for QA Agent

The QA Agent (or human reviewer) should perform the following checks:

1.  **Environment Check**:
    *   `uv sync` installs all dependencies.
    *   `lammps` and `eonclient` binaries are accessible (or mocked properly).

2.  **Execution (CI Mode)**:
    *   Run: `export CI=true`
    *   Run: `pytest tests/uat/test_notebooks.py` (which executes the notebooks using `nbmake` or similar).
    *   **Expectation**: Pass within 10 minutes. No specialized hardware required.

3.  **Execution (Real Mode - Optional)**:
    *   If HPC is available, unset `CI` and run the notebooks interactively.
    *   **Expectation**: Physically meaningful results (L10 ordering observed).

4.  **Artifact Inspection**:
    *   Check `tutorials/outputs/` for:
        *   `potential.yace` (The trained model)
        *   `trajectory.lammpstrj` (The MD movie)
        *   `ordering_plot.png` (The scientific proof)

5.  **Code Quality**:
    *   Notebooks must be clean, with clear Markdown explanations between cells.
    *   No hardcoded paths (use relative paths).
    *   Proper error handling if external tools (QE, LAMMPS) are missing (fallback to "Demo Mode" with warning).
