# Final User Acceptance Test (UAT) & Tutorial Plan

## 1. Tutorial Strategy

The primary goal of the UAT is to prove that **PYACEMAKER** can autonomously handle a scientifically complex scenario: **The Hetero-epitaxial Growth of Fe/Pt Magnetic Alloys on an MgO Substrate**. This involves multiple physical regimes—bulk crystal, surface, interface adhesion, and dynamic deposition.

To ensure both usability for new users and reliability for developers, we adopt a **Dual-Mode Strategy**:

### 1.1. Real Mode (The "Aha!" Moment)
*   **Target**: Users with HPC access or powerful workstations.
*   **Data**: Real DFT calculations (via Quantum Espresso).
*   **Time**: Hours to Days.
*   **Outcome**: A scientifically valid potential that reproduces the L10 ordering of FePt nanoparticles on MgO.

### 1.2. Mock Mode (CI/CD Verification)
*   **Target**: GitHub Actions and quick user demos.
*   **Data**: Pre-calculated dataset or Mock Oracle (returning analytical forces).
*   **Time**: < 5 minutes.
*   **Outcome**: Verification that the *code logic* (Active Learning loop, file handling, error recovery) works without crashing.

**Implementation**:
All notebooks will include a `CI_MODE` flag.
```python
import os
CI_MODE = os.getenv("CI", "False").lower() == "true"

if CI_MODE:
    # Use tiny supercell and mock calculator
    atoms = bulk("MgO") * (2, 2, 2)
    steps = 10
else:
    # Use production size
    atoms = read("large_slab.xyz")
    steps = 10000
```

## 2. Notebook Plan

The tutorial is split into two sequential Jupyter Notebooks located in the `tutorials/` directory.

### **Notebook 01: The "Divide & Conquer" Training (`tutorials/01_MgO_FePt_Training.ipynb`)**
*   **Objective**: Train a robust potential by learning the components separately before combining them.
*   **Story**:
    1.  **MgO Substrate**: Train a potential for Bulk MgO and the (001) Surface.
    2.  **Fe-Pt Alloy**: Train for Bulk Fe, Bulk Pt, and L10-FePt phases.
    3.  **Interface Adhesion**: Generate "Cluster on Slab" structures. Run Active Learning to capture the interaction energy between the metal cluster and the oxide surface.
*   **Key Features Demonstrated**:
    *   `StructureGenerator` with chemical constraints.
    *   `Oracle` handling multiple elements.
    *   `Trainer` utilizing Delta Learning (ZBL baseline) for core repulsion.

### **Notebook 02: Dynamic Deposition & Ordering (`tutorials/02_Deposition_and_Ordering.ipynb`)**
*   **Objective**: Demonstrate the application of the potential in a complex multi-physics simulation.
*   **Story**:
    1.  **Deposition (MD)**: Use LAMMPS `fix deposit` to simulate Fe and Pt atoms raining down on the MgO substrate at 600K. Observe the formation of disordered clusters.
    2.  **Ordering (aKMC)**: Take the final structure from MD and pass it to the kMC engine (EON). Simulate the long-timescale diffusion that allows Fe and Pt atoms to rearrange into the chemically ordered L10 phase.
*   **Key Features Demonstrated**:
    *   Loading the `.yace` potential trained in NB01.
    *   `DynamicsEngine` switching between MD and kMC modes.
    *   Visualization of the "Ordering Parameter" (Fe-Pt bonds vs Fe-Fe/Pt-Pt bonds).

## 3. Validation Steps

The **Quality Assurance Agent** (or human reviewer) must verify the following:

### 3.1. Execution Success
*   [ ] **CI Mode**: Both notebooks run from top to bottom without errors in the CI environment (using Mock components).
*   [ ] **Dependency Check**: Notebooks gracefully handle missing external binaries (e.g., `lmp`, `pw.x`) by printing a warning or skipping specific cells, rather than crashing with a traceback.

### 3.2. Scientific Sanity (Real Mode)
*   [ ] **Stability**: The MgO substrate maintains its crystal structure during MD (no melting at 300K).
*   [ ] **Physics**:
    *   Potential Energy < 0 (System is bound).
    *   Nearest neighbor distance > 2.0 Å (No nuclear fusion/overlapping atoms).
*   [ ] **Adhesion**: Fe/Pt atoms stick to the MgO surface, they do not fly away or sink *into* the slab.

### 3.3. Artifact Generation
*   [ ] `tutorials/outputs/potential.yace` exists.
*   [ ] `tutorials/outputs/deposition_movie.xyz` exists.
*   [ ] `tutorials/outputs/ordering_plot.png` shows the evolution of the system energy over time.
