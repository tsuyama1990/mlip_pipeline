# Final User Acceptance Test (UAT) Plan

## 1. Tutorial Strategy

The goal of the UAT is to prove that **PYACEMAKER** can solve a "Grand Challenge" problem in materials science: the hetero-epitaxial growth and ordering of FePt nanoparticles on an MgO substrate. This complex multi-scale problem is broken down into digestible "executable papers" (Jupyter Notebooks) that guide the user from zero to a full simulation.

### 1.1. The "Divide & Conquer" Philosophy
Instead of training a universal potential from scratch (which takes weeks), we adopt a modular approach:
1.  **Phase 1 (Component Training)**: Train potentials for Bulk MgO and Bulk FePt separately.
2.  **Phase 2 (Interface Learning)**: Learn the interaction (adhesion) between the two phases.
3.  **Phase 3 (Application)**: Use the combined knowledge to simulate deposition and ordering.

### 1.2. Dual-Mode Execution (CI vs. Real)
To ensure the tutorials are robust and testable in our CI/CD pipeline, every notebook must support a "Dual-Mode" flag:
*   `IS_CI_MODE = os.getenv("CI", "False").lower() == "true"`

| Feature | CI Mode (Mock/Fast) | Real Mode (Production) |
| :--- | :--- | :--- |
| **System Size** | Tiny Supercells (2x2x1), < 10 atoms | Large Slabs (10x10x5), > 500 atoms |
| **DFT/Oracle** | Mock (Pre-calculated or LJ-based) | Real Quantum Espresso (SCF) |
| **Training** | 1 Epoch, tiny dataset | 500+ Epochs, full active set |
| **MD Steps** | 100 steps (verify no crash) | 1,000,000+ steps (verify physics) |
| **kMC** | Mock (Load "After" structure) | Real EON execution |

## 2. Notebook Plan

The tutorials will be located in the `tutorials/` directory.

### **NB01: Pre-Training Components (The Foundation)**
*   **Objective**: Generate and train independent potentials for the substrate (MgO) and the deposit (FePt).
*   **Key Concepts**: `StructureGenerator`, `ActiveLearningLoop`, `ParityPlot`.
*   **Workflow**:
    1.  Define `StructureGenerator` for MgO (ionic oxide).
    2.  Run a short active learning loop (Gen -> DFT -> Train).
    3.  Define `StructureGenerator` for FePt (metallic alloy).
    4.  Run active learning.
    5.  **Output**: `mgo_potential.yace`, `fept_potential.yace`.

### **NB02: Interface Learning (The Bridge)**
*   **Objective**: Teach the potential how Fe/Pt atoms interact with the MgO surface.
*   **Key Concepts**: `PeriodicEmbedding`, `InterfaceConstruction`.
*   **Workflow**:
    1.  Load pre-trained potentials.
    2.  Construct Fe/Pt clusters on an MgO slab.
    3.  Run DFT on these interface structures.
    4.  Refine the potential to minimise adhesion energy error.
    5.  **Output**: `combined_potential.yace` (capable of handling all species).

### **NB03: Deposition Dynamics (The Simulation)**
*   **Objective**: Simulate the physical vapor deposition (PVD) process using LAMMPS.
*   **Key Concepts**: `pair_style hybrid/overlay`, `fix deposit`, `CoreRepulsion`.
*   **Workflow**:
    1.  Setup LAMMPS with the `combined_potential.yace`.
    2.  **Crucial**: Configure `pair_style hybrid/overlay pace zbl` to prevent nuclear fusion during high-energy impacts.
    3.  Use `fix deposit` to drop Fe and Pt atoms alternately at 600K.
    4.  Visualise the formation of disordered islands.
    5.  **Output**: `deposition_trajectory.dump`, `final_disordered.data`.

### **NB04: Long-Term Ordering (The Scale-Up)**
*   **Objective**: Overcome MD time-scale limitations using Adaptive Kinetic Monte Carlo (aKMC) to observe L10 ordering.
*   **Key Concepts**: `EON`, `TimeScaleProblem`, `OrderParameter`.
*   **Workflow**:
    1.  Take `final_disordered.data` from NB03.
    2.  Initialise EON (Client/Server) with the potential.
    3.  Run saddle point searches (Dimer method) to find diffusion pathways.
    4.  Evolve the system over "simulated seconds/hours".
    5.  **Analysis**: Calculate the L10 order parameter (Fe-Pt bond count vs. Fe-Fe bond count).
    6.  **Output**: `ordered_structure.xyz`, `energy_landscape.pdf`.

## 3. Validation Steps

The QA Agent (or human reviewer) must verify the following in the generated notebooks:

1.  **Execution without Errors**: All cells must run sequentially without raising exceptions (in CI mode).
2.  **Visualisation Presence**:
    *   NB01: Parity plots showing RMSE convergence.
    *   NB03: Snapshot of atoms *on* the surface (not inside).
    *   NB04: Plot showing energy decrease over kMC steps.
3.  **Physical Validity Checks (Assertions)**:
    *   `assert final_energy < initial_energy` (System relaxes).
    *   `assert min_interatomic_distance > 1.5` (ZBL is working).
    *   `assert ordering_parameter increases` (in Real Mode results).
4.  **Self-Contained**: The notebooks must install necessary python dependencies (via `!pip install`) or clearly state prerequisites.
