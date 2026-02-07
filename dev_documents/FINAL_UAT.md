# FINAL UAT: User Acceptance Testing & Tutorial Plan

## 1. Tutorial Strategy

The "Grand Challenge" for PYACEMAKER is to enable users to simulate the **Hetero-epitaxial Growth of FePt magnetic nanoparticles on an MgO substrate**. This complex, multi-scale problem requires not only a high-quality machine learning potential (MLIP) but also the integration of Molecular Dynamics (MD) for deposition and Kinetic Monte Carlo (kMC) for long-term ordering.

Our tutorial strategy focuses on breaking down this scientific workflow into executable, verified steps ("Executable Papers"). We adopt a **"Mock vs Real"** approach to ensure that these tutorials are valuable both as quick demonstrations and as templates for serious research.

### Strategy for "Mock Mode" vs "Real Mode"
To ensure the tutorials are runnable in Continuous Integration (CI) environments and on standard laptops, we introduce a toggle:

*   **Mock Mode (CI / Demo)**:
    *   **Trigger**: `IS_CI_MODE = os.getenv("CI", "False").lower() == "true"`
    *   **Behaviour**:
        *   **Skip Heavy DFT**: Instead of calling Quantum Espresso, the `MockOracle` is used (or pre-calculated data is loaded).
        *   **Tiny Systems**: Use minimal unit cells (e.g., 2x2x1 MgO slab) and very few deposited atoms (e.g., 5 atoms).
        *   **Shortened Dynamics**: Run MD for only 100 steps and kMC for 5 steps.
        *   **Goal**: Verify that the *code logic* and *data flow* are correct without waiting for hours.
*   **Real Mode (Production)**:
    *   **Trigger**: User explicitly sets `IS_CI_MODE = False`.
    *   **Behaviour**:
        *   **Full Physics**: Execute actual DFT calculations via `DFTManager`, full MD runs (1000+ atoms, ns timescales), and extensive kMC exploration.
        *   **Goal**: Produce publication-quality results.

## 2. Notebook Plan

We will deliver two core Jupyter Notebooks in the `tutorials/` directory.

### `tutorials/01_MgO_FePt_Training.ipynb`
**Title**: "Divide & Conquer: Training a Multi-Component MLIP for FePt/MgO"

**Objective**: Demonstrate the active learning workflow to generate a potential that describes both the substrate (MgO), the nanoparticle (FePt), and their interface.

**Content Flow**:
1.  **Setup**: Initialise the `GlobalConfig` for a multi-species system (Fe, Pt, Mg, O).
2.  **Phase 1: Component Training**:
    *   Define `StructureGenerator` rules for bulk MgO (distorted rocksalt) and FePt (A1/L10 phases).
    *   Run a short "Active Learning" loop (using `MockOracle` in demo) to gather data.
3.  **Phase 2: Interface Learning**:
    *   Programmatically construct an Fe/Pt cluster on an MgO(001) surface.
    *   Run the "Oracle" (DFT) on these interface structures to capture adhesion energetics.
4.  **Validation**:
    *   Run `Validator` to check the phonon stability of the MgO substrate.
    *   Check the formation energy of L10-FePt alloy.
5.  **Output**: Save the trained `potential.yace`.

### `tutorials/02_Deposition_and_Ordering.ipynb`
**Title**: "From Deposition to Ordering: Bridging MD and kMC"

**Objective**: Simulate the physical process of nanoparticle growth and ordering using the potential trained in NB01.

**Content Flow**:
1.  **Initialisation**: Load `potential.yace`. Setup the simulation environment.
2.  **Phase 1: Dynamic Deposition (MD)**:
    *   Use `DynamicsEngine` (LAMMPS wrapper) to setup a `fix deposit` simulation.
    *   Inject Fe and Pt atoms alternately onto the heated MgO substrate (600K).
    *   **Key Feature**: Explicitly enable `pair_style hybrid/overlay pace zbl` to demonstrate stability against high-energy impacts.
    *   *Visualisation*: Show snapshots of the growing disordered island.
3.  **Phase 2: Long-Term Ordering (aKMC)**:
    *   Take the final configuration from the MD phase.
    *   Hand over the structure to **EON** (via a Python driver script).
    *   Run an Adaptive KMC simulation to explore the potential energy surface and find lower-energy (ordered) states.
    *   *Visualisation*: Plot the "Chemical Order Parameter" over KMC steps to show the transition to the L10 phase.

## 3. Validation Steps

The automated QA agent or human reviewer must verify the following criteria when running the notebooks:

1.  **Execution Integrity**:
    *   Can the notebook run from top to bottom (`Restart & Run All`) without throwing an exception?
    *   Does it correctly detect the environment (CI vs Local) and print the mode ("Running in Mock Mode" / "Running in Real Mode")?

2.  **Artifact Verification**:
    *   **NB01**: Must produce a valid `potential.yace` file (even a dummy one in Mock mode) and a `validation_report.yaml` (or similar).
    *   **NB02**: Must produce a trajectory file (e.g., `deposition.lammpstrj`) and a plot of the ordering parameter (e.g., `ordering.png`).

3.  **Scientific Sanity Checks (Assertions)**:
    *   **Stability**: The final potential energy of the system must be negative (indicating binding).
    *   **Geometry**: The minimum distance between any two atoms must be greater than 1.5 Å (checking that the Core Repulsion/ZBL is working).
    *   **Interface Adhesion**: The Fe/Pt atoms must remain on the surface of MgO, not fly away (vacuum) or sink into the bulk (unless physical).

4.  **Visual Confirmation**:
    *   The generated images (using `ase.visualise.plot`) should clearly show:
        *   A slab geometry for MgO.
        *   Distinct atoms for Fe (e.g., Gold colour) and Pt (e.g., Grey colour).
        *   Adsorption of the cluster on the surface.
