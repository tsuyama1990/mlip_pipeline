# Final User Acceptance Test (UAT) Plan

## 1. Tutorial Strategy

The UAT for PYACEMAKER is designed around the concept of "Executable Scientific Papers". Instead of dry test scripts, we provide Jupyter Notebooks that guide the user through a realistic and scientifically challenging scenario: **Fe/Pt Nanoparticle Growth on MgO(001)**.

### 1.1 The "Mock vs Real" Strategy
Since scientific simulations (DFT, MD training) can take days on HPC clusters, we implement a dual-mode execution strategy to ensure the tutorials are verifiable in a CI/CD environment (GitHub Actions) within minutes.

*   **Mock Mode (CI Mode)**:
    *   Triggered when `CI=true` environment variable is set.
    *   Uses tiny supercells (e.g., 2x2x1 MgO).
    *   Replaces heavy DFT calculations with a fast "Toy Potential" or pre-calculated lookups.
    *   Reduces MD/kMC steps to minimal values (e.g., 10 steps).
    *   **Goal**: Verify that the *code* runs without errors, files are created, and the pipeline logic is sound.

*   **Real Mode (User Mode)**:
    *   The default behaviour when running locally or on a cluster.
    *   Uses production-grade settings (large slabs, full DFT parameters).
    *   Runs for hours/days.
    *   **Goal**: produce publishable scientific data.

## 2. Notebook Plan

We will deliver two core tutorials that cover the entire lifecycle of MLIP construction and application.

### Tutorial 01: Divide & Conquer Training
**File**: `tutorials/01_MgO_FePt_Training.ipynb`

**Objective**: Train high-accuracy potentials for the substrate (MgO) and the deposit (FePt) separately, then learn their interface. This demonstrates the "Active Learning" and "Structure Generator" capabilities.

**Scenario Steps**:
1.  **MgO Bulk & Surface**: Generate MgO structures with vacancies. Run Active Learning to train a baseline potential.
2.  **FePt Alloy**: Generate Fe, Pt, and FePt alloy structures. Train a metallic potential.
3.  **Interface Learning**: Place Fe/Pt clusters on an MgO slab. Use Active Learning to capture the adhesion energy and interface forces.

**Validation Criteria**:
*   The potential must predict the correct lattice constant of MgO within 1%.
*   The adhesion energy of Fe on MgO should be negative (stable binding).
*   **CI Check**: The notebook runs to completion and generates `potential_mgo_fept.yace`.

### Tutorial 02: Dynamic Deposition & Ordering (Hybrid MD + kMC)
**File**: `tutorials/02_Deposition_and_Ordering.ipynb`

**Objective**: Simulate the physical vapour deposition (PVD) of Fe and Pt atoms onto the MgO substrate and observe their ordering into the L10 phase using a combination of MD and Adaptive Kinetic Monte Carlo (aKMC).

**Scenario Steps**:
1.  **Deposition (MD)**: Use LAMMPS `fix deposit` to drop Fe and Pt atoms onto the heated MgO substrate (600K).
    *   *Observation*: Atoms diffuse and form disordered islands.
    *   *Mechanism*: Use `pair_style hybrid/overlay pace zbl` to prevent nuclear fusion upon high-velocity impact.
2.  **Ordering (aKMC)**: Take the final disordered structure from MD and pass it to EON (aKMC engine).
    *   *Observation*: Overcome the time-scale bottleneck. Observe Fe and Pt atoms swapping places to maximise chemical order (L10 formation).
    *   *Bridge*: A Python script seamlessly transfers the state between LAMMPS and EON.

**Validation Criteria**:
*   **Visual**: The notebook produces snapshots showing islands on the surface, not atoms sinking *into* the substrate (Core Repulsion check).
*   **Ordering Parameter**: A calculated order parameter (e.g., number of Fe-Pt bonds vs Fe-Fe bonds) increases after the kMC stage.
*   **CI Check**: The bridging script works, and EON (or its mock) executes without crashing.

## 3. Validation Steps

The QA Agent (or human reviewer) should perform the following checks:

1.  **Environment Setup**:
    *   Run `uv sync --dev`.
    *   Ensure `lammps` and `pace_train` are in the PATH (or mocked).

2.  **Execution**:
    *   Run `pytest` to verify unit tests pass.
    *   Execute the notebooks using `jupyter nbconvert --to python --execute --ExecutePreprocessor.timeout=600 tutorials/*.ipynb`.

3.  **Artifact Verification**:
    *   Check for the existence of `potential.yace` files in the output directories.
    *   Check for `log.lammps` and `results.dat` files.

4.  **Scientific Validity (Manual/Expert Check)**:
    *   Does the potential energy curve look smooth?
    *   Are the forces in the "Interface Learning" phase physically reasonable (< 10 eV/Å)?
