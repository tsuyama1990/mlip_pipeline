# Final User Acceptance Testing (UAT) Plan

## 1. Tutorial Strategy: "Executable Scientific Papers"

The core of our UAT strategy is to provide users with **Jupyter Notebooks** that serve a dual purpose: they are step-by-step tutorials for learning the system and rigorous acceptance tests for verifying its functionality.

### 1.1. The "Mock vs Real" Philosophy
Scientific simulations are computationally expensive. A full Active Learning campaign for Fe-Pt on MgO can take days on a cluster. To make our UATs executable in a Continuous Integration (CI) environment (and by users on laptops), we implement a strict **"Mock Mode"**.

*   **Real Mode (Production)**:
    *   **Trigger**: Default behaviour when API keys (if any) and external binaries (QE, LAMMPS) are fully available and `CI` env var is not set.
    *   **Scale**: Full supercells ($4 \times 4 \times 4$), high DFT cutoffs, long MD runs (1 ns+).
    *   **Outcome**: Publication-quality data.

*   **Mock Mode (CI/Demo)**:
    *   **Trigger**: `os.getenv("CI", "False").lower() == "true"` or missing external dependencies.
    *   **Scale**: Tiny unit cells, minimal k-points, very short MD (10 steps).
    *   **Mocking**:
        *   **DFT**: If Quantum Espresso is missing, the Oracle returns energies based on a simple Lennard-Jones potential + random noise to simulate "learning".
        *   **Training**: `pacemaker` runs for 1 epoch.
        *   **kMC**: Instead of running EON (which might take hours), load a pre-calculated "Ordering Event" log and visualise it.
    *   **Outcome**: Verifies that the *pipeline* (Orchestrator logic, file I/O, data conversion) works correctly without waiting for physics.

## 2. Notebook Plan

We will deliver two comprehensive notebooks that cover the entire lifecycle defined in the `USER_TEST_SCENARIO.md`.

### 2.1. Notebook 01: The Foundation - Active Learning
**File**: `tutorials/01_MgO_FePt_Training.ipynb`
**Goal**: Demonstrate how to build a potential from scratch for a complex heterogeneous system.

**Workflow Steps:**
1.  **Setup & Config**: Initialize the `GlobalConfig` in Python. Explain the "Zero-Config" philosophy.
2.  **Phase A: Bulk & Surface (MgO)**:
    *   Define MgO structure.
    *   Run `Orchestrator` to learn bulk equation of state and surface reconstruction.
    *   *Visualisation*: Plot Energy vs Volume (EOS) for DFT vs ACE.
3.  **Phase B: Alloy System (FePt)**:
    *   Define FePt L10 structure.
    *   Run Active Learning to capture disordered vs ordered phases.
    *   *Visualisation*: Parity plot (RMSE check).
4.  **Phase C: The Interface**:
    *   Place a small FePt cluster on an MgO slab.
    *   Demonstrate **Periodic Embedding**: Show how the system cuts the cluster for DFT.
    *   Refine the potential to learn adhesion energy.

**Success Criteria:**
*   Notebook completes active learning loop.
*   Generated potential `.yace` file exists.
*   Validation plot shows RMSE < 10 meV/atom (in Mock mode, this may be higher, but code must pass).

### 2.2. Notebook 02: The Experiment - Deposition & Ordering
**File**: `tutorials/02_Deposition_and_Ordering.ipynb`
**Goal**: Demonstrate the "Dynamics Engine" and "Scale-Up" capabilities (MD + kMC).

**Workflow Steps:**
1.  **Load Potential**: Load the `.yace` file generated in NB01 (or download a pre-trained one).
2.  **Phase A: Dynamic Deposition (MD)**:
    *   Setup a large MgO slab.
    *   Use `fix deposit` to rain Fe and Pt atoms at 600K.
    *   *Observation*: Demonstrate the **Hybrid Potential** (ACE + ZBL) preventing atoms from crashing through the slab.
    *   *Visualisation*: Snapshot of the deposited island (disordered).
3.  **Phase B: Bridging the Timescale (Bridge to EON)**:
    *   Take the final MD snapshot.
    *   Explain the "Time-Scale Problem" (MD is too slow to see ordering).
    *   Setup EON (kMC) calculation input.
4.  **Phase C: Ordering (aKMC)**:
    *   Run (or Mock) the EON client.
    *   Show the energy landscape descent as atoms swap places to form L10 order.
    *   *Visualisation*: Before/After comparison of Chemical Order Parameter (Fe-Fe vs Fe-Pt bonds).

**Success Criteria:**
*   MD simulation runs without "Lost Atoms" error (proving Core Repulsion works).
*   Ordering logic (kMC bridge) executes without syntax errors.
*   Final visualisation shows a stable cluster on the surface.

## 3. Validation Steps for QA Agent

The QA/CI agent must perform the following checks on the generated notebooks:

1.  **Execution Check**:
    ```bash
    export CI=true
    pytest --nbval tutorials/01_MgO_FePt_Training.ipynb
    pytest --nbval tutorials/02_Deposition_and_Ordering.ipynb
    ```
    (Using `nbval` plugin to execute notebooks as tests).

2.  **Artifact Check**:
    *   Ensure `tutorials/outputs/potential.yace` is created.
    *   Ensure `tutorials/outputs/deposition.dump` exists.

3.  **Scientific Sanity Check (grep-based)**:
    *   Search logs for "Core Repulsion Active" or similar confirmations.
    *   Search for "Uncertainty Halt" to verify the watchdog is functioning (even if mocked).

4.  **Visual Check**:
    *   Ensure `.png` files are generated in the notebook outputs (Parity plots, structure snapshots).
