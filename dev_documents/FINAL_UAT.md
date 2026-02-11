# Final User Acceptance Testing (UAT) Plan: PYACEMAKER

## 1. Tutorial Strategy

The ultimate test of the PYACEMAKER system is its ability to enable a non-expert user to perform a complex, scientifically significant computational experiment. We have selected the **"Hetero-epitaxial Growth and Ordering of FePt Nanoparticles on MgO(001)"** as the Grand Challenge scenario. This problem is scientifically rich—involving deposition (MD), surface diffusion, and chemical ordering (kMC)—and technically demanding, requiring the seamless integration of all system components.

### 1.1. Dual-Mode Execution Strategy
To ensure that these tutorials serve both as educational material for users and as rigorous integration tests for the CI/CD pipeline, every notebook must support two distinct execution modes:

*   **Real Mode (`IS_CI_MODE = False`)**:
    *   **Goal**: Reproduce physical reality.
    *   **Resources**: Uses full DFT (Quantum Espresso), large MD cells (thousands of atoms), and runs for hours/days.
    *   **Target**: Users with HPC access or powerful workstations.
    *   **Outcome**: Publication-quality plots (e.g., correct L10 ordering parameter, accurate adsorption energies).

*   **Mock Mode (`IS_CI_MODE = True`)**:
    *   **Goal**: Verify code paths and API integrity.
    *   **Resources**: Uses "Mock Oracle" (returning random or pre-calculated forces), tiny cells (2-10 atoms), and minimal steps (10 MD steps).
    *   **Target**: GitHub Actions runners and developers checking for regressions.
    *   **Outcome**: The notebook runs from top to bottom without error in < 5 minutes. The physics will be nonsense, but the *process* is validated.

All notebooks must begin with:
```python
import os
IS_CI_MODE = os.getenv("CI", "False").lower() == "true"
```

## 2. Notebook Plan

The UAT will be delivered as a set of two Jupyter Notebooks located in the `tutorials/` directory.

### Notebook 1: `tutorials/01_MgO_FePt_Training.ipynb`
**Title:** "Divide & Conquer: Active Learning of Multicomponent Systems"
**Objective:** Teach the user how to build a robust potential by training on subsystems before tackling the complex interface.

**Workflow Steps:**
1.  **System Initialization**:
    *   Define `config.yaml` for the "MgO" (substrate) and "FePt" (deposit) systems separately.
2.  **Phase A: MgO Substrate Training**:
    *   Run `StructureGenerator` to create bulk and surface MgO structures.
    *   Run `Oracle` and `Trainer` to build `potential_mgo.yace`.
    *   *Validation*: Check lattice constant and band gap (M3GNet prediction vs DFT).
3.  **Phase B: FePt Alloy Training**:
    *   Run `AdaptiveExploration` to sample FePt clusters and bulk L10/A1 phases.
    *   Train `potential_fept.yace`.
    *   *Validation*: Check mixing energy and convex hull.
4.  **Phase C: Interface Learning (The "Aha!" Moment)**:
    *   Combine the two potentials? No. We must learn the interaction.
    *   Generate "Cluster-on-Slab" configurations.
    *   Run Active Learning specifically on the interface region.
    *   **Outcome**: A final `potential_interface.yace` capable of handling Mg-O-Fe-Pt interactions.

**Success Criteria:**
*   **Real Mode**: Final RMSE < 2 meV/atom.
*   **Mock Mode**: Notebook completes all cells. Generated `.yace` file exists.

### Notebook 2: `tutorials/02_Deposition_and_Ordering.ipynb`
**Title:** "From Deposition to Ordering: Bridging Time Scales with MD and aKMC"
**Objective:** Demonstrate the full power of the system by simulating a realistic growth process.

**Workflow Steps:**
1.  **Setup**: Load `potential_interface.yace` generated in Notebook 1 (or a pre-supplied one for the tutorial).
2.  **Phase D: Dynamic Deposition (MD)**:
    *   Setup LAMMPS with `fix deposit`.
    *   Inject Fe and Pt atoms alternately onto the MgO substrate at 600K.
    *   *Observation*: Visualize atoms landing and diffusing using `ase.visualize`.
    *   *Check*: Ensure atoms do not penetrate the substrate (ZBL check).
3.  **Phase E: Long-Term Ordering (aKMC)**:
    *   Take the final disordered cluster from the MD stage.
    *   Bridge to **EON**. Initialize the aKMC simulation.
    *   Run `process_search` to find diffusion barriers.
    *   Evolve the system over "seconds" or "hours" (simulated time).
    *   *Observation*: Watch the disordered alloy transform into the chemically ordered L10 phase (Fe and Pt layers alternating).

**Success Criteria:**
*   **Real Mode**: Observe an increase in the "Chemical Ordering Parameter" (count of Fe-Fe vs Fe-Pt bonds).
*   **Mock Mode**: The code successfully calls EON (or mocks the call) and produces a dummy trajectory file.

## 3. Validation Steps for QA Agent

When the Auditor Agent reviews the implementation, it must verify the following:

### 3.1. Verification of "Mockability"
*   **Action**: Run `export CI=true && pytest --nbval tutorials/`
*   **Expectation**: Both notebooks must execute fully without requiring an API key for Quantum Espresso or a license for VASP. They must not try to submit SLURM jobs.

### 3.2. Verification of Physical Correctness (Real Mode)
*   **Action**: Inspect the `validation_report.html` generated in Notebook 1.
*   **Expectation**:
    *   Phonon spectra for MgO should be real (no imaginary modes).
    *   Elastic constants should satisfy Born stability criteria.
    *   Parity plots should show tight correlation ($R^2 > 0.99$).

### 3.3. Verification of Hybrid Architecture
*   **Action**: Inspect the generated `in.lammps` files in Notebook 2.
*   **Expectation**: The command `pair_style hybrid/overlay pace zbl` (or similar) MUST be present. This confirms the system is correctly applying the core-repulsion safety net defined in the architecture.

### 3.4. Verification of Self-Healing
*   **Action**: In Cycle 03 tests, artificially corrupt a DFT input file.
*   **Expectation**: The system should log a "Calculation Failed" warning, attempt to fix parameters (e.g., `mixing_beta`), and retry. It should NOT crash with a python traceback.

## 4. Scientific Validity Checks

To ensure the tutorial is not just "running code" but "doing science," the following checks are embedded:

1.  **Time-Scale Bridging**: The transition from MD (Phase D) to kMC (Phase E) must be explicit. The tutorial must explain *why* MD cannot see the ordering (it's too slow) and how kMC solves it.
2.  **Interface Construction**: The training set for Phase C must include "detached" clusters and "embedded" atoms to capture the full range of adsorption energies.
3.  **Visual Confirmation**: The notebooks must produce inline PNGs of the atomic structures. A black box result is not acceptable for UAT.
