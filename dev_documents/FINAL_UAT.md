# User Acceptance Testing (UAT) Master Plan

## 1. Tutorial Strategy

The goal of the UAT is to prove that PYACEMAKER transforms a complex scientific workflow into a "Zero-Config" experience, while still allowing for advanced "Scientific Discovery" scenarios. We will utilize a "Dual-Mode" strategy for the tutorials to ensure they are runnable by everyone, everywhere.

### The "Dual-Mode" Execution Strategy
To address the challenge that real scientific simulations (DFT, long MD) take days to run, every tutorial notebook will implement a `IS_CI_MODE` flag.

*   **Real Mode (User)**:
    *   Uses actual binaries (Quantum Espresso, LAMMPS, Pacemaker).
    *   Runs on the user's workstation or cluster.
    *   Performs full active learning loops.
    *   **Goal**: Scientific Validation.
*   **Mock Mode (CI / GitHub Actions)**:
    *   Uses Mock objects or internal lightweight calculators (e.g., `ase.calculators.emt`).
    *   Simulates "Time" by skipping heavy calculations and loading pre-calculated results.
    *   **Goal**: Functional Verification (The code runs without crashing).

### Tutorial Progression
The tutorials are designed to tell a story: "From basics to advanced discovery."

1.  **Phase 1: The Basics (Training)**
    *   Focus on the "Zero-Config" promise.
    *   Show how to train a potential for a bulk material (MgO) and a metal (FePt) separately.
    *   Demonstrate the "Active Learning" graph (Error vs. Data size).
2.  **Phase 2: The Application (Dynamics)**
    *   Use the trained potential.
    *   Simulate the deposition process (MD).
    *   Demonstrate the "Hybrid Potential" safety net (robustness).
3.  **Phase 3: The Discovery (Scale-Up)**
    *   Bridge the gap to long timescales (kMC).
    *   Observe ordering phenomena that MD cannot reach.

## 2. Notebook Plan

We will generate the following Jupyter Notebooks in the `tutorials/` directory.

### `tutorials/01_MgO_FePt_Training.ipynb`
**Title**: "Zero-Config Active Learning for MgO and FePt"
**The "Aha!" Moment**: "I just wrote one YAML file, and the system automatically explored the phase space, ran DFT, and trained a potential that is 99% accurate."
**Key Steps**:
1.  **Setup**: Import `Orchestrator`. Load `config_mgo.yaml`.
2.  **Execution**: Run `orchestrator.run()`. Watch the cycle logs in real-time.
3.  **Visualization**:
    *   Plot the "Learning Curve" (RMSE vs Cycle).
    *   Visualize "Halted Structures" (What did the AI find difficult?).
    *   Show Phonon Dispersion of the final MgO potential (Validation).

### `tutorials/02_Deposition_and_Ordering.ipynb`
**Title**: "Simulating Fe/Pt Nanoparticle Growth and Ordering on MgO"
**The "Aha!" Moment**: "I can simulate a complex hetero-epitaxial growth process and observe chemical ordering, all with the potential I just trained."
**Key Steps**:
1.  **Preparation**: Load the `potential.yace` generated in Tutorial 01.
2.  **MD Deposition (LAMMPS)**:
    *   Setup a `DepositionDynamics` config.
    *   Run MD: Fe and Pt atoms raining down on MgO.
    *   **Visual**: Animate the deposition process (using `nglview` or embedded GIF).
3.  **Ordering (aKMC/EON)**:
    *   Take the final cluster from MD.
    *   Run EON kMC to relax the structure over "hours" (simulated time).
    *   **Visual**: Show the transition from "Disordered Alloy" to "L10 Ordered Phase" (Color code Fe/Pt).
4.  **Analysis**:
    *   Calculate Adhesion Energy.
    *   Calculate Order Parameter (count Fe-Pt bonds).

## 3. Validation Steps

The Quality Assurance (QA) Agent (or the human user) should verify the following when running these notebooks:

### Functional Checks
*   [ ] **No Crashes**: Notebooks run from top to bottom without exceptions (in both CI and Real modes).
*   [ ] **Dependency Handling**: If `lammps` is missing, the notebook gracefully degrades (skips cell or warns) rather than crashing with `FileNotFoundError`.
*   [ ] **Output Generation**: Check that `active_learning/` directories and `potential.yace` files are actually created on disk.

### Scientific Checks (Real Mode only)
*   [ ] **Physics Baseline**: In the MD section, verify that atoms do not overlap (distance < 1.0 Å). If they do, the Hybrid Potential failed.
*   [ ] **Convergence**: The Learning Curve in NB01 should show a decreasing trend.
*   [ ] **Ordering**: In NB02, the kMC result should have a higher Order Parameter (more mixed bonds) than the initial random cluster, or lower energy.

### UX Checks
*   [ ] **Clarity**: Are the explanations in Markdown cells clear?
*   [ ] **Visuals**: Do the plots appear inline? Are they labeled?
*   [ ] **Logs**: Are the logs informative but not overwhelming?
