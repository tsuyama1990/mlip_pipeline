# Cycle 05 Specification: Dynamics Phase II (Active Learning Loop)

## 1. Summary

Cycle 05 implements the core logic of the **Active Learning** strategy: the ability to detect when the simulation enters an "unknown" region and automatically stop to collect data. This effectively closes the loop between the Dynamics Engine and the Oracle.

We enhance the LAMMPS interface to use the `fix halt` command triggered by the `compute pace` extrapolation grade ($\gamma$). When a halt occurs, the system must perform a sophisticated "surgery": extracting the specific atomic configuration that caused the high uncertainty, and—crucially—embedding this local cluster into a periodic supercell suitable for DFT calculation (Periodic Embedding).

This cycle transforms the system from a passive simulator into an active explorer.

## 2. System Architecture

We introduce the `active_learning` package and enhance the `dynamics` module.

### 2.1 File Structure

```ascii
src/mlip_autopipec/
├── dynamics/
│   ├── **writer.py**               # Updated with 'fix halt'
│   └── **parser.py**               # Updated to parse Halt logs
├── **active_learning/**
│   ├── **__init__.py**
│   ├── **manager.py**              # ActiveLearningManager
│   └── **embedding.py**            # Periodic Embedding Logic
└── orchestration/
    └── phases/
        └── **active_learning.py**  # Integrated Phase
```

## 3. Design Architecture

### 3.1 Watchdog Implementation (`dynamics/writer.py`)

We need to inject the following logic into `in.lammps`:
```lammps
compute pace_gamma all pace potential.yace ... gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt 10 v_max_gamma > ${threshold} error hard
```
We interpret "error hard" (non-zero exit code) as a **successful detection** event, distinguishing it from a system crash.

### 3.2 Candidate Processor (`active_learning/manager.py`)

*   **Responsibility**: Handle the "Halted" state.
*   **Methods**:
    *   `extract_halt_structure(dump_file: Path) -> Atoms`: Reads the last frame of the simulation dump.
    *   `generate_local_candidates(structure: Atoms) -> List[Atoms]`: Generates perturbations (e.g., small displacements) around the halted structure to learn the local curvature.

### 3.3 Periodic Embedding (`active_learning/embedding.py`)

*   **Problem**: Active learning often detects local defects. We cannot simply cut a sphere and run DFT (boundary issues).
*   **Solution**:
    1.  Identify the "active region" (atoms with high $\gamma$).
    2.  Define a bounding box (Orthorhombic) large enough to cover the interaction cutoff.
    3.  Create a small supercell that repeats this box.
    4.  Return this periodic structure for the Oracle.

## 4. Implementation Approach

1.  **Step 1: Update LAMMPS Writer.**
    *   Add `enable_uncertainty_check` flag to the writer.
    *   Inject the `compute pace` and `fix halt` commands.

2.  **Step 2: Update LAMMPS Runner.**
    *   Modify error handling: If exit code is non-zero, check the log file.
    *   If log says "Fix halt condition met", treat as `Status.HALTED` (not error).

3.  **Step 3: Implement Periodic Embedding.**
    *   Use `ase.build.cut` or manual slicing to create the supercell.
    *   Ensure the cell size is at least $2 \times R_{cut}$ to avoid self-interaction in DFT.

4.  **Step 4: Integrate into Orchestrator.**
    *   The `ExplorationPhase` should now return a status `HALTED` and the path to the dump file.
    *   The `WorkflowManager` catches this and triggers the `SelectionPhase` (extract -> embed -> oracle).

## 5. Test Strategy

### 5.1 Unit Testing
*   **Embedding Logic:**
    *   Create a large supercell with a defect in the center.
    *   Mark the center atom as "high gamma".
    *   Call `embed_cluster`.
    *   Assert the resulting `Atoms` object is smaller than the original but larger than $R_{cut}$, and has periodic boundary conditions.

### 5.2 Integration Testing
*   **Halt Simulation:**
    *   Since we can't easily force Pacemaker to output high gamma without a real potential, we will mock the "Halt" by setting a very low threshold (e.g., $\gamma > 0.0$) or using a dummy variable in LAMMPS.
    *   Verify that the Python runner catches the exit code and correctly identifies it as a Halt event.
