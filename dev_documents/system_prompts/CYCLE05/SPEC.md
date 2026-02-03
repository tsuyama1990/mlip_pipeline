# Cycle 05 Specification: The Active Learning Loop (OTF Dynamics)

## 1. Summary

Cycle 05 closes the loop. It connects the "Explorer" (Structure Gen), "Oracle" (DFT), and "Trainer" (Pacemaker) into a self-driving autonomous system. The centerpiece of this cycle is the **Dynamics Engine** capable of On-the-Fly (OTF) active learning. Instead of simple random exploration (Cycle 03), we now run full Molecular Dynamics simulations driven by the ACE potential. Crucially, we implement a "Watchdog" mechanism using LAMMPS's `fix halt` command: the simulation monitors the extrapolation grade ($\gamma$) at every step. If the potential enters an uncertain region ($\gamma > \text{threshold}$), the simulation halts, the system extracts the problematic structure, labels it via the Oracle, retrains, and resumes. This "Halt-and-Repair" loop is the key to robustness.

## 2. System Architecture

### 2.1 File Structure

**Bold** files are to be created or modified in this cycle.

```text
.
├── src/
│   └── mlip_autopipec/
│       ├── physics/
│       │   ├── dynamics/
│       │   │   ├── **__init__.py**
│       │   │   ├── **lammps_runner.py** # LAMMPS Wrapper
│       │   │   ├── **log_parser.py**    # Parse log.lammps
│       │   │   └── **input_gen.py**     # Generate in.lammps
│       │   └── __init__.py
│       ├── orchestration/
│       │   └── **otf_loop.py**          # The Loop Logic
│       └── domain_models/
│           └── **dynamics.py**          # MDStatus, MDResult
└── tests/
    ├── unit/
    │   └── **test_lammps_runner.py**
    └── integration/
        └── **test_otf_loop.py**
```

## 3. Design Architecture

### 3.1 `LammpsRunner` Class
-   **Role**: Executes MD simulations with strict monitoring.
-   **Responsibilities**:
    -   Generate `in.lammps`: Must include `pair_style hybrid/overlay pace zbl` and `fix halt`.
    -   Execute LAMMPS: using `subprocess` (calling `lmp_serial` or `lmp_mpi`).
    -   Detect Halt: Check the exit code or log file to see if the run finished or was halted by the watchdog.
    -   Extract Structure: If halted, identify the specific timestep and atom(s) causing high $\gamma$.

### 3.2 The Watchdog Logic (`fix halt`)
-   **LAMMPS Command**:
    ```lammps
    compute pace all pace ... gamma_mode=1
    variable max_gamma equal max(c_pace)
    fix watchdog all halt 10 v_max_gamma > 5.0 error hard
    ```
-   **Interpretation**: Every 10 steps, check if the maximum uncertainty in the system exceeds 5.0. If so, stop immediately.

### 3.3 `OTFLoop` Logic (in `otf_loop.py`)
-   **State Machine**:
    1.  **Deploy**: Write current `.yace` to working dir.
    2.  **Run**: Start MD.
    3.  **Check**:
        -   If `Converged`: Task complete.
        -   If `Halted`:
            1.  Parse dump file for the last frame.
            2.  **Local Selection**: (Optional) Perturb the halted structure to sample gradients.
            3.  **Oracle**: Calculate DFT for these structures.
            4.  **Train**: Update potential.
            5.  **Resume**: Restart MD from the halt point (or slightly before).

## 4. Implementation Approach

1.  **LAMMPS Input Generator**:
    -   Create `input_gen.py`.
    -   Must handle dynamic assignment of atom types (Species A -> Type 1).
    -   Must insert the `fix halt` block automatically if `uncertainty_threshold` is set.

2.  **Log Parser**:
    -   Create `log_parser.py`.
    -   Parse thermodynamic output (Temp, Press, Vol) to track stability.
    -   Detect specific error messages ("Rule ... met").

3.  **Structure Extraction**:
    -   When halted, the dump file contains the "dangerous" configuration.
    -   Read it using `ase.io.read`.
    -   Pass it to `PeriodicEmbedding` (from Cycle 02) to prepare for DFT.

4.  **Orchestrator Integration**:
    -   Update `Orchestrator` to delegate the "Exploration" phase to `OTFLoop` instead of the simple `AdaptiveExplorer` when in Active Learning mode.

## 5. Test Strategy

### 5.1 Unit Testing
-   **`test_lammps_runner.py`**:
    -   **Input Gen**: Verify the generated string contains `fix halt`.
    -   **Parser**: Feed a fake log file where `v_max_gamma` goes above 5.0. Verify the parser returns `status=HALTED`.

### 5.2 Integration Testing
-   **`test_otf_loop.py`**:
    -   **Mock LAMMPS**: Since running real LAMMPS in CI is hard/slow, we Mock the `LammpsRunner`.
    -   **Scenario**:
        1.  Step 1: MockRunner returns `HALTED` and a random structure.
        2.  Orchestrator calls Oracle (Mock).
        3.  Orchestrator calls Trainer (Mock).
        4.  Step 2: MockRunner returns `CONVERGED`.
    -   **Assertion**: Verify the loop ran exactly 2 times and the dataset grew by 1 batch.
