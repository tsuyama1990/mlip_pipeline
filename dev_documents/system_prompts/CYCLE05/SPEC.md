# Cycle 05 Specification: Dynamics Engine I (LAMMPS & MD)

## 1. Summary
This cycle implements the "Runner" component: the `DynamicsEngine`. Its role is to execute Molecular Dynamics (MD) simulations using the potentials trained in the previous cycles. This is not a standard MD runner; it is an "Active Learning" runner. It implements two critical safety features:
1.  **Hybrid/Overlay Potentials**: It automatically constructs LAMMPS input scripts that superimpose a ZBL/LJ baseline onto the ACE potential, ensuring physical behavior at short interatomic distances.
2.  **Uncertainty Watchdog**: It configures LAMMPS to monitor the extrapolation grade ($\gamma$) of the ACE potential in real-time. If $\gamma$ exceeds a safety threshold, the simulation is halted immediately, and the problematic configuration is captured for labeling. This "Halt & Diagnose" loop is the heart of the active learning process.

## 2. System Architecture

### 2.1. File Structure
files to be created/modified in this cycle are bolded.

```text
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── base.py                 # [CREATE] Abstract Base Class
│   │   ├── lammps_driver.py        # [CREATE] LAMMPS Wrapper
│   │   ├── input_builder.py        # [CREATE] Input Script Factory
│   │   └── analyze_halt.py         # [CREATE] Log/Dump Parser
├── domain_models/
│   ├── config.py                   # [MODIFY] Add DynamicsConfig
│   └── enums.py                    # [MODIFY] Add DynamicsStatus
└── core/
    └── orchestrator.py             # [MODIFY] Integrate Dynamics into explore()
```

### 2.2. Component Interaction
1.  **`Orchestrator`** calls `dynamics.run_exploration(potential, context=state)`.
2.  **`InputBuilder`**:
    *   Generates `in.lammps`.
    *   Includes `pair_style hybrid/overlay pace zbl`.
    *   Includes `fix halt` condition on `c_pace_gamma`.
3.  **`LammpsDriver`**:
    *   Executes `lmp -in in.lammps`.
    *   Streams output to log.
4.  **`AnalyzeHalt`**:
    *   If exit code indicates "Halt" (or specific message):
        *   Reads `dump.lammps`.
        *   Extracts the snapshot where $\gamma$ spiked.
        *   Returns `ExplorationResult(status=HALTED, structure=Atoms)`.
    *   If completed normally:
        *   Returns `ExplorationResult(status=CONVERGED)`.

## 3. Design Architecture

### 3.1. Domain Models

#### `config.py`
*   `DynamicsConfig`:
    *   `timestep`: float (default 1.0 fs)
    *   `temperature`: float
    *   `steps`: int
    *   `uncertainty_threshold`: float (default 5.0)

#### `enums.py`
*   `DynamicsStatus`: `RUNNING`, `COMPLETED`, `HALTED`, `CRASHED`.

### 3.2. Core Logic

#### `input_builder.py`
*   **Responsibility**: Create robust LAMMPS scripts.
*   **Key Logic**:
    ```lammps
    pair_style hybrid/overlay pace zbl 1.0 2.0
    pair_coeff * * pace potential.yace Ti O
    pair_coeff * * zbl 22 8
    compute gamma all pace potential.yace gamma_mode=1
    variable max_gamma equal max(c_gamma)
    fix watchdog all halt 10 v_max_gamma > 5.0 error hard
    ```

#### `analyze_halt.py`
*   **Responsibility**: Post-mortem analysis.
*   **Logic**:
    *   Read `log.lammps` to find the timestep of failure.
    *   Read `dump.lammps` (using `ase.io.read`) to get the atoms.
    *   Identify which atoms had high gamma (if per-atom gamma is dumped).

## 4. Implementation Approach

### Step 1: Interface Definition
*   Define `BaseDynamics` in `components/dynamics/base.py`.
*   Define `run_exploration(...) -> ExplorationResult`.

### Step 2: Input Builder
*   Implement `LammpsInputBuilder`.
*   Need lookup tables for ZBL parameters (Atomic numbers).

### Step 3: LAMMPS Driver
*   Implement `LammpsDriver`.
*   Use `subprocess` or `lammps` python module (if available). Prefer `subprocess` for isolation in Cycle 05.

### Step 4: Halt Analysis
*   Implement parsing logic.
*   Create a test dump file with a "bad" structure.

### Step 5: Orchestrator Integration
*   Update `Orchestrator` to handle `HALTED` status.
*   If halted, the returned structure becomes the seed for the next cycle's Oracle.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_input_builder.py`**:
    *   Verify string contains `fix halt`.
    *   Verify `pair_coeff` matches element list.

### 5.2. Integration Testing
*   **`test_lammps_mock.py`**:
    *   Since we might not have `lmp` in CI:
    *   Create a `MockLammpsDriver` that writes a dummy log file saying "Halted at step 500" and creates a dummy dump file.
    *   Verify `AnalyzeHalt` correctly parses this mock output and returns `HALTED`.
