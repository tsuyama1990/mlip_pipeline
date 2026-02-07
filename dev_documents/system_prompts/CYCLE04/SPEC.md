# Cycle 04 Specification: Dynamics Engine (MD) & Hybrid Potentials

## 1. Summary

**Goal**: Implement the real Molecular Dynamics (MD) capabilities using **LAMMPS**, the workhorse of classical simulations. This cycle transforms the system from a static learner to a dynamic explorer. The `MDInterface` will execute LAMMPS, apply **Hybrid Potentials** (ACE + ZBL/LJ) for physical robustness, and implement the critical **"Uncertainty Watchdog"** (fix halt) that stops the simulation when the potential enters unknown territory.

**Key Deliverables**:
1.  **`MDInterface` (Dynamics)**: A wrapper around LAMMPS (via Python `lammps` module or `ase.calculators.lammps`). It manages the simulation setup, execution, and trajectory parsing.
2.  **Hybrid Potential Logic**: Automatically generates `pair_style hybrid/overlay pace zbl` commands to ensure core repulsion.
3.  **Halt Logic**: Configures `fix halt` based on the extrapolation grade ($\gamma$) calculated by Pacemaker's LAMMPS plugin.

## 2. System Architecture

Files in **bold** are the primary focus of this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **config.py**         # DynamicsConfig (LAMMPS settings)
├── infrastructure/
│   ├── dynamics/
│   │   ├── **__init__.py**
│   │   └── **lammps_md.py**  # MDInterface (LAMMPS wrapper)
│   └── ...
└── utils/
    └── **lammps_utils.py**   # Helpers for input generation
```

## 3. Design Architecture

### 3.1 `MDInterface` (Dynamics)

*   **Config**: `command` (str), `time_step` (float), `temperature` (float), `pressure` (float), `ensemble` ("nvt", "npt"), `uncertainty_threshold` (float).
*   **Logic**:
    1.  Initialise `lammps` object (Python interface preferred for control).
    2.  Set up the simulation box from `Structure`.
    3.  **Hybrid Potential Setup**:
        *   `pair_style hybrid/overlay pace zbl 1.0 2.0`
        *   `pair_coeff * * pace potential.yace ...`
        *   `pair_coeff * * zbl ...` (Generate ZBL parameters from atomic numbers).
    4.  **Watchdog Setup**:
        *   `compute pace_gamma all pace ... gamma_mode=1`
        *   `variable max_gamma equal max(c_pace_gamma)`
        *   `fix watchdog all halt 10 v_max_gamma > ${threshold} error hard`
    5.  Run simulation (`run 100000`).
    6.  **Catch Halt**: Detect if LAMMPS stopped due to the watchdog.
    7.  **Return**: `ExplorationResult` containing the final structure (if halted) or trajectory.

## 4. Implementation Approach

1.  **Enhance Config**: Update `domain_models/config.py` for LAMMPS settings.
2.  **Implement `lammps_md.py`**: Use `lammps` Python module if available, otherwise fallback to `subprocess` with input file generation.
3.  **Implement Hybrid Logic**: Write a utility to generate ZBL parameters for any pair of elements.
4.  **Integration Test**: Verify `Orchestrator` can run `dynamics.type="lammps"`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Input Generation**: Verify that `MDInterface` generates valid LAMMPS input commands, especially the `hybrid/overlay` part.
*   **ZBL Parameters**: Verify correct ZBL parameters for standard pairs (e.g., Fe-Fe, O-O).

### 5.2 Integration Testing
*   **"Mock LAMMPS"**: Create a dummy script that behaves like LAMMPS (reads input, writes a dump file).
*   **Real Execution (Halt Test)**: If LAMMPS+USER-PACE is installed:
    *   Run a simulation with a very low uncertainty threshold (e.g., `gamma=0.0`).
    *   Verify that it halts immediately.
    *   Verify that `ExplorationResult.halted` is True.
