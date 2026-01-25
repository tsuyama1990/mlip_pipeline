# Cycle 04: Dynamics (LAMMPS Engine)

## 1. Summary
This cycle implements the **Dynamics** module, which runs Molecular Dynamics (MD) simulations to explore the chemical space. We use **LAMMPS** as the engine. A critical feature is the **Uncertainty Watchdog**, which monitors the extrapolation grade ($\gamma$) in real-time and halts the simulation if it enters an unknown region, triggering the Active Learning loop.

## 2. System Architecture

We add the `phases/dynamics` module.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── orchestration/
│   └── phases/
│       └── **dynamics/**
│           ├── **__init__.py**
│           ├── **manager.py**       # DynamicsPhase implementation
│           ├── **lammps_driver.py** # LAMMPS execution wrapper
│           ├── **input_writer.py**  # in.lammps generation
│           └── **parser.py**        # Log & Dump file parser
└── tests/
    └── **test_dynamics.py**
```

## 3. Design Architecture

### Hybrid Potential Strategy (`input_writer.py`)
To ensure safety, we never run pure ACE. We generates `in.lammps` with:
```lammps
pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace potential.yace Element1 Element2
pair_coeff * * zbl Z1 Z2
```
This ensures that if atoms overlap, ZBL repulsive forces take over.

### Uncertainty Watchdog (`lammps_driver.py`)
We utilize the `compute pace` command from the USER-PACE package.
```lammps
compute gamma all pace potential.yace gamma_mode=1
variable max_gamma equal max(c_gamma)
fix watchdog all halt 10 v_max_gamma > ${threshold} error hard
```
The driver must detect the specific error code/message associated with `fix halt` and classify it as a "Discovery" rather than a "Crash".

## 4. Implementation Approach

1.  **Input Writer**: Implement a class that generates `in.lammps`. It needs to handle the mapping of element names to IDs and inject the correct `pair_style` commands.
2.  **LAMMPS Driver**:
    *   Ideally use `lammps` Python module (`pip install lammps`) for direct control.
    *   Fallback to `subprocess` if the module is not available.
    *   Must handle `UncertaintyThresholdExceeded` events.
3.  **Result Parsing**: If halted, identify *which* timestep and *which* structure caused the halt. Return the path to the dump file and the timestep index.

## 5. Test Strategy

### Unit Testing
*   **`test_input_writer.py`**: Verify that the generated LAMMPS input contains the `hybrid/overlay` and `fix halt` commands.

### Integration Testing
*   **Mock LAMMPS**: Create a mock script that behaves like LAMMPS.
    *   Case A: Run to completion (exit 0).
    *   Case B: Run and trigger halt (print "Fix halt condition met", exit 1).
*   **Watchdog Verification**: Verify that `DynamicsPhase` correctly interprets the halt signal and returns a status of "HALTED" with metadata.
