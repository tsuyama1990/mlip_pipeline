# Cycle 05 Specification: Inference Engine (MD)

## 1. Summary
Cycle 05 builds the **Dynamics Engine**, enabling the system to run Molecular Dynamics (MD) simulations using **LAMMPS**. Key features include the **Hybrid Potential** setup (overlaying ACE with ZBL/LJ for safety) and the **Uncertainty Watchdog**, which monitors the extrapolation grade ($\gamma$) and halts the simulation if it enters dangerous territory.

## 2. System Architecture

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       └── inference.py        # **Inference Config Schema**
├── inference/
│   ├── __init__.py
│   ├── runner.py               # **LammpsRunner Class**
│   ├── generator.py            # **LAMMPS Input Generator**
│   ├── parsers.py              # **Log/Dump Parser**
│   └── watchdog.py             # **Uncertainty Logic**
```

## 3. Design Architecture

### 3.1. LAMMPS Runner (`inference/runner.py`)
- **Execution**: Runs `lmp_serial` or `lmp_mpi` via subprocess.
- **Safety**: Uses `shell=False`.
- **Interface**: `run_md(structure, potential_path, params) -> MDResult`

### 3.2. Input Generator (`inference/generator.py`)
Responsible for writing `in.lammps`.
- **Hybrid Overlay**: Logic to write:
  ```lammps
  pair_style hybrid/overlay pace zbl 1.0 2.0
  pair_coeff * * pace potential.yace ...
  pair_coeff * * zbl ...
  ```
- **Watchdog Integration**: Writes `fix halt` commands:
  ```lammps
  compute gamma all pace ...
  fix halter all halt 10 v_gamma > 5.0 error hard
  ```

### 3.3. Output Parser (`inference/parsers.py`)
- Detects if the run finished normally or was halted.
- If halted, extracts the **Halt Step** and the **High Uncertainty Atoms**.

## 4. Implementation Approach

1.  **Config**: `InferenceConfig` (temperature, pressure, steps, uncertainty_threshold).
2.  **Generator**: Implement `LammpsInputWriter`. Requires mapping ASE species to LAMMPS types.
3.  **Runner**: Implement `LammpsRunner`. Must handle the specific error code returned by `fix halt`.
4.  **Parser**: Use regex to find "Halt" messages in stdout.

## 5. Test Strategy

### 5.1. Unit Testing
- **Input Generation**: Check if `pair_style hybrid/overlay` is correctly formatted.
- **Parsing**: Feed a log file with a halt event. Verify it detects the halt and extracts the step number.

### 5.2. Integration Testing
- **Mock LAMMPS**: Create a dummy script that prints a halt message after 100 steps.
- **Watchdog**: Run the runner against the mock. Verify `MDResult` reports `status='halted'` and captures the dump file path.
