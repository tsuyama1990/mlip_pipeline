# Cycle 04 Specification: Dynamics Engine (LAMMPS Inference)

## 1. Summary

Cycle 04 implements the "Dynamics Engine", specifically focusing on Molecular Dynamics (MD) using LAMMPS. This module serves two purposes: evaluating the performance of the potential and acting as the "Explorer" in the active learning loop.

Critical features include:
1.  **Hybrid Potential Application**: To prevent simulation crashes in high-energy regions (where the ACE potential might be undefined), we must overlay a physical baseline (ZBL or LJ) using `pair_style hybrid/overlay`.
2.  **Uncertainty Watchdog**: The system must monitor the extrapolation grade ($\gamma$) calculated by Pacemaker during the simulation. Using the `fix halt` command, LAMMPS will automatically abort the simulation if $\gamma$ exceeds a threshold.
3.  **Halt Recovery**: When a simulation halts, the Python engine must parse the logs, identify the exact timestep, and extract the atomic configuration responsible for the high uncertainty.

## 2. System Architecture

**Files to be created/modified in this cycle are marked in Bold.**

```ascii
src/mlip_autopipec/
├── inference/
│   ├── **__init__.py**
│   ├── **lammps_runner.py**    # Manages LAMMPS process
│   ├── **inputs.py**           # Generates in.lammps and data files
│   ├── **parsers.py**          # Parses log files for halt info
│   └── **processing.py**       # Extracts structures from dump files
```

## 3. Design Architecture

### `LammpsRunner` (in `inference/lammps_runner.py`)
*   **Responsibility**: Execute LAMMPS.
*   **Method**: `run_md(structure, potential_path, parameters) -> RunResult`
*   **Logic**:
    *   Generates input files using `InputWriter`.
    *   Runs LAMMPS via `subprocess`.
    *   Uses `LogParser` to determine if the run finished normally or was halted by the watchdog.

### `InputWriter` (in `inference/inputs.py`)
*   **Responsibility**: Create `in.lammps`.
*   **Key Feature**:
    *   Must generate the complex `pair_style` command.
    *   Example: `pair_style hybrid/overlay pace zbl 1.0 2.0`
    *   Example: `fix watchdog all halt 10 v_max_gamma > 5.0 error hard`

### `LogParser` (in `inference/parsers.py`)
*   **Responsibility**: Read `log.lammps` or stdout.
*   **Logic**:
    *   Detects if the exit code was non-zero.
    *   Scans for the "Halt" message.
    *   Extracts the timestep where the halt occurred.

### `CandidateProcessor` (in `inference/processing.py`)
*   **Responsibility**: Efficiently extract specific frames from large MD dump files.
*   **Logic**:
    *   Uses `ase.io.read` (or optimized custom reader) to load the frame at the halted timestep.
    *   This structure becomes a "Candidate" for the next learning cycle.

## 4. Implementation Approach

1.  **Input Generation**: Implement `InputWriter`. Ensure correct syntax for `pace` pair style and `fix halt`.
2.  **Parsing Logic**: specific regex to catch LAMMPS errors and custom halt messages.
3.  **Runner**: Integrate the above.
4.  **Structure Extraction**: Implement `CandidateProcessor`. Important: Handle large dump files efficiently (don't load whole trajectory into RAM).

## 5. Test Strategy

### Unit Testing
*   **Input**: Verify `in.lammps` contains `pair_style hybrid/overlay` and `fix halt`.
*   **Parser**: Feed a sample log file where `fix halt` was triggered. Assert the parser returns `halted=True` and the correct timestep.
*   **Processor**: Create a dummy trajectory file. Ask to extract frame N. Verify the returned Atoms object matches.

### Integration Testing
*   **Mock Execution**:
    *   Mock `subprocess.run` to simulate a LAMMPS run that crashes with the specific error code for `fix halt`.
    *   Verify that `LammpsRunner` catches this and returns a result object with `status="halted"`.
