# Cycle 04 Specification: Automated DFT Factory

## 1. Summary

Cycle 04 implements **Module C: DFT Factory**. This is the most critical, expensive, and fragile part of the pipeline. We need to take the "Selected" candidates from Cycle 03 and calculate their true quantum-mechanical properties (Energy, Forces, Stress) using Quantum Espresso (QE).

The challenge is that DFT is notoriously finicky. Calculations can fail for a multitude of reasons:
-   **SCF Convergence**: The electronic density oscillates and never settles (common in metals/magnetic systems).
-   **Hardware**: Out of memory (OOM), walltime limits, disk quota exceeded.
-   **Physics**: Bad initial magnetic moments, metallic systems treated as insulators.

This cycle implements a **Robust Auto-Recovery Logic**. Instead of simply crashing on failure, the system catches the error, diagnoses it (e.g., "Convergence Error"), and resubmits the job with a "Softer" set of parameters (e.g., higher smearing, lower mixing beta). This "Ladder of Robustness" ensures high throughput and minimizes manual intervention.

## 2. System Architecture

### File Structure
**bold** files are to be created or modified.

```
mlip_autopipec/
├── dft/
│   ├── **__init__.py**
│   ├── **runner.py**           # QERunner (Process Management)
│   ├── **inputs.py**           # Input File Generator (The Writer)
│   ├── **parsers.py**          # Output File Parser (The Reader)
│   ├── **errors.py**           # Custom DFT Exception Classes
│   └── **recovery.py**         # The Logic for Fixing Errors
├── config/
│   └── schemas/
│       └── **dft.py**          # Enhanced DFT options
```

### Data Dictionary

| Model Name | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **DFTConfig** | pseudopotential_dir | Path | Directory containing .UPF files. |
| | ecutwfc | float | Wavefunction cutoff (Ry). |
| | kspacing | float | K-point density (1/A). |
| | mixing_beta | float | SCF Mixing parameter (0.0-1.0). Default 0.7. |
| | diagonalization | str | Solver ('david', 'cg'). |
| | smearing | str | Smearing type ('mv', 'mp', 'fd'). |
| | degauss | float | Smearing width (Ry). |
| **DFTResult** | energy | float | Total Energy (eV). |
| | forces | ndarray | Force vectors (eV/A). |
| | stress | ndarray | Virial stress tensor (eV/A^3). |

### Component Interaction
-   **`QERunner`** uses **`inputs.write_pw_input`** to generate the file.
-   **`QERunner`** spawns the process using `subprocess`.
-   **`QERunner`** uses **`parsers.read_pw_output`** to verify success.
-   If failure, **`QERunner`** calls **`recovery.get_recovery_strategy`**.

## 3. Design Architecture

### QERunner Class
-   **Responsibility**: Execute `pw.x` binaries safely.
-   **Methods**:
    -   `run(structure: Atoms, run_dir: Path, calculation_id: str)`: Main entry point.
    -   `_submit_subprocess()`: Wraps `subprocess.Popen`. Handles timeouts.
-   **MPI Handling**: Reads `resources.parallel_cores` config to generate `mpirun -np N pw.x -in pw.in > pw.out`.

### Input Generator (`inputs.py`)
-   **Standardization**: We enforce specific flags required for MLIP training data.
    -   `tprnfor = .true.`: Essential for Force training.
    -   `tstress = .true.`: Essential for Stress/Virial training.
    -   `disk_io = 'low'`: To prevent filling the disk with massive wavefunction files (`.wfc`).
-   **K-points**: We use `kspacing` (inverse density) to auto-generate grids ($N_k \sim 1/L$). This ensures consistent sampling density across different cell sizes.
-   **Pseudopotentials**: The system must look up the correct `.UPF` file for each element in the `pseudopotential_dir`.

### Auto-Recovery Logic (`recovery.py`)
This is a State Pattern or a simple decision tree.
1.  **Level 0 (Default)**: `mixing_beta=0.7`, `diagonalization='david'`.
2.  **Level 1 (Convergence Fail)**: "Soft Mixing". Reduce mixing to `beta=0.3`. This slows down the updates to the density matrix, preventing oscillations.
3.  **Level 2 (Still Failing)**: "Robust Solver". Change solver to `diagonalization='cg'` (Conjugate Gradient is slower but more robust than Davidson).
4.  **Level 3 (Still Failing)**: "High Temperature". Increase electronic temperature `degauss += 0.01` to smooth the Fermi surface (useful for metals).
5.  **Level 4**: Give up. Mark as `FAILED`.

### Output Parser (`parsers.py`)
-   **Robustness**: Must verify the file ended cleanly (look for "JOB DONE" marker).
-   **Sanity Check**: If Forces contains `NaN` or `Inf`, raise `DFTRuntimeError`. This happens when atoms are too close.
-   **Unit Conversion**: QE uses Ry/Bohr; ASE uses eV/Angstrom. The parser (or ASE wrapper) must handle this.

## 4. Implementation Approach

1.  **Develop Input Writer**: In `inputs.py`, create `write_pw_input`. Use `ase.io.espresso` as a base but explicitly post-process the parameter dictionary to ensure our mandatory flags (`tprnfor`) are present.
2.  **Develop Parser**: In `parsers.py`, create `read_pw_output`. Prefer parsing the XML (`pw.xml`) if available as it is machine-readable. Fallback to text parsing. Ensure stress is converted to the correct units (kBar -> eV/A^3).
3.  **Implement Runner with Retry Loop**:
    -   In `runner.py`, implement a loop: `while attempt < max_retries`.
    -   Inside the loop, run QE.
    -   If return code != 0 or parser raises Error, catch it.
    -   Call `recovery.get_next_params(current_params, error_type)`.
    -   Update params and `continue`.
4.  **Integration**: Update `WorkflowManager` (in Cycle 06, but tested here) to poll `status=SELECTED`, run `QERunner`, and save `DFTResult` to DB.

## 5. Test Strategy

### Unit Testing
-   **Input Generation**:
    -   Pass an Al atom. Check if `pw.in` contains `ATOMIC_SPECIES Al ...`.
    -   Check if K-points are odd numbers (Monkhorst-Pack).
    -   Check if `tprnfor=.true.` is present.
-   **Parser**:
    -   Create a dummy `pw.out` file with known Energy/Forces.
    -   Parse it. Assert values match.
    -   Create a truncated `pw.out` (simulation crashed). Assert `ParserError`.
-   **Recovery**:
    -   Mock the execution failure.
    -   Start with beta=0.7.
    -   Assert next attempt uses beta=0.3.

### Integration Testing
-   **Real Execution**:
    -   Requires `pw.x` on the path (or a mock script that behaves like it).
    -   Run a calculation on H2 molecule.
    -   Check DB for results.
    -   Check that `wfc` files are deleted (cleanup).
-   **Failure Simulation**:
    -   Create a script that exits with code 1.
    -   Run QERunner. Verify it retries and eventually marks as failed.
