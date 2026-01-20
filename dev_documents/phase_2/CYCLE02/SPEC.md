# Cycle 02 Specification: Automated DFT Factory

## 1. Summary

Cycle 02 implements the **Automated DFT Factory** (Module C). This module is the "engine room" of the pipeline, responsible for generating the ground-truth data required to train the machine learning potential. Without reliable, accurate, and automated DFT calculations, the entire pipeline fails.

The primary objective is to implement a robust interface to **Quantum Espresso (QE)**. Unlike a human user who manually tweaks parameters when a calculation crashes, this module must handle the entire lifecycle of a calculation autonomously. This includes:
1.  **Input Generation**: Automatically selecting appropriate Pseudopotentials (SSSP library), K-point grids (based on cell density), and parallelization flags.
2.  **Execution**: Managing the `subprocess` call to `pw.x`, handling MPI run commands, and managing timeout constraints.
3.  **Auto-Recovery**: The critical differentiator. The system must parse error logs to detect specific failures (e.g., "convergence NOT achieved", "diagonalization error") and automatically retry the calculation with an adjusted strategy (e.g., reducing mixing beta, increasing temperature).
4.  **Data Sanitization**: Before saving to the database, results must be checked for physicality (e.g., no forces > 1000 eV/A due to SCF collapse).

By the end of this cycle, the system will be able to take an `ase.Atoms` object and reliably return its DFT energy and forces, or a detailed error report if it is fundamentally incalculable.

## 2. System Architecture

New components are added to the `src/dft` directory.

```ascii
mlip_autopipec/
├── src/
│   ├── dft/
│   │   ├── __init__.py
│   │   ├── **runner.py**       # QERunner class: Main entry point
│   │   ├── **inputs.py**       # Logic for generating pw.in files
│   │   ├── **recovery.py**     # Error analysis and retry strategy
│   │   ├── **constants.py**    # SSSP definitions, default flags
│   │   └── **parsers.py**      # Custom output parsers (if ASE is insufficient)
│   └── config/
│       └── models.py           # Updated with DFTConfig
├── tests/
│   └── dft/
│       ├── **test_runner.py**
│       ├── **test_inputs.py**
│       └── **test_recovery.py**
```

### Key Components

1.  **`QERunner`**: The facade class. It exposes a simple method `run(atoms: Atoms, id: str) -> DFTResult`. It orchestrates the directory creation, input writing, execution, and parsing. It implements the "Retry Loop".
2.  **`InputGenerator`**: Encapsulates the physics rules. It knows that for "Fe", we need a specific UPF file. It knows that for a 10A cell, we need a 4x4x4 K-grid. It handles the "Sane Defaults" (e.g., `tstress=.true.`, `tprnfor=.true.`).
3.  **`RecoveryHandler`**: A stateless or stateful component that takes a `failure_type` and `current_params` and returns `new_params`. It implements the "Recovery Tree" (e.g., MixBeta -> Preconditioner -> Temp -> Abort).
4.  **`DFTConfig`**: A Pydantic model defining the path to the executable (`pw.x`), the pseudopotential directory, the timeout limits, and the max retries.

## 3. Design Architecture

### Domain Concepts

**The "Static" Protocol**:
For MLIP training, we strictly perform `calculation = 'scf'` (Self-Consistent Field). We do **not** run `relax` or `vc-relax`. The goal is to sample the potential energy surface (PES) at the specific coordinates provided by the generator. If we relaxed them, all our high-energy training data would collapse into the ground state, making the potential blind to repulsive interactions.

**Heuristic Parameter Selection**:
-   **K-Points**: We target a specific density in reciprocal space (e.g., $0.15 \text{\AA}^{-1}$).
    $N_k = \max(1, \text{int}(L^{-1} / \text{density}))$
-   **Magnetism**: If Fe, Co, or Ni are present, we automatically initialize `nspin=2` and assign initial magnetic moments to help convergence.
-   **Symmetry**: We generally use `nosym=.true.` to prevent QE from rotating our structure, which complicates force mapping, unless the structure is explicitly built for symmetry.

**Auto-Recovery State Machine**:
-   **State 0**: Default parameters (Mixing 0.7).
-   **State 1 (Convergence Fail)**: Reduce `mixing_beta` from 0.7 to 0.3.
-   **State 2 (Convergence Fail)**: Change `mixing_mode` to `local-tf`.
-   **State 3 (Convergence Fail)**: Increase `degauss` (temperature) slightly to smooth the Fermi surface.
-   **State 4 (Cholesky Error)**: Switch diagonalization to `cg`.
-   **State 5**: Fail.

### Data Models

```python
class DFTConfig(BaseModel):
    command: str = "mpirun -np 4 pw.x"
    pseudo_dir: Path
    timeout: int = 3600
    recoverable: bool = True
    max_retries: int = 5

class DFTResult(BaseModel):
    uid: str
    energy: float
    forces: List[List[float]] # Nx3 array
    stress: List[List[float]] # 3x3 array
    succeeded: bool
    error_message: Optional[str] = None
    wall_time: float
    # Metadata for provenance
    parameters: Dict[str, Any]
    final_mixing_beta: float
```

## 4. Implementation Approach

1.  **Step 1: Input Generation Logic (`inputs.py`)**:
    -   Create `constants.py` with a dictionary mapping Elements -> Pseudopotential filenames (SSSP Efficiency 1.1).
    -   Implement `InputGenerator.create_input_string()`. It should take an `Atoms` object and a `params` dict.
    -   Implement K-point density logic.
    -   Implement Magnetism logic (check atomic numbers).
2.  **Step 2: The Runner Happy Path (`runner.py`)**:
    -   Implement `QERunner._run_command()`. Use `subprocess.run` with `capture_output=True` and `timeout`.
    -   Implement `QERunner.run()`. It should create a temp dir `run_{uid}`, write `pw.in`, run, and use `ase.io.read` to parse `pw.out`.
3.  **Step 3: Error Detection (`recovery.py`)**:
    -   Create a library of regex patterns.
    -   `"convergence NOT achieved"`
    -   `"error in diagonalization"`
    -   `"maximum CPU time exceeded"`
    -   `"oom-kill"`
    -   Implement `RecoveryHandler.analyze(stdout, stderr)`. Return an Enum `DFTErrorType`.
4.  **Step 4: The Recovery Loop**:
    -   Refactor `QERunner.run()` to use a loop: `while attempt < max_retries`.
    -   If `subprocess` fails (non-zero exit) or `ase` fails to read:
        -   Call `RecoveryHandler.analyze`.
        -   Call `RecoveryHandler.get_strategy(error_type, current_params)`.
        -   Update `input_params` and `continue`.
        -   Log the retry event ("Retrying with mixing_beta=0.3").
    -   If successful, return `DFTResult`.
    -   If loop finishes, raise `DFTFatalError`.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)
Testing the DFT Factory is challenging because we cannot rely on `pw.x` being present in the test environment. Therefore, we heavily rely on **Dependency Injection** and **Mocking**.
-   **Input Verification**: We will create `Atoms` objects with various tricky properties (e.g., highly distorted cells, magnetic elements). We will call `InputGenerator` and inspect the resulting string. We verify that:
    -   `nspin=2` is set for Iron.
    -   The K-points are inversely proportional to cell size.
    -   The correct pseudopotential files are referenced.
    -   `nosym=True` is present.
-   **Recovery Logic**: We will test `RecoveryHandler` in isolation. We will feed it text strings that simulate QE crash logs. We will assert that it returns the correct "Next Strategy". For example, given "convergence NOT achieved", it must return a dict with `mixing_beta=0.3`. Given "Cholesky decomposition failed", it must return `diagonalization='cg'`.
-   **Output Parsing**: We will store actual `pw.out` files (both successful and failed) in `tests/data/`. We will verify that our parser correctly extracts Energy, Forces, and Stress from valid files and raises appropriate errors for incomplete files. We will test the parsing of the Virial stress tensor specifically, as units (Ry/Bohr^3 vs kbar) are a common source of bugs.

### Integration Testing Approach (Min 300 words)
Integration testing involves the `QERunner` orchestration.
-   **Mock Execution**: We will use `pytest-mock` to mock `subprocess.run`.
    -   **Scenario 1 (Success)**: We mock `subprocess.run` to return 0 and write a valid `pw.out` to the disk (copied from test data). We verify `QERunner` returns a populated `DFTResult`.
    -   **Scenario 2 (Recoverable Failure)**: We mock `subprocess.run` to fail the first time (return code 1, write "convergence failed" to stdout). On the *second* call, we check that the command or input file was modified (e.g., looking for `mixing_beta 0.3` in the input), and then make it succeed. This proves the loop works.
    -   **Scenario 3 (Fatal Failure)**: We mock persistent failure. Verify `QERunner` gives up after N tries and raises a clean exception (not a crash).
-   **File Cleanup**: We verify that `QERunner` cleans up large temporary files (`.wfc`, `.hub`) after execution to prevent filling the disk, respecting the `disk_io='low'` setting. We will run a test where we check the directory contents after `run()` returns and ensure only inputs and outputs remain, not scratch files.
