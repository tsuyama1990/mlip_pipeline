# Cycle 02 Specification: Oracle Phase (DFT Engine)

## 1. Summary

Cycle 02 focuses on the "Oracle" module, which is responsible for generating ground-truth data using First-Principles (DFT) calculations. In the context of PyAcemaker, the Oracle must be autonomous and robust. It receives atomic structures (candidates) and returns their energy, forces, and stress tensors.

The primary challenge in automating DFT is reliability. Calculations often fail due to SCF (Self-Consistent Field) non-convergence, especially when dealing with the high-energy, distorted structures extracted from active learning. Therefore, a simple "fire and forget" script is insufficient. This cycle implements a **Self-Correcting Oracle** that can detect failures and automatically retry with adjusted parameters (e.g., increased smearing, different mixing schemes) before giving up.

We will strictly target **Quantum Espresso (QE)** as the backend engine for this cycle, utilizing the **ASE (Atomic Simulation Environment)** library for input file generation and output parsing logic, while wrapping the execution in our robust runner.

## 2. System Architecture

We expand the `dft/` directory to include the runner and input handling logic.

### 2.1 File Structure

```ascii
src/mlip_autopipec/
├── config/
│   └── schemas/
│       └── **dft.py**              # Enhanced with retry parameters
├── **dft/**
│   ├── **__init__.py**
│   ├── **runner.py**               # Abstract Base Class (DFTRunner)
│   ├── **qe_runner.py**            # Quantum Espresso Implementation
│   ├── **inputs.py**               # Input generation (ASE adapters)
│   └── **results.py**              # Data Models for DFT Outputs
└── orchestration/
    └── phases/
        └── **oracle.py**           # Oracle Phase Logic
```

## 3. Design Architecture

### 3.1 Data Models (`dft/results.py`)

We need a standardized format to return DFT results, decoupling the consumer (Trainer) from the specific DFT code output.

*   **`DFTResult`**:
    *   `atoms`: ASE Atoms object (final positions)
    *   `energy`: float (eV)
    *   `forces`: List[List[float]] (eV/Å)
    *   `stress`: List[List[float]] (Voigt or full tensor, eV/Å³)
    *   `success`: bool
    *   `error_message`: Optional[str]
    *   `meta`: Dict (e.g., number of SCF steps, wall time)

### 3.2 Abstract Runner (`dft/runner.py`)

*   **`DFTRunner` (ABC)**:
    *   `run_single(atoms: Atoms) -> DFTResult`: Abstract method.
    *   `run_batch(atoms_list: List[Atoms]) -> List[DFTResult]`: Helper for parallel execution (future proofing).

### 3.3 Quantum Espresso Runner (`dft/qe_runner.py`)

*   **`QERunner(DFTRunner)`**:
    *   **Attributes**: `command` (from config), `pseudopotentials` (dict).
    *   **Methods**:
        *   `_generate_input(atoms, params)`: Uses `dft.inputs`.
        *   `_execute_subprocess(input_str)`: Runs the command.
        *   `_parse_output(stdout)`: Extracts physics quantities.
        *   `run_with_recovery(atoms)`: The core loop.

### 3.4 Self-Correction Logic

The `run_with_recovery` method implements a strategy pattern:
1.  **Attempt 1**: Standard parameters (from Config).
2.  **Failure Detection**: Regex search for "convergence NOT achieved".
3.  **Attempt 2**: Decrease `mixing_beta` (e.g., 0.7 -> 0.3).
4.  **Attempt 3**: Increase `smearing` (electron temperature).
5.  **Final**: Mark as failed if all retries fail.

## 4. Implementation Approach

1.  **Step 1: Define `DFTResult`.**
    *   Create the Pydantic model or Dataclass in `dft/results.py`.

2.  **Step 2: Implement Input Generator (`dft/inputs.py`).**
    *   Use `ase.io.write(format='espresso-in')`.
    *   Important: Ensure `tprnfor=True` and `tstress=True` are always injected into the input dictionary, as we need forces and stress for MLIP training.

3.  **Step 3: Implement `QERunner` Basic Execution.**
    *   Implement `subprocess.run` calls.
    *   Implement strict path checking for the executable (security).

4.  **Step 4: Implement Output Parsing.**
    *   Use `ase.io.read(format='espresso-out')` if reliable, or write a custom lightweight parser for robust extraction of unconverged steps if needed. *Decision: Start with ASE's parser.*

5.  **Step 5: Implement Recovery Loop.**
    *   Wrap the execution in a `tenacity` retry block or a custom `while` loop that modifies the parameters dictionary upon specific exceptions.

6.  **Step 6: Integrate into Orchestrator.**
    *   Create `orchestration/phases/oracle.py` that instantiates `QERunner` and calls it.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Input Generation:**
    *   Create a dummy `Atoms` object (e.g., H2 molecule).
    *   Generate QE input string.
    *   Assert that `tprnfor=.true.` is present in the text.
*   **Parser Tests:**
    *   Store a real `pw.x` output file (stdout) in `tests/data/`.
    *   Feed it to `_parse_output` and assert the Energy matches the value in the file.
*   **Recovery Logic:**
    *   Mock `subprocess.run` to simulate failure (return code 1 or specific error text) for the first 2 calls, then success on the 3rd.
    *   Assert that the runner tried the fallback parameters.

### 5.2 Integration Testing
*   **Mock Execution:**
    *   Since we cannot assume `pw.x` is installed in the CI environment, we will create a `MockQERunner` that just returns random forces/energies for integration tests of the pipeline.
