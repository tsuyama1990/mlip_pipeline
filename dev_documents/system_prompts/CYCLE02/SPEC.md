# Specification: Cycle 02 - The Oracle (DFT Automation)

## 1. Summary

Cycle 02 focuses on building the "Oracle" component of the system. The Oracle is the source of truth; it is responsible for calculating the quantum mechanical properties (Energy, Forces, Stress) of atomic structures using Density Functional Theory (DFT). In this project, we primarily target **Quantum Espresso (QE)** as the DFT engine.

The core challenge addressed in this cycle is **Robustness**. DFT calculations are notoriously fragile; they can fail due to Self-Consistent Field (SCF) convergence issues, insufficient memory, or poor initial guesses. A "Zero-Config" system cannot expect the user to manually fix every crashed calculation. Therefore, this cycle introduces a **"Self-Healing"** mechanism that detects common failure modes and automatically retries the calculation with adjusted parameters (e.g., increased electron temperature/smearing, reduced mixing beta).

By the end of this cycle, the Orchestrator will be able to send a list of `ase.Atoms` objects to the Oracle and receive back a labelled dataset, with the system autonomously handling the complexity of the underlying physics codes.

## 2. System Architecture

### 2.1 File Structure

Files to be created or modified (bold) in this cycle:

```
mlip-autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── **config_model.py**         # Update with DFTConfig
│       ├── domain_models/
│       │   └── **structure.py**            # Update Structure with labels
│       ├── physics/
│       │   └── oracle/
│       │       ├── __init__.py
│       │       ├── **interface.py**        # Protocol definition
│       │       ├── **manager.py**          # The high-level Oracle class
│       │       └── **espresso.py**         # Quantum Espresso implementation
│       └── utils/
│           └── **parsers.py**              # Helpers for parsing QE output
├── tests/
│   ├── unit/
│   │   └── **test_espresso.py**
│   └── data/
│       └── **qe_outputs/**                 # Sample output files for testing
│           ├── **converged.out**
│           └── **scf_error.out**
└── config.yaml
```

### 2.2 Component Blueprints

#### `src/mlip_autopipec/config/config_model.py` (Update)

```python
class DFTConfig(BaseModel):
    command: str = "pw.x"
    pseudopotentials: Dict[str, str]  # Element -> Filename
    kspacing: float = 0.04            # Inverse distance for K-grid
    ecutwfc: float = 50.0             # Wavefunction cutoff (Ry)
    max_retries: int = 3

class Config(BaseModel):
    # ... previous fields ...
    dft: DFTConfig
```

#### `src/mlip_autopipec/physics/oracle/interface.py`

```python
from typing import Protocol, List
from ase import Atoms

class OracleInterface(Protocol):
    def compute(self, structures: List[Atoms]) -> List[Atoms]:
        """
        Takes a list of structures, performs DFT, and returns them
        with 'calc' attached (Energy/Forces/Stress).
        """
        ...
```

#### `src/mlip_autopipec/physics/oracle/espresso.py`

```python
from ase.calculators.espresso import Espresso
import subprocess

class EspressoRunner:
    def __init__(self, config: DFTConfig):
        self.config = config

    def run_single(self, atoms: Atoms) -> Atoms:
        """Runs a single SCF calculation with self-healing."""
        params = self._get_base_params()

        for attempt in range(self.config.max_retries + 1):
            try:
                return self._execute_calculation(atoms, params)
            except DFTConvergenceError:
                if attempt == self.config.max_retries:
                    raise
                logging.warning(f"SCF failed. Retrying (Attempt {attempt+1})...")
                params = self._adjust_params_for_healing(params)
```

## 3. Design Architecture

### 3.1 The "Self-Healing" Strategy pattern
The logic for recovering from errors is encapsulated within the `EspressoRunner`.
*   **Detection**: The system parses the `stdout` or `CRASH` file of the DFT code. It looks for specific keywords (e.g., `convergence not achieved`, `scalapack error`).
*   **Mitigation Strategies**:
    1.  **Mixing Beta**: If oscillation is detected, reduce `mixing_beta` (e.g., 0.7 -> 0.3).
    2.  **Smearing**: If gap is small or system is metallic, increase `degauss` (smearing width).
    3.  **Algorithm**: Switch from `diagonalization='david'` to `'cg'` (Conjugate Gradient) which is slower but more robust.

### 3.2 Data Abstraction via ASE
We use the **Atomic Simulation Environment (ASE)** as the lingua franca.
*   **Input**: The `Oracle` receives `ase.Atoms`.
*   **Output**: It returns `ase.Atoms` with a `SinglePointCalculator` attached. This standardizes the results (eV for energy, eV/Å for forces), regardless of whether the underlying code (QE) uses Rydbergs or Hartrees.

### 3.3 Pseudopotential Management
The user provides a directory path to pseudopotentials (e.g., SSSP library). The `DFTConfig` verifies that potentials exist for all elements present in the structure before starting any heavy calculation. This prevents the job from queuing for hours only to fail instantly due to a missing file.

## 4. Implementation Approach

### Step 1: Configuration Update
1.  Update `DFTConfig` in `config_model.py`.
2.  Add validation: Ensure `kspacing` is positive. Ensure `command` is safe (no shell injection risks).

### Step 2: Espresso Wrapper (Base)
1.  Create `src/mlip_autopipec/physics/oracle/espresso.py`.
2.  Implement `_generate_input_file(atoms)` method. Use ASE's built-in `write_espresso_in` if possible, or template it for fine-grained control.
3.  Implement `_execute_calculation(atoms)`. Use `subprocess.run` to call `pw.x < input.in > output.out`.

### Step 3: Output Parsing
1.  Implement logic to read the QE output file.
2.  Extract `Energy` (grep `!    total energy`), `Forces` (parse the forces table), and `Stress` (parse stress tensor).
3.  **Crucial**: Implement the error detection regex. Identify lines like `Error in routine c_bands` or `convergence not achieved`.

### Step 4: Self-Healing Logic
1.  Wrap the execution in a loop `for attempt in range(retries):`.
2.  If an error is detected, apply a `params.update(...)` to change settings.
3.  Log every retry action ("Self-healing triggered: Reducing mixing beta to 0.3").

### Step 5: Orchestrator Integration
1.  Update `Orchestrator` to initialize `DFTManager`.
2.  Replace the mock `_run_oracle` method with the real call to `dft_manager.compute(candidates)`.

## 5. Test Strategy

### 5.1 Unit Testing Approach (Min 300 words)
Testing the Oracle without running expensive physics calculations is the main challenge.
*   **Mocking `subprocess`**: We will test `EspressoRunner` by mocking `subprocess.run`. We will provide pre-written strings representing QE output files.
    *   **Success Case**: Feed a string containing `!    total energy = -123.4 Ry` and verify the parser extracts `-123.4 * 13.605` eV correctly.
    *   **Failure Case**: Feed a string containing `convergence not achieved`. Verify that the `_execute_calculation` method raises a custom `DFTConvergenceError`.
*   **Parameter Adjustment**: We will test the healing logic. We will set up a test where the "Mock" fails the first 2 times (raises Error) and succeeds the 3rd time. We verify that the `EspressoRunner` correctly calls the mock 3 times, and that the parameters passed to the input generator are different each time (e.g., `mixing_beta` decreasing).

### 5.2 Integration Testing Approach (Min 300 words)
*   **Mock-Mode Integration**: In the CI environment, we will use a "MockOracle" that bypasses `pw.x` entirely and simply assigns a Lennard-Jones potential energy to the atoms using ASE's `LJ` calculator. This allows us to test the *data flow* (Orchestrator -> Oracle -> Atoms with Labels) without needing the binary.
*   **Real-Mode Integration (Local)**: A developer test script will be provided (`tests/manual/test_real_qe.py`). This script attempts to run a calculation on a single Hydrogen atom or a simple Silicon unit cell using the locally installed `pw.x`. This verifies that the input files generated are syntactically correct and accepted by the actual version of Quantum Espresso installed.
*   **Stress Test**: We will create a test case with a "nasty" structure (atoms very close together). We expect the initial SCF to fail. We assert that the system logs a retry and eventually either succeeds or fails gracefully with a permanent error, without hanging indefinitely.
