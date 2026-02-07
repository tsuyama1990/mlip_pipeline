# Cycle 03 Specification: Oracle & DFT Automation

## 1. Summary
This cycle implements the `Oracle` component, responsible for generating ground-truth data (energy, forces, stress) using Density Functional Theory (DFT). The system will initially support **Quantum Espresso (QE)** via the ASE interface. A key feature is the **Self-Healing Mechanism**, which automatically adjusts calculation parameters (mixing beta, smearing, etc.) upon convergence failure, ensuring robust data generation without manual intervention.

## 2. System Architecture

### 2.1. File Structure
The following file structure must be created/modified. **Bold** files are to be implemented in this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── oracle/
│   │   ├── **__init__.py**
│   │   ├── **base.py**     # Base Oracle
│   │   ├── **dft_manager.py** # High-level DFT Manager
│   │   ├── **qe.py**       # Quantum Espresso Implementation
│   │   └── **vasp.py**     # Placeholder for VASP
│   └── **embedding.py**    # Periodic Embedding Logic
tests/
    ├── **test_oracle.py**
    └── **test_embedding.py**
```

### 2.2. Class Blueprints

#### `src/mlip_autopipec/components/oracle/dft_manager.py`
```python
from typing import Iterator
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.components.oracle.base import BaseOracle

class DFTManager(BaseOracle):
    def __init__(self, config: OracleConfig):
        self.config = config
        self.calculator = self._init_calculator(config)

    def compute(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        for s in structures:
            try:
                yield self._run_single(s)
            except CalculationError:
                yield self._retry_with_healing(s)
```

## 3. Design Architecture

### 3.1. Oracle Configuration
*   **`OracleConfig`**:
    *   `type`: "qe" or "vasp".
    *   `command`: Path to `pw.x` executable.
    *   `pseudo_dir`: Path to pseudopotentials.
    *   `kspacing`: Float (e.g., 0.04 Å^-1) for automatic K-point grid generation.
    *   `smearing_width`: Float (e.g., 0.02 Ry).

### 3.2. Self-Healing Strategies
When an SCF cycle fails to converge:
1.  **Reduce Mixing Beta**: `mixing_beta` 0.7 -> 0.3 -> 0.1.
2.  **Increase Smearing**: `degauss` 0.02 -> 0.05.
3.  **Change Diagonalization**: `david` -> `cg`.

### 3.3. Periodic Embedding
For isolated clusters or "active set" structures carved out from MD:
*   Place the cluster in a vacuum box.
*   Ensure the box is large enough to avoid spurious image interactions (min vacuum ~10 Å).
*   Convert to a periodic supercell for plane-wave DFT codes.

## 4. Implementation Approach

1.  **QE Interface**: Use `ase.calculators.espresso.Espresso`.
2.  **Retry Logic**: Implement a `retry` decorator or loop inside `_run_single`.
3.  **K-Point Generation**: Implement `kspacing_to_grid(atoms, spacing)` helper.
4.  **Embedding**: Implement `embed_in_box(atoms, vacuum)` function.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_embedding.py`**:
    *   Verify that `embed_in_box` correctly adds vacuum and sets PBC to True.
*   **`test_oracle.py` (Mocked QE)**:
    *   Mock `subprocess.run` or `ase.calculators.espresso.Espresso`.
    *   Simulate a "Convergence Error" (raise `ase.calculators.calculator.CalculationFailed`).
    *   Verify that `_retry_with_healing` is called and parameters are updated.

### 5.2. Integration Testing
*   **Real DFT (Optional/CI-dependent)**:
    *   If `pw.x` is available, run a calculation on a single H2 molecule.
    *   Verify energy and forces are returned.
