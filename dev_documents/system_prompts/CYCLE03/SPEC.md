# Cycle 03 Specification: The Robust Oracle (DFT)

## 1. Summary

In Cycle 03, we replace `MockOracle` with a production-grade interface to First-Principles codes. The `Oracle` is responsible for assigning "Ground Truth" labels (Energy, Forces, Stress) to the candidate structures.

Crucially, DFT calculations are prone to numerical instability (SCF non-convergence). A key feature of this cycle is the **Self-Healing Mechanism**. The Oracle doesn't just run `pw.x`; it monitors the execution. If a calculation fails, it analyzes the error (e.g., "convergence not achieved") and automatically retries with "safer" parameters (e.g., increased smearing, reduced mixing beta, changed diagonalization algorithm).

We also implement **Periodic Embedding**. When an interesting local structure is detected (e.g., a dislocation core), we cannot simply calculate it in vacuum. We must cut out a cluster and embed it in a periodic supercell that respects the boundary conditions, ensuring the forces are physically meaningful.

## 2. System Architecture

```ascii
src/mlip_autopipec/
├── ...
├── oracle/
│   ├── __init__.py
│   ├── **dft_manager.py**   # Main Entry point
│   ├── **calculators.py**   # Wrappers for QE/VASP
│   ├── **embedding.py**     # Periodic Embedding Logic
│   └── **healing.py**       # Error analysis and recovery
└── ...
```

## 3. Design Architecture

### 3.1. DFT Manager & Calculators (`dft_manager.py`)
*   **`DFTManager`**: The facade. Accepts a list of `Structure`, distributes them (conceptually) to workers, and returns labeled structures.
*   **`EspressoCalculator`**: A specialized wrapper around `ase.calculators.espresso`. It handles the generation of `input.pw` with strict settings (`tprnfor=True`, `tstress=True`).

### 3.2. Self-Healing Logic (`healing.py`)
*   **`HealingStrategy`**: A state machine.
    *   State 0: Standard params.
    *   Fail -> State 1: `mixing_beta = 0.3`.
    *   Fail -> State 2: `smearing = 'mv', width = 0.02`.
    *   Fail -> State 3: `diagonalization = 'cg'`.
    *   Fail -> Give up (Drop structure).

### 3.3. Periodic Embedding (`embedding.py`)
*   **Problem**: We have a huge MD snapshot (10,000 atoms). We only want forces for the 50 atoms around a defect.
*   **Algorithm**:
    1.  Select central atom.
    2.  Select neighbors within $R_{cut} + R_{buffer}$.
    3.  Define an Orthorhombic Box that bounds these atoms.
    4.  Pad with vacuum (if surface) or replicate (if bulk) - *Simplified for Cycle 03: Extract cluster and place in a new periodic box with sufficient vacuum/padding to minimize image interaction, or strictly enforce PBC if extracting from bulk.*

## 4. Implementation Approach

1.  **Develop Embedding**: Implement `ClusterExtractor`. Ensure it preserves element types and handles PBC wrap-around correctly.
2.  **ASE Integration**: Implement the `EspressoCalculator` class. Add validation to ensure `pseudopotentials` are correctly mapped.
3.  **Implement Healing**: Create the retry loop in `DFTManager`.
    *   `try: calc.get_potential_energy()`
    *   `except DFTError as e: params = healer.suggest_fix(e); retry()`
4.  **Orchestrator Update**: Inject the real `DFTManager`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Embedding Test**: Extract a cluster from the center of a 4x4x4 FCC supercell. Verify the neighbor distances are preserved.
*   **Healer Test**: Simulate a `OneElectronOperatorError`. Verify the healer returns a new param dictionary with `mixing_beta` reduced.

### 5.2. Integration Testing
*   **Pseudo-DFT Run**: Since we might not have `pw.x` in the test environment, we use a **"Mock DFT Binary"** approach (or ASE's `EMT` calculator wrapped to look like DFT) for CI.
*   **Real-DFT Run (Local)**: If `pw.x` is available, run a static calculation on a water molecule. Verify the output parsing (forces are not zero).
