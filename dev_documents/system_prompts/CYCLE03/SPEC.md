# Cycle 03 Specification: Oracle (DFT Automation)

## 1. Summary

Cycle 03 introduces the **Oracle** module, which serves as the "Ground Truth" generator for the system. Its primary responsibility is to take candidate structures (proposed by the Generator or Dynamics Engine) and compute their accurate energy, forces, and stress tensors using Density Functional Theory (DFT).

This cycle focuses on robust automation. DFT calculations are notorious for convergence failures and sensitivity to parameters. The Oracle must implement **Self-Healing Logic** to automatically retry failed calculations with adjusted parameters (e.g., increased smearing or reduced mixing beta). Additionally, it implements **Periodic Embedding**, a critical technique for cutting out small, periodic clusters from larger simulations to enable efficient local learning without boundary artifacts.

## 2. System Architecture

We expand the `components/oracle` module and introduce DFT-specific configurations.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```
.
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **config.py**      # Add OracleConfig (DFT parameters)
│       │   └── **results.py**     # Add SinglePointResult (Energy, Forces, Stress)
│       └── components/
│           ├── **oracle/**
│           │   ├── **__init__.py**
│           │   ├── **base.py**        # BaseOracle (Abstract)
│           │   ├── **dft_manager.py** # The main controller
│           │   ├── **calculators/**   # ASE Wrappers
│           │   │   ├── **espresso.py** # Quantum Espresso interface
│           │   │   └── **vasp.py**     # (Optional) VASP interface
│           │   └── **embedding.py**   # Periodic Embedding logic
│           └── **factory.py**         # Update to include DFTOracle
```

### Key Components
1.  **DFTManager (`src/mlip_autopipec/components/oracle/dft_manager.py`)**: The high-level interface. It takes a list of structures, manages the calculation queue (possibly parallelized), and returns the results. It orchestrates the self-healing loops.
2.  **Espresso Calculator (`src/mlip_autopipec/components/oracle/calculators/espresso.py`)**: A wrapper around `ase.calculators.espresso`. It handles input file generation (`pw.x` input) and output parsing.
3.  **Periodic Embedding (`src/mlip_autopipec/components/oracle/embedding.py`)**: Implements the logic to take a local cluster (e.g., around a defect), place it in a vacuum, and then wrap it in a minimal periodic box that respects the cut-off radius $R_{cut}$ and buffer $R_{buffer}$.

## 3. Design Architecture

### 3.1. Domain Models
*   **OracleConfig**:
    *   `calculator_type`: "espresso" or "vasp" (Enum).
    *   `command`: The shell command to run the binary (e.g., `mpirun -np 4 pw.x`).
    *   `kspacing`: Density of k-points (inverse Angstrom) instead of fixed grid.
    *   `scf_params`: Dict of parameters (ecutwfc, mixing_beta, etc.).
*   **SinglePointResult**: Pydantic model storing:
    *   `energy`: Float (eV).
    *   `forces`: Array (eV/Angstrom).
    *   `stress`: Array (Voigt notation, GPa).
    *   `converged`: Boolean.

### 3.2. Self-Healing Logic
The `DFTManager` implements a `retry` mechanism with a strategy pattern.
*   **Strategy 1 (Default)**: Use standard parameters.
*   **Strategy 2 (Mixing)**: If SCF fails, reduce `mixing_beta` (e.g., 0.7 -> 0.3).
*   **Strategy 3 (Smearing)**: If still failing, increase `smearing` width (electronic temperature).
*   **Strategy 4 (Algorithm)**: Switch diagonalization algorithm (e.g., `david` -> `cg`).

### 3.3. Periodic Embedding
The core algorithm:
1.  Identify "active" atoms (high uncertainty or interest).
2.  Select neighbors within $R_{cut} + R_{buffer}$.
3.  Create a new `Atoms` object with these atoms.
4.  Construct a bounding box that ensures the image distance is $> 2 \times (R_{cut} + R_{buffer})$.
5.  Set periodic boundary conditions (PBC) to True.
6.  (Crucially) Mask forces on buffer atoms during training (this is handled in the Trainer, but the Oracle must preserve the "active" tags).

## 4. Implementation Approach

1.  **Dependencies**: Ensure `ase` is installed.
2.  **Domain Models**: Add `OracleConfig` and `SinglePointResult`.
3.  **Embedding Logic**: Implement `embedding.py` first as it's pure logic. Test it with simple cubic lattices.
4.  **ASE Wrapper**: Implement `Espresso` wrapper. Focus on the `get_potential_energy()` call. Use `ase.io.write` to generate inputs and `subprocess` to run the command if ASE's internal run is insufficient.
5.  **DFT Manager**: Implement the loop over structures. Add the `try-except` block for `ase.calculators.calculator.PropertyNotImplementedError` or convergence errors.
6.  **Self-Healing**: Implement the parameter adjustment logic. This requires creating a new Calculator instance with modified parameters for each retry.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Embedding**: Create a dummy cluster. Verify the generated supercell has the correct size and PBC. Ensure atom indices are tracked correctly.
*   **Self-Healing**: Mock the `Calculator.get_potential_energy` method to raise an exception on the first call and succeed on the second. Verify that the manager retries with new parameters.

### 5.2. Integration Testing
*   **Mock DFT**: Since we can't rely on `pw.x` being installed in CI, we will use the **MockOracle** (from Cycle 01) extended to simulate convergence failures (probabilistically).
*   **Real DFT (Local)**: If `pw.x` is available, run a tiny calculation (e.g., one H2 molecule) to verify the file I/O and command execution.
