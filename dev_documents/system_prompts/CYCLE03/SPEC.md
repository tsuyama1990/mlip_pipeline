# Cycle 03 Specification: Oracle & DFT Interface

## 1. Summary
Cycle 03 implements the "Oracle" component, which provides ground-truth labels (energy, forces, stress) for atomic structures using Density Functional Theory (DFT). We will primarily support Quantum Espresso (QE) via the ASE interface. This cycle is critical for data quality. We must implement a "Self-Healing" mechanism to automatically recover from common DFT failures (e.g., SCF convergence issues) and a "Periodic Embedding" technique to efficiently calculate properties of cluster-like structures within a periodic box.

## 2. System Architecture

### 2.1 File Structure
**Bold** files are to be created or modified in this cycle.

```ascii
.
├── src/
│   └── mlip_autopipec/
│       ├── components/
│       │   ├── oracle/
│       │   │   ├── **base.py**         # Enhanced Base Class
│       │   │   ├── **qe_oracle.py**    # Quantum Espresso Implementation
│       │   │   ├── **vasp_oracle.py**  # VASP Implementation (Stub/Optional)
│       │   │   └── **embedding.py**    # Periodic Embedding Logic
│       ├── domain_models/
│       │   └── **oracle_config.py**    # Config for DFT (pseudopotentials, k-points)
│       └── utils/
│           └── **dft_utils.py**        # Helper functions for input generation
```

## 3. Design Architecture

### 3.1 Oracle Component (`src/mlip_autopipec/components/oracle/`)

*   **`QEOracle` Class**:
    *   Inherits from `BaseOracle`.
    *   Uses `ase.calculators.espresso.Espresso`.
    *   **Input Generation**: Automatically determines k-points based on cell size (k-spacing) rather than fixed grid. Selects pseudopotentials from a configured SSSP library path.
    *   **Self-Healing**: Wraps the calculation in a retry loop.
        *   If `SCF correction failed`, reduce mixing beta (e.g., 0.7 -> 0.3).
        *   If still fails, increase temperature (smearing) or change diagonalization method.

### 3.2 Periodic Embedding (`src/mlip_autopipec/components/oracle/embedding.py`)
When active learning selects a "cluster" or a local region of high uncertainty, we cannot just calculate it in vacuum if we want to learn bulk properties.

*   **Algorithm**:
    1.  Receive a local cluster of atoms (from `Dynamics` component).
    2.  Place it in a sufficiently large box (vacuum padding).
    3.  **Crucial Step**: If the intention is to learn bulk behavior, we might need to embed it in a "host" lattice, but for Cycle 03, we will focus on **"Cluster-in-Box"** approach for surface/cluster data, ensuring the box is large enough to avoid spurious image interactions, OR a **"Supercell"** approach where we carve out a region from a large MD snapshot and treat it as a small periodic system.

### 3.3 Configuration (`OracleConfig`)
*   `calculator_type`: "qe" or "vasp".
*   `pseudopotential_dir`: Path to PP files.
*   `kspacing`: Float (e.g., 0.04 1/Å).
*   `scf_max_steps`: Int.
*   `ncore`: Int (parallelization).

## 4. Implementation Approach

1.  **DFT Utils**: Implement `k_grid_from_spacing(cell, spacing)` in `dft_utils.py`.
2.  **QE Oracle**:
    *   Implement `QEOracle.compute()`.
    *   Use `shutil.which('pw.x')` to check if QE is installed. If not, log warning or fallback to Mock (if in dev mode).
    *   Implement the `try...except` block for `ase.calculators.calculator.CalculationFailed`.
3.  **Self-Healing Logic**:
    *   Create a `ErrorHandler` class that parses the QE error output (stdout) and suggests new parameters.
4.  **Integration**:
    *   Update `Orchestrator` to initialize `QEOracle`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **K-points**: Verify `k_grid_from_spacing` returns [1,1,1] for large cells and larger grids for small cells.
*   **Error Handler**: Feed a fake QE output string containing "convergence not achieved" and verify the handler returns a corrected parameter dict (e.g., `{'mixing_beta': 0.3}`).

### 5.2 Integration Testing
*   **Mock QE**: Since running actual DFT in CI is hard, we will create a `MockQE` that behaves like `QEOracle` (generates input files, "runs" a command) but the command just touches the output file or copies a pre-calculated output.
*   **Real QE (Local)**: If the user has `pw.x`, run a calculation on a Silicon primitive cell. Check if `energy` is negative and `forces` are close to zero.
