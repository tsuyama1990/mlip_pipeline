# Cycle 03 Specification: Oracle (DFT Automation)

## 1. Summary

Cycle 03 implements the **Oracle** module, the component responsible for generating the "Ground Truth" data required to train the Machine Learning Interatomic Potential (MLIP). This module interfaces with Density Functional Theory (DFT) codes, primarily **Quantum Espresso (QE)** and **VASP**.

The key innovations in this cycle are:
1.  **Robust Automation**: The system automatically generates input files based on the structure and specified accuracy (k-points, smearing, pseudopotentials).
2.  **Self-Healing Logic**: DFT calculations are prone to failure (SCF non-convergence). The Oracle implements a "Doctor" that diagnoses common errors (e.g., charge sloshing) and automatically retries with adjusted parameters (e.g., mixing beta, electronic temperature) without user intervention.
3.  **Periodic Embedding**: When the Active Learning loop identifies a local high-uncertainty region in a large MD simulation, it is inefficient to run DFT on the entire system. We implement a "Periodic Embedding" strategy to cut out a cluster around the target atom, pad it with a buffer, and embed it into a smaller, periodic supercell that is computationally tractable for DFT.

## 2. System Architecture

Files in **bold** are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **config.py**         # Update OracleConfig (DFT settings)
├── oracle/
│   ├── **__init__.py**
│   ├── **interface.py**      # Enhanced BaseOracle
│   ├── **dft_manager.py**    # Main DFT Controller
│   ├── **qe_driver.py**      # Quantum Espresso Interface
│   ├── **vasp_driver.py**    # VASP Interface (Stub)
│   ├── **self_healing.py**   # Error recovery logic
│   └── **embedding.py**      # Cluster cutting & embedding
└── tests/
    └── unit/
        └── **test_oracle.py**
```

## 3. Design Architecture

### 3.1 DFT Manager (`oracle/dft_manager.py`)
The `DFTManager` orchestrates the calculation process. It takes a list of `Structure` objects and returns a `Dataset`.
*   **Parallelism**: It manages a pool of workers (using `concurrent.futures` or `dask`) to run multiple DFT calculations in parallel, respecting the available CPU/GPU resources.
*   **Driver Selection**: It selects the appropriate driver (`QEDriver`, `VASPDriver`) based on `OracleConfig.code`.

### 3.2 Self-Healing Logic (`oracle/self_healing.py`)
The `SelfHealing` class wraps the execution of a calculation.
*   **Mechanism**: It uses a `while` loop with a `max_retries` counter.
*   **Diagnosis**: It parses the `stderr` or log file for specific error keywords (e.g., "convergence not achieved").
*   **Prescription**: It applies a sequence of fixes:
    1.  Increase `electron_maxstep` (more iterations).
    2.  Decrease `mixing_beta` (slower but safer mixing).
    3.  Increase `smearing` (higher electronic temperature).
    4.  Change `diagonalization` algorithm (e.g., david -> cg).

### 3.3 Periodic Embedding (`oracle/embedding.py`)
The `Embedding` class handles the conversion from a large, non-periodic or defective supercell to a small, periodic DFT cell.
*   **Input**: `center_atom_index`, `cutoff_radius`, `buffer_radius`.
*   **Output**: `ase.Atoms` object representing the embedded cluster.
*   **Constraints**: The box size must be at least `2 * (cutoff + buffer)` to avoid self-interaction images.

## 4. Implementation Approach

1.  **Enhance Domain Models**: Update `OracleConfig` to include DFT parameters (k-spacing, encut, pseudos).
2.  **Implement QEDriver**: Use `ase.calculators.espresso` or direct input file generation. Crucially, ensure `tprnfor=.true.` and `tstress=.true.` are set for force/stress output.
3.  **Implement SelfHealing**: Create a wrapper function `run_with_healing(calc, atoms)` that catches exceptions and retries.
4.  **Implement PeriodicEmbedding**: Write the geometric logic to select atoms within a radius and define a new unit cell.
5.  **Integration**: Update `Orchestrator` to use the real `DFTManager` instead of the Mock (when config allows).

## 5. Test Strategy

### 5.1 Unit Testing
*   **Embedding Test**: Create a large dummy lattice. Select an atom and verify the cut cluster contains the correct neighbors and has valid periodic boundary conditions.
*   **Self-Healing Test**: Mock a calculator that fails twice with "convergence error" and succeeds on the third try. Verify the logic applies the correct fixes and eventually returns the result.

### 5.2 Integration Testing
*   **QE Integration**: (Requires QE installed in CI or local) Run a simple SCF calculation on a water molecule or Si bulk. Verify energies and forces are extracted correctly.
