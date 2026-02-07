# Cycle 02 Specification: Data Generation & DFT Integration

## 1. Summary
Cycle 02 focuses on replacing the mock components for data generation with real physics engines. We will implement the `DFTManager` (Oracle) using the Atomic Simulation Environment (ASE) to interface with Quantum Espresso (QE). Additionally, we will implement the first real `StructureGenerator` strategy: `RandomDisplacement`, which creates perturbed structures to sample the local potential energy surface. Crucially, this cycle introduces "Self-Healing" capabilities for DFT calculations and "Periodic Embedding" to handle the efficient labeling of large structures.

## 2. System Architecture

### File Structure
Files to be created/modified are marked in **bold**.

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **config.py**       # Add OracleConfig (QE parameters)
│       │   └── **structure.py**    # Add periodic embedding methods
│       ├── infrastructure/
│       │   ├── oracle/
│       │   │   ├── **__init__.py**
│       │   │   └── **dft_manager.py**  # Real Oracle implementation
│       │   ├── generator/
│       │   │   ├── **__init__.py**
│       │   │   └── **strategies.py**   # RandomDisplacement, Supercell
│       └── utils/
│           └── **physics.py**      # Unit conversions, k-point grid calc
└── tests/
    └── integration/
        └── **test_dft_pipeline.py**
```

## 3. Design Architecture

### Domain Models (`domain_models/`)

-   **`OracleConfig`**:
    -   `calculator_type`: Literal["qe", "vasp"]
    -   `command`: str (e.g., "mpirun -np 4 pw.x")
    -   `pseudo_dir`: Path
    -   `pseudopotentials`: Dict[str, str] (e.g., {"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"})
    -   `kspacing`: float (Default: 0.04)
    -   `smearing_width`: float (Default: 0.02)

-   **`Structure`**:
    -   `apply_periodic_embedding(center: np.ndarray, radius: float, buffer: float) -> Structure`: Creates a supercell containing only atoms within `radius + buffer` of `center`, ensuring minimal image convention.

### Infrastructure (`infrastructure/`)

-   **`DFTManager` (implements `BaseOracle`)**:
    -   `compute_batch(structures: List[Structure]) -> List[Structure]`: Parallel execution of DFT tasks.
    -   `_run_single(structure: Structure) -> Structure`:
        -   Generates input file (using `ase.io.write`).
        -   Executes command.
        -   Parses output.
        -   **Self-Healing**: If SCF fails, reduce `mixing_beta` and retry.

-   **`RandomDisplacement` (implements `BaseStructureGenerator`)**:
    -   `generate(structure: Structure, strategy="random_displacement", magnitude=0.01) -> List[Structure]`: Perturbs atomic positions by a random vector.

## 4. Implementation Approach

1.  **Enhance Config**: Update `domain_models/config.py` to include `OracleConfig` with strict validation for file paths.
2.  **Implement Physics Utils**: Add helper functions in `utils/physics.py` to calculate k-point grids from cell dimensions and k-spacing.
3.  **Implement DFTManager**:
    -   Use `ase.calculators.espresso.Espresso` as the driver.
    -   Implement the retry loop for convergence errors (catch `ase.calculators.calculator.CalculationFailed`).
    -   Implement strict output parsing (Energy, Forces, Stress).
4.  **Implement Structure Generator**:
    -   Implement `strategies.py` using `numpy.random`.
    -   Ensure the generator returns deep copies of the structure to avoid side effects.
5.  **Mock Integration**: Update `main.py` to allow selecting `type: qe` in the config, even if `pw.x` is not installed (it should fail gracefully or check for binary existence).

## 5. Test Strategy

### Unit Testing (`tests/unit/`)
-   **`test_dft_manager.py`**:
    -   Mock `subprocess.run` to simulate `pw.x` execution.
    -   Verify that `compute_batch` correctly generates `input.pwi` with the parameters from `OracleConfig`.
    -   Verify that the "Self-Healing" logic retries with lower `mixing_beta` when a specific error string is returned.

### Integration Testing (`tests/integration/`)
-   **`test_dft_pipeline.py`**:
    -   Requires `pw.x` (mocked via a shell script in the test environment if not available).
    -   Create a minimal structure (H2 molecule).
    -   Run `DFTManager.compute`.
    -   Assert that the returned structure has `energy` and `forces` properties populated.
