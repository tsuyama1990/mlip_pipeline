# Cycle 02 Specification: The Oracle (DFT Automation)

## 1. Summary

In Cycle 02, we replace the `MockOracle` with a production-grade interface to **Quantum Espresso**, the `EspressoOracle`. The Oracle is responsible for taking a list of atomic structures (candidates), computing their ground-truth energies, forces, and stresses via Density Functional Theory (DFT), and returning the labeled dataset.

This cycle focuses on **robustness**. DFT calculations often fail due to SCF non-convergence, especially for the high-energy, distorted structures generated during active learning. The Oracle must implement a **Self-Healing** mechanism that automatically adjusts calculation parameters to salvage these jobs. Additionally, we implement **Periodic Embedding** to correctly handle cluster structures cut from MD simulations.

## 2. System Architecture

Files to create/modify are **bolded**.

```ascii
src/mlip_autopipec/
├── config/
│   └── config_model.py         # Update OracleConfig
├── infrastructure/
│   └── oracle/
│       ├── **__init__.py**
│       └── **espresso_oracle.py**  # Implementation of EspressoOracle
└── utils/
    └── **physics.py**          # Periodic Embedding logic
```

## 3. Design Architecture

### 3.1. Oracle Configuration
We extend `OracleConfig` in `config_model.py`:
*   `type`: Literal["mock", "espresso"]
*   `command`: str (e.g., "mpirun -np 4 pw.x")
*   `pseudo_dir`: Path
*   `pseudopotentials`: Dict[str, str] (Map element to filename)
*   `kspacing`: float (Target reciprocal space density, e.g., 0.04)
*   `scf_params`: Dict[str, Any] (Default overrides, e.g., `mixing_beta`)

### 3.2. EspressoOracle Logic
*   **Inheritance**: Inherits from `BaseOracle`.
*   **Dependencies**: Uses `ase.calculators.espresso.Espresso`.
*   **Self-Healing**: Wraps the `get_potential_energy()` call in a loop. If `ase.calculators.calculator.CalculationFailed` is raised:
    1.  Catch the error.
    2.  Consult `RecoveryStrategy` to get the next set of parameters (e.g., Reduce beta from 0.7 -> 0.3).
    3.  Update calculator and retry.
    4.  If all recipes fail, discard the structure.

### 3.3. Periodic Embedding
Located in `utils/physics.py`.
*   **Input**: An `Atoms` object (cluster) that may have vacuum or non-periodic boundary conditions.
*   **Logic**:
    1.  Determine the bounding box of the atoms.
    2.  Add a buffer (e.g., 5-8 Angstroms) to each side.
    3.  Create a new Orthorhombic cell.
    4.  Center the atoms.
    5.  Set `pbc=[True, True, True]`.

## 4. Implementation Approach

1.  **Physics Utilities**: Implement `embed_structure` in `utils/physics.py`.
2.  **Config Update**: Add fields to `OracleConfig` to support QE settings.
3.  **Oracle Skeleton**: Create `EspressoOracle` class.
4.  **ASE Integration**: Implement the `compute` method using `ase.calculators.espresso`.
    *   Ensure `tprnfor=True` and `tstress=True` are always set.
    *   Implement logic to convert `kspacing` to `kpts` grid (using `ase.calculators.calculator.kpts2mp`).
5.  **Recovery Loop**: Implement the `try...except` block and the list of "Recovery Recipes".
    *   Recipe 1: Default.
    *   Recipe 2: `mixing_beta` = 0.3.
    *   Recipe 3: `smearing` = 'mv', `degauss` = 0.02.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Embedding**: Create a 2-atom cluster. Run `embed_structure`. Assert cell size is correct and atoms are centered.
*   **K-points**: Test logic that converts `kspacing` to `kpts` tuple.

### 5.2. Integration Testing (Simulated QE)
We cannot rely on `pw.x` being present in the CI environment. We will **Mock the ASE Calculator** inside the `EspressoOracle` test.
*   **The "Mock Calculator"**: Create a class that inherits from `Espresso` but overrides `run()`.
*   **Scenario: Recovery**:
    *   On 1st call, raise `CalculationFailed`.
    *   On 2nd call (with `mixing_beta=0.3`), return success.
    *   Assert that `EspressoOracle` correctly retried and returned the result.
