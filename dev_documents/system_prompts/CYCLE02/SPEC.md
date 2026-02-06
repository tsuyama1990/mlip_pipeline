# Cycle 02 Specification: Oracle Module (DFT Integration)

## 1. Summary
This cycle implements the **Oracle** component, which is responsible for generating ground-truth data (Energy, Forces, Virials) using Density Functional Theory (DFT). We will implement a concrete adapter for **Quantum Espresso (QE)** using the Atomic Simulation Environment (ASE). Key features include robust error handling (e.g., automatically retrying calculations if SCF fails), automatic management of pseudopotentials (SSSP), and intelligent k-point density selection. This module bridges the gap between the abstract `StructureMetadata` used by the Orchestrator and the complex input files required by QE.

## 2. System Architecture

### 2.1. File Structure

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── **config_model.py**     # [MODIFY] Add detailed OracleConfig
│       ├── infrastructure/
│       │   ├── **espresso/**
│       │   │   ├── **__init__.py**
│       │   │   ├── **adapter.py**      # [NEW] EspressoOracle implementation
│       │   │   └── **recovery.py**     # [NEW] Self-healing logic
└── tests/
    └── unit/
        └── **test_oracle.py**          # [NEW] Tests for EspressoOracle
```

## 3. Design Architecture

### 3.1. `OracleConfig` (Pydantic)
We expand the configuration model to support DFT specifics.
*   `command`: str (e.g., `mpirun -np 4 pw.x`).
*   `pseudo_dir`: Path.
*   `pseudopotentials`: Dict[str, str] (Element -> Filename).
*   `kspacing`: float (Inverse distance, e.g., 0.04 A^-1).
*   `scf_params`: Dict (mixing_beta, electron_maxstep, etc.).

### 3.2. `EspressoOracle` Class
Implements `BaseOracle`.
*   **Responsibilities**:
    1.  Receive a list of `StructureMetadata` (ASE Atoms).
    2.  Check if calculation is already done (hash check).
    3.  Configure `ase.calculators.espresso.Espresso`.
    4.  Run calculation.
    5.  Catch `ase.Calculators.CalculatorError`.
    6.  If error, trigger `RecoveryStrategy`.
    7.  Return labeled structures.

### 3.3. Self-Healing (`RecoveryStrategy`)
A simple state machine to handle convergence failures.
*   **Attempt 1**: Default settings.
*   **If Fail**: Reduce `mixing_beta` (0.7 -> 0.3).
*   **If Fail**: Increase `smearing`.
*   **If Fail**: Raise `OracleError` (skip this structure).

## 4. Implementation Approach

1.  **Update Config**: Modify `config_model.py` to include `OracleConfig`.
2.  **Implement Adapter**: Create `infrastructure/espresso/adapter.py`.
    *   Use `ase.calculators.espresso.Espresso`.
    *   Map `kspacing` to `kpts` grid (using `ase.calculators.calculator.kpts2mp`).
    *   Ensure `tprnfor=True` and `tstress=True` are always set (we need forces/stress).
3.  **Implement Recovery**: Create `infrastructure/espresso/recovery.py`.
    *   Define a list of "recovery recipes" (dictionaries of parameter overrides).
    *   Wrap the `get_potential_energy()` call in a loop that iterates through these recipes.
4.  **Integration**: Update `main.py` (CLI) to instantiate `EspressoOracle` instead of `MockOracle` if configured.

## 5. Test Strategy

### 5.1. Unit Testing (`test_oracle.py`)
Since we cannot assume `pw.x` is installed on the CI machine, we must **Mock** the actual execution.

*   **Mocking ASE**: Use `unittest.mock.patch` to replace `ase.calculators.espresso.Espresso`.
*   **Test Input Generation**:
    *   Call `oracle.label(atoms)`.
    *   Inspect the Mock's `write_input()` arguments or the generated `espresso.pwi` file content.
    *   **Assert**: `k_points` are correct for a given cell size.
    *   **Assert**: `tprnfor=.true.` is present.
*   **Test Recovery Logic**:
    *   Configure the Mock to raise `CalculatorError` on the first call.
    *   Configure it to succeed on the second call.
    *   **Assert**: `label()` returns successfully.
    *   **Assert**: The second call used the modified parameters (e.g., lower beta).

### 5.2. Integration Testing
*   **Dry Run**: If `pw.x` is available, run a tiny calculation (H2 molecule).
*   **Fallback**: If not available, skip with `pytest.mark.skipif`.
