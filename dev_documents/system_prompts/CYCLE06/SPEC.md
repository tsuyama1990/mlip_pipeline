# Cycle 06 Specification: Validation & kMC Integration

## 1. Summary
Cycle 06 focuses on the final critical components: Quality Assurance (Validation) and Long-Timescale Dynamics (Adaptive Kinetic Monte Carlo).

Key features:
1.  **Physics Validation**: A comprehensive suite of tests to verify the physical correctness of the trained potential. This includes Phonon dispersion stability (no imaginary modes), Equation of State (EOS) curves, and Elastic Constants ($C_{ij}$).
2.  **Adaptive kMC Integration**: Integration with the EON software to perform long-timescale simulations (diffusion, ripening) that are inaccessible to standard MD.
3.  **Reporting**: Automatic generation of HTML validation reports with plots (Parity, Phonon bands, EOS) for user inspection.

## 2. System Architecture

The file structure expands `src/pyacemaker/validator` and `src/pyacemaker/dynamics`. **Bold files** are new.

```text
src/
└── pyacemaker/
    ├── core/
    │   └── **config.py**       # Updated with ValidatorConfig/EONConfig
    ├── dynamics/
    │   └── **kmc.py**          # EON Wrapper
    └── **validator/**
        ├── **__init__.py**
        ├── **manager.py**      # Validation Orchestrator
        ├── **physics.py**      # Physics Checks (Phonon, EOS, Elastic)
        └── **report.py**       # HTML Report Generator
```

### File Details
-   `src/pyacemaker/validator/manager.py`: The `Validator` class. It runs defined checks and aggregates results.
-   `src/pyacemaker/validator/physics.py`: Contains specific check logic using `phonopy` and `ase`.
-   `src/pyacemaker/validator/report.py`: Uses `jinja2` and `matplotlib` to create visual reports.
-   `src/pyacemaker/dynamics/kmc.py`: `EONWrapper` to configure and run EON client.

## 3. Design Architecture

### 3.1. Validator Configuration
```python
class ValidatorConfig(BaseModel):
    test_set_ratio: float = 0.1
    phonon_supercell: List[int] = [2, 2, 2]
    eos_strain: float = 0.1
    elastic_strain: float = 0.01
```

### 3.2. Validation Result
```python
class ValidationResult(BaseModel):
    passed: bool
    metrics: Dict[str, float]  # RMSE_E, RMSE_F
    phonon_stable: bool
    elastic_stable: bool
    artifacts: List[Path]      # Paths to plots (png)
```

### 3.3. EON Wrapper
```python
class EONWrapper:
    def run_search(self, initial_state: Atoms, potential: Path):
        # Generate EON config.ini
        # Run eonclient
        pass
```

## 4. Implementation Approach

### Step 1: Update Configuration
-   Modify `src/pyacemaker/core/config.py` to add `ValidatorConfig` and `EONConfig`.

### Step 2: Physics Validation
-   Implement `src/pyacemaker/validator/physics.py`.
-   **Phonon**: Use `phonopy.Phonopy` API. Calculate band structure. Check for $\omega^2 < -\epsilon$.
-   **EOS**: Use `ase.eos.EquationOfState`. Fit Birch-Murnaghan. Check Bulk Modulus.
-   **Elastic**: Apply small strains, compute stress, fit linear elasticity. Check Born stability criteria.

### Step 3: Reporting
-   Implement `src/pyacemaker/validator/report.py`.
-   Create a Jinja2 template for the report.
-   Generate Matplotlib plots for EOS and Phonons.

### Step 4: EON Integration
-   Implement `src/pyacemaker/dynamics/kmc.py`.
-   Create a helper script `pace_driver.py` that EON calls to get energy/forces from the `.yace` potential.
-   Implement the `run_search` method.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Physics Logic (`tests/validator/test_physics.py`)**:
    -   Mock `phonopy`.
    -   Verify that imaginary frequencies trigger `phonon_stable=False`.
    -   Verify that Born criteria logic is correct for cubic systems.
-   **EON Wrapper (`tests/dynamics/test_kmc.py`)**:
    -   Mock `subprocess.run`.
    -   Verify `config.ini` generation.
    -   Verify `pace_driver.py` is created correctly.

### 5.2. Integration Testing
-   **Validation Pipeline (`tests/validator/test_manager.py`)**:
    -   Mock Physics checks.
    -   Run `Validator.validate(potential, test_set)`.
    -   Verify that a `report.html` is generated.
