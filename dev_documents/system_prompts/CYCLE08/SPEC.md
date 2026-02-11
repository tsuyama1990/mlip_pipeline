# Cycle 08 Specification: Validator & Quality Gate

## 1. Summary

Cycle 08 implements the **Validator**, the final quality control gate. Before a potential is marked as "Production Ready", it must pass a series of physical tests. These tests ensure that the potential not only reproduces the training data (RMSE) but also captures fundamental physical properties like lattice stability (Phonons), mechanical stiffness (Elastic Constants), and volume response (Equation of State).

We also implement a **Report Generator** that compiles these metrics into a user-friendly HTML report.

## 2. System Architecture

### File Structure

Files in **bold** are new or modified in this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── validator/
│   │   ├── **__init__.py**
│   │   ├── **base.py**          # BaseValidator class
│   │   ├── **phonon.py**        # Phonon calculation (Phonopy wrapper)
│   │   ├── **elastic.py**       # Elastic constant calculation
│   │   ├── **eos.py**           # Equation of State calculation
│   │   └── **report.py**        # HTML report generator
│   └── mock.py
├── domain_models/
│   ├── **validation.py**        # ValidationResult models
│   └── config.py                # Updated with ValidatorConfig
└── core/
    └── orchestrator.py          # Updated to use Validator
```

### Component Interaction
1.  **Orchestrator** calls `validator.validate(potential, structure)`.
2.  **Validator** runs sub-modules:
    -   `phonon.check_stability()`: Returns True if no imaginary frequencies.
    -   `elastic.calculate_tensor()`: Returns Cij matrix and Born stability boolean.
    -   `eos.fit_birch_murnaghan()`: Returns Bulk Modulus (B0).
3.  **Validator** aggregates results into `ValidationResult`.
4.  **Reporter** generates `report.html` with plots.

## 3. Design Architecture

### 3.1. Validator Configuration (`domain_models/config.py`)

```python
class ValidatorConfig(BaseComponentConfig):
    phonon_supercell: List[int] = [2, 2, 2]
    elastic_strain_range: float = 0.01
    eos_vol_range: float = 0.1
```

### 3.2. Validation Result Model (`domain_models/validation.py`)

```python
class ValidationResult(BaseModel):
    passed: bool
    phonon_stable: bool
    elastic_stable: bool
    bulk_modulus: float
    metrics: Dict[str, Any]
    report_path: Path
```

### 3.3. Phonon Calculation (`components/validator/phonon.py`)

Wrapper around `phonopy` (if installed) or a simplified frozen phonon approach using ASE.

```python
def check_stability(atoms, potential):
    # Calculate force constants
    # Build dynamical matrix
    # Get eigenvalues
    # If any eigenvalue < -tolerance -> Unstable
    pass
```

### 3.4. Elastic Calculation (`components/validator/elastic.py`)

Applies +/- strain in Voigt notation directions (xx, yy, zz, yz, xz, xy), measures stress, and fits linear regression to get Cij.

## 4. Implementation Approach

1.  **Implement Elastic Calc**: Write logic to apply strain and fit stress-strain curve using ASE.
2.  **Implement EOS Calc**: Write logic to scale volume and fit Birch-Murnaghan equation (`ase.eos`).
3.  **Implement Phonon Calc**:
    -   Check for `phonopy`.
    -   If missing, skip or use a very simple "force constant check" (sanity check).
4.  **Implement Reporter**: Use `jinja2` or simple string manipulation to create an HTML page with `matplotlib` plots (encoded as base64 images).
5.  **Mocking**:
    -   Mock Validator returns "Pass" with fake B0 and Cij values.
    -   Mock Validator returns "Fail" (Unstable Phonon) for testing.

## 5. Test Strategy

### 5.1. Unit Testing
-   **EOS Fit**: Provide known V, E data points (e.g., LJ argon) and verify B0 is correct.
-   **Elastic Fit**: Provide a known stress-strain response and verify Cij.

### 5.2. Integration Testing
-   **Full Workflow**: Run the loop until convergence, then trigger validation.
-   **Report Generation**: Verify `report.html` is created and contains the text "Validation Passed" or "Failed".
