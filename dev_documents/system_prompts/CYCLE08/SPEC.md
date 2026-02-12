# Cycle 08 Specification: Validation & Reporting

## 1. Summary
**Goal**: Implement the `Validator` component to ensure the physical correctness of the trained potentials. This cycle acts as the "Quality Gate" before deployment, performing rigorous tests (Phonons, Elastic Constants, EOS) and generating a comprehensive HTML report.

**Key Features**:
*   **Phonon Stability**: Check for imaginary frequencies (dynamic instability) using `phonopy`.
*   **Elastic Constants**: Calculate $C_{ij}$ and verify Born stability criteria.
*   **Equation of State (EOS)**: Fit Birch-Murnaghan curves.
*   **Reporting**: Generate `report.html` with interactive plots (RMSE, CDF, Uncertainty).

## 2. System Architecture

Files to be implemented/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── **validation.py**       # Validation Config & Results
│   └── ...
├── validator/
│   ├── **__init__.py**
│   ├── **base.py**             # Abstract Base Class
│   ├── **phonon.py**           # Phonopy Wrapper
│   ├── **elastic.py**          # Elasticity Calculator
│   └── **eos.py**              # EOS Calculator
└── reporting/
    ├── **__init__.py**
    └── **html_report.py**      # Report Generator
├── tests/
    └── **test_validator/**
        ├── **test_phonon.py**
        └── **test_elastic.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`src/mlip_autopipec/domain_models/validation.py`)

*   **`ValidationResult`**:
    *   `passed`: bool.
    *   `warnings`: List[str].
    *   `phonon_stability`: bool (True if no imaginary modes).
    *   `elastic_stability`: bool (True if Born criteria met).
    *   `elastic_constants`: dict ($C_{11}, C_{12}, C_{44}$).
    *   `eos_params`: dict ($V_0, E_0, B_0$).

### 3.2. Validator Component (`src/mlip_autopipec/validator/`)

#### `phonon.py`
*   **`PhononValidator`**:
    *   Uses `phonopy` (if available).
    *   **Logic**:
        1.  Create supercell with finite displacement.
        2.  Calculate forces using `potential.yace`.
        3.  Compute phonon band structure.
        4.  If min(frequency) < -0.1 THz (ignoring $\Gamma$-point acoustic modes), mark unstable.

#### `elastic.py`
*   **`ElasticValidator`**:
    *   Apply strain ($\pm 1\%$) to unit cell.
    *   Compute stress tensor.
    *   Fit linear regression $\sigma = C \epsilon$.
    *   Check Born stability conditions (e.g., cubic: $C_{11}-C_{12}>0, C_{44}>0$).

#### `eos.py`
*   **`EOSValidator`**:
    *   Scan volume $\pm 10\%$.
    *   Fit Birch-Murnaghan EOS.
    *   Return $B_0$ (Bulk Modulus).

#### `reporting/html_report.py`
*   **`ReportGenerator`**:
    *   Collects `TrainingMetrics`, `ValidationResult`, and Plots.
    *   Generates `report.html` using a Jinja2 template.
    *   **Plots**:
        *   RMSE Energy/Force vs Epoch.
        *   Cumulative Uncertainty Distribution (CDF).
        *   Phonon Band Structure (image).
        *   EOS Curve (image).

## 4. Implementation Approach

1.  **Implement Validator Interfaces**: Create `base.py`.
2.  **Implement Elastic/EOS Logic**: Simple Python calculations using `ase`.
3.  **Implement Phonon Logic**: Optional dependency handling (`try-import phonopy`).
4.  **Implement Report Generator**: Use `pandas` and `matplotlib` to create static HTML.
5.  **Mock Validation**: If deps missing, return dummy "Pass" results.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_elastic.py`**:
    *   Use a Lennard-Jones potential (known elastic constants).
    *   Verify calculated $C_{ij}$ matches analytical values within tolerance.
*   **`test_phonon.py`**:
    *   Mock force constants.
    *   Verify stability check logic correctly identifies imaginary modes.

### 5.2. Integration Testing
*   **Full Report**:
    *   Run `Validator.validate()`.
    *   Run `ReportGenerator.generate()`.
    *   Assert `report.html` exists and contains critical sections.
