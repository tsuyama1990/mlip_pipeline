# Cycle 05 Specification: Validation Framework

## 1. Summary

Cycle 05 introduces the "Validator", a critical quality assurance module. Before any potential is accepted as "Production Ready" or used for the next cycle of exploration, it must pass a battery of physical tests. These tests go beyond simple error metrics (RMSE) and verify the physical behavior of the potential: dynamical stability (phonons), mechanical stability (elastic constants), and thermodynamic behavior (Equation of State). This cycle creates the "Gatekeeper" logic that automatically rejects unphysical potentials.

## 2. System Architecture

Files to be added/modified (in bold):

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       └── **validation.py**  # Validation thresholds (e.g., max error %)
├── **validation/**
│   ├── **__init__.py**
│   ├── **phonons.py**         # Phonopy wrapper
│   ├── **elastic.py**         # Elastic constant calculator
│   ├── **eos.py**             # Birch-Murnaghan fitter
│   └── **report.py**          # HTML report generator
└── orchestration/
    └── ...
```

## 3. Design Architecture

### 3.1 Validation Config

**`ValidationConfig` (in `schemas/validation.py`)**
-   **Fields**:
    -   `phonon_check`: bool
    -   `elastic_check`: bool
    -   `eos_check`: bool
    -   `elastic_tolerance`: float (e.g., 0.15 for 15% error vs DFT)
    -   `reference_data`: Dict (Optional experimental/DFT values for comparison)

### 3.2 Phonon Check

**`PhononCheck` (in `validation/phonons.py`)**
-   **Responsibilities**:
    -   Calculate force constants using finite displacement (via Phonopy or internal logic).
    -   Calculate phonon band structure.
    -   **Criteria**: Pass if no imaginary frequencies (negative $\omega^2$) exist across the Brillouin zone (ignoring acoustic modes at $\Gamma$).

### 3.3 Elastic Check

**`ElasticCheck` (in `validation/elastic.py`)**
-   **Responsibilities**:
    -   Apply small strains to the unit cell.
    -   Calculate stress tensor.
    -   Fit to obtain Elastic Matrix ($C_{ij}$).
    -   **Criteria**: Check Born Stability Criteria (e.g., $C_{11}-C_{12} > 0$). Compare Bulk Modulus ($B$) with reference.

### 3.4 EOS Check

**`EOSCheck` (in `validation/eos.py`)**
-   **Responsibilities**:
    -   Calculate Energy vs Volume curve.
    -   Fit Birch-Murnaghan equation.
    -   **Criteria**: Curve must be convex (positive curvature). $V_0$ should be close to equilibrium.

## 4. Implementation Approach

1.  **Phonopy Integration**:
    -   Use `phonopy` library if available.
    -   Generate supercells with `phonopy.structure.atoms.PhonopyAtoms` (converted from ASE).
    -   Calculate forces using the ACE potential.
    -   Feed forces back to Phonopy to get eigenvalues.

2.  **Elasticity**:
    -   Use `ase.constraints.StrainFilter` or custom implementation.
    -   Apply $\pm \delta$ strain for 6 Voigt components.

3.  **Report Generation**:
    -   Use `jinja2` and `matplotlib`/`plotly`.
    -   Create a standalone HTML file containing:
        -   Parity Plots (Energy/Force).
        -   Phonon Band Structure Plot.
        -   EOS Plot.
        -   Table of elastic constants.
        -   Final "PASS/FAIL" stamp.

## 5. Test Strategy

### 5.1 Unit Testing
-   **EOS Fit**: Provide exact data from an EOS equation. Verify the fitter recovers the parameters.
-   **Stability Logic**: Provide a $C_{ij}$ matrix that violates Born criteria. Verify `ElasticCheck` returns `False`.

### 5.2 Integration Testing
-   **Validation Pipeline**:
    -   Train a "Good" potential (e.g., on perfect crystal data).
    -   Train a "Bad" potential (e.g., on random noise).
    -   Run Validator on both.
    -   Assert "Good" potential passes.
    -   Assert "Bad" potential fails (likely due to imaginary phonons or bad EOS).
