# Cycle 08 Specification: Validation & Production

## 1. Summary

Cycle 08 is the final stage of the development process, focusing on **Validation** and **Quality Assurance**. A machine learning potential may have low errors on the training set (RMSE) but still produce unphysical results (e.g., imaginary phonon modes, negative bulk modulus). This cycle implements a comprehensive suite of physics-based tests to act as a "Gatekeeper" before a potential is deployed.

Key validations include:
1.  **Phonon Stability**: Calculating phonon dispersion curves using **Phonopy** to check for imaginary frequencies (dynamic instability).
2.  **Elastic Constants**: Computing the stiffness tensor ($C_{ij}$) to ensure mechanical stability (Born criteria) and agreement with DFT/Experimental values.
3.  **Equation of State (EOS)**: Fitting energy-volume curves to the Birch-Murnaghan equation to extract equilibrium volume ($V_0$) and bulk modulus ($B_0$).

Additionally, this cycle implements the **Report Generator**, which compiles all metrics and plots into a user-friendly HTML dashboard.

## 2. System Architecture

Files in **bold** are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **config.py**         # Update ValidatorConfig
├── validator/
│   ├── **__init__.py**
│   ├── **interface.py**      # Enhanced BaseValidator
│   ├── **phonon.py**         # Phonopy Interface
│   ├── **elastic.py**        # Elastic Constants Calculator
│   ├── **eos.py**            # EOS Fitting Logic
│   └── **report.py**         # HTML Report Generator
└── tests/
    └── unit/
        └── **test_validator.py**
```

## 3. Design Architecture

### 3.1 Phonon Analyzer (`validator/phonon.py`)
This class interfaces with `phonopy`.
*   **Method**: Finite displacement method.
*   **Input**: `Potential`, `UnitCell`, `SupercellMatrix`.
*   **Output**: `PhononBandStructure` (plot), `MaxImaginaryFrequency` (metric).

### 3.2 Elastic Analyzer (`validator/elastic.py`)
This class calculates elastic constants.
*   **Method**: Apply small strains ($\pm \delta$), compute stress, fit to linear elasticity.
*   **Output**: $C_{11}, C_{12}, C_{44}$ (for cubic), Bulk Modulus ($B$), Shear Modulus ($G$).

### 3.3 Report Generator (`validator/report.py`)
This class aggregates results.
*   **Input**: `ValidationResults` from all tests.
*   **Output**: `report.html` using `jinja2` or `pandas`.

## 4. Implementation Approach

1.  **Enhance Domain Models**: Add `ValidatorConfig` (phonon supercell size, strain magnitude).
2.  **Implement EOS**: Write `fit_birch_murnaghan(volumes, energies)`.
3.  **Implement Elastic**: Write `calculate_elastic_constants(potential, structure)`.
4.  **Implement Phonon**: Use `phonopy` API (if available) or `ase.phonons`.
5.  **Implement Report**: Generate a summary HTML.
6.  **Integration**: The `Orchestrator` calls `validate()` at the end of each cycle. If validation fails, it may trigger more exploration or simply mark the potential as "Beta".

## 5. Test Strategy

### 5.1 Unit Testing
*   **EOS Fit**: Feed known V-E data points (e.g., for Cu) and verify $B_0$ is correct.
*   **Elastic Matrix**: Feed a known stress-strain response and verify $C_{ij}$.

### 5.2 Integration Testing
*   **Full Validation**: Run the validator on a dummy potential (e.g., LJ for Argon). Verify that phonon curves are real and positive, and the HTML report is generated.
