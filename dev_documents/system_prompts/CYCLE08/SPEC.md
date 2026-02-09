# Cycle 08 Specification: Validator (The Gatekeeper)

## 1. Summary

In this final cycle, we implement the **Validator** module, the component responsible for the rigorous physical verification of the generated MLIP. It acts as the "Quality Assurance" gatekeeper. Before a potential is marked as "Production Ready," it must pass a suite of standard materials science tests.

These tests include:
1.  **Phonon Dispersion**: Calculating vibrational modes across the Brillouin Zone to ensure dynamical stability (no imaginary frequencies).
2.  **Elastic Constants**: Computing the stiffness tensor ($C_{ij}$) to verify mechanical stability (Born criteria) and accuracy against DFT/Experiment.
3.  **Equation of State (EOS)**: Fitting energy-volume curves to the Birch-Murnaghan equation to extract bulk modulus and equilibrium volume.
4.  **Reporting**: Aggregating all results into a human-readable HTML report with interactive plots.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the specific deliverables for this cycle.

```ascii
src/mlip_pipeline/
├── components/
│   ├── validators/
│   │   ├── **__init__.py**
│   │   ├── **phonons.py**      # Phonopy Wrapper
│   │   ├── **elastic.py**      # Elastic Constants Calculator
│   │   ├── **eos.py**          # EOS Calculator
│   │   └── **report.py**       # HTML Report Generator
│   └── base.py                 # (Modified) Enhance BaseValidator interface
└── domain_models/
    └── **results.py**          # (Enhanced) ValidationResult schema
```

## 3. Design Architecture

### 3.1. Validator Interface
The `BaseValidator` in `src/mlip_pipeline/components/base.py` will be refined.

*   `validate(self, potential: Potential, test_structures: Dict[str, Structure]) -> ValidationReport`
    *   Input: The potential and a dictionary of reference structures (e.g., ground state, strained).
    *   Output: `ValidationReport` object containing pass/fail status and detailed metrics.

### 3.2. Phonon Calculator
Located in `src/mlip_pipeline/components/validators/phonons.py`.

*   **Config**: `StandardValidatorConfig`.
    *   `phonon_displacement`: Float (e.g., 0.01 Å).
    *   `supercell_matrix`: List[int].
*   **Logic**:
    1.  Use **Phonopy** via its Python API.
    2.  Generate displaced supercells.
    3.  Compute forces using the Potential (via ASE calculator).
    4.  Calculate force constants and phonon bands.
    5.  Check for imaginary frequencies ($\omega^2 < -\epsilon$).

### 3.3. Elastic & EOS Calculator
Located in `elastic.py` and `eos.py`.
*   **Elastic**: Apply strains ($\pm \delta$), compute stress, fit linear relationship $\sigma = C \epsilon$. Check Born stability criteria for the specific crystal class.
*   **EOS**: Scale volume ($\pm 10\%$), compute energy, fit Birch-Murnaghan. Return $V_0, B_0, B'_0$.

### 3.4. HTML Report Generator
Located in `src/mlip_pipeline/components/validators/report.py`.
*   **Input**: `ValidationReport` object (Metrics, Plots as base64/paths).
*   **Output**: `validation_report.html`.
*   **Tech**: Use `jinja2` templating and `matplotlib`/`plotly` for visualization.

## 4. Implementation Approach

1.  **Metric Calculators**: Implement independent classes for `PhononCalc`, `ElasticCalc`, `EOSCalc`. Each should take an ASE atoms object and a potential calculator.
2.  **Report Template**: Create a simple HTML template with placeholders for tables (metrics) and images (plots).
3.  **Integration**: The `StandardValidator` class orchestrates the execution of these sub-calculators and compiles the final report.

## 5. Test Strategy

### 5.1. Unit Testing
*   **EOS Fit**: Provide synthetic E-V data. Assert `EOSCalc` returns correct $B_0$.
*   **Elastic Check**: Provide a known stiffness matrix (e.g., unstable). Assert `check_stability` returns False.

### 5.2. Integration Testing
*   **Full Validation Run**: Run the validator on a dummy potential (LJ).
    *   Assert `phonons` are calculated (no crash).
    *   Assert `elastic` constants are reasonable (positive).
    *   Assert `report.html` is generated.

### 5.3. Requirements
*   `phonopy` must be installed.
*   `matplotlib` and `pandas` are required for reporting.
