# Cycle 06 Specification: Validation & Production Readiness

## 1. Summary

The final cycle focuses on **Quality Assurance** and **User Experience**. A potential that produces low RMSE on a test set is not necessarily useful for physics; it must reproduce fundamental material properties like phonon spectra, elastic constants, and melting points.

We will implement the **Validator** module, a comprehensive test suite that runs after every training cycle. It determines if a potential is "Production Ready". This module interfaces with **Phonopy** for lattice dynamics and performs internal stress-strain calculations for elasticity.

We also finalize the **User Interface**, ensuring that the logs are readable, error messages are helpful, and the generated output (the "Final Product") is packaged correctly with versioning and metadata. The system is polished to handle the "Fe/Pt on MgO" tutorial scenario seamlessly.

## 2. System Architecture

Files to be created/modified are marked in **bold**.

```
PYACEMAKER/
├── src/
│   └── mlip_autopipec/
│       ├── **validator/**
│       │   ├── **__init__.py**
│       │   ├── **elastic.py**              # Elastic constant calc
│       │   ├── **phonon.py**               # Phonopy wrapper
│       │   ├── **eos.py**                  # Equation of State
│       │   └── **report_generator.py**     # HTML/Markdown reporting
│       ├── main.py                         # Final Polish
│       └── utils/
│           └── **versioning.py**           # Potential versioning
└── tests/
    ├── **integration/**
    │   └── **test_validation_suite.py**
    └── **e2e/**
        └── **test_tutorial_fept.py**       # The Grand Finale
```

## 3. Design Architecture

### 3.1. Validator Suite
The Validator runs a series of `Tests`.
*   **Elasticity**: Deforms the unit cell by $\pm 1\%$ in 6 Voigt channels, fits the Energy-Strain curve to a parabola, and extracts $C_{ij}$. Checks Born stability criteria.
*   **Phonons**: Generates supercells with displacements (frozen phonon method), calculates forces using the Potential, feeds them to Phonopy, and checks for imaginary frequencies ($\omega^2 < 0$) which indicate instability.
*   **EOS**: Calculates Energy vs Volume curve. Fits Birch-Murnaghan equation to get $V_0, B_0, B_0'$.

### 3.2. Reporting
The `ValidationResult` object from Cycle 01 is expanded.
*   It now holds plots (base64 encoded images or paths) and numerical tables.
*   `report_generator.py` compiles these into a `validation_report.html` for easy human consumption.

## 4. Implementation Approach

1.  **Dependencies**: Add `phonopy` and `matplotlib` to `pyproject.toml`.
2.  **Elastic Module**: Implement the strain-energy method. Verify against known values for Si (calculated with DFT or available in literature).
3.  **Phonon Module**: Implement the wrapper around `phonopy`.
    *   **Crucial**: Phonopy is complex. We must automate the supercell generation (e.g., $2 \times 2 \times 2$) and path selection (Band structure).
4.  **Final Polish**:
    *   Review all log messages.
    *   Ensure `config.yaml` has a "production" section.
    *   Add a `--version` flag to the CLI.

## 5. Test Strategy

### 5.1. Unit Testing Approach
*   **Elasticity**: Create a "Harmonic Potential" mock that returns energy $E = \frac{1}{2} k x^2$. Verify that the Elastic module correctly extracts the spring constant $k$.
*   **Phonons**: Verify that the Phonopy interface correctly constructs the `Phonopy` object from an ASE atoms object.

### 5.2. Integration Testing Approach
*   **Full Validation Run**: Train a quick potential for Si. Run the Validator.
    *   Check that `elastic.json` and `phonon_band.pdf` are generated.
    *   Check that the stability check passes (or fails if the potential is bad).
*   **Tutorial Verification**: The ultimate test. Execute the code paths required for the `Fe/Pt on MgO` scenario (Notebook 01 and 02) programmatically and ensure no exceptions are raised.
