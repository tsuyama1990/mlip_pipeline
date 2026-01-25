# Cycle 06: Validation Suite

## 1. Summary
This cycle implements the **Validator** module, the Quality Assurance gatekeeper of the system. Before any potential is deployed or marked as a "Generation", it must pass a series of physical tests. These tests go beyond simple RMSE metrics and verify that the potential reproduces fundamental physical properties like lattice stability (Phonons), mechanical stiffness (Elasticity), and thermodynamic behavior (EOS).

## 2. System Architecture

We add the `phases/validation` module.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── orchestration/
│   └── phases/
│       └── **validation/**
│           ├── **__init__.py**
│           ├── **manager.py**         # ValidationPhase implementation
│           ├── **phonons.py**         # Phonopy Interface
│           ├── **elastic.py**         # Elastic Tensor Calculation
│           ├── **eos.py**             # EOS (Birch-Murnaghan)
│           └── **reporter.py**        # Validation Report Generator
└── tests/
    └── **test_validation.py**
```

## 3. Design Architecture

### Phonon Stability (`phonons.py`)
*   **Dependency**: `phonopy` (must be installed in the environment).
*   **Method**: Finite displacement method.
*   **Criterion**: No imaginary frequencies ($\omega^2 < 0$) in the phonon band structure, except for acoustic modes at $\Gamma$ point (translation).

### Elasticity Check (`elastic.py`)
*   **Method**: Apply small strains ($\pm \delta$) to the unit cell and measure stress response. Fit to obtain $C_{ij}$.
*   **Criterion**: Check Born stability criteria (e.g., $C_{11} - C_{12} > 0$ for cubic).
*   **Comparison**: Compare Bulk Modulus ($B$) and Shear Modulus ($G$) against reference DFT values.

### Equation of State (`eos.py`)
*   **Method**: Calculate energy at volumes $0.8 V_0$ to $1.2 V_0$.
*   **Fit**: Birch-Murnaghan EOS.
*   **Criterion**: Curve must be convex ($B > 0$) and smooth.

## 4. Implementation Approach

1.  **Phonopy Wrapper**: Implement a class that takes a `potential.yace` and a structure, runs Phonopy, and returns a boolean `is_stable` and the band plot data.
2.  **Elastic Calculator**: Implement a routine using `ase.eos` or custom strain-stress logic.
3.  **Phase Manager**: `ValidationPhase.run(potential_path)` runs all sub-validators in parallel/sequence.
4.  **Reporting**: Aggregate results into a dictionary (Pass/Fail, Values, Plots) and generate a summary.

## 5. Test Strategy

### Unit Testing
*   **`test_elastic.py`**: Calculate elastic constants for a known Lennard-Jones potential (using ASE's LJ calculator) and verify results match theory.
*   **`test_phonons.py`**: Mock the Phonopy return object to simulate stable and unstable cases.

### Integration Testing
*   **End-to-End Validation**: Run the full validator on a dummy potential (or LJ). Verify it generates a report and correctly flags "PASS".
