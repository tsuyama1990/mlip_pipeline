# Cycle 06 Specification: Validation Suite

## 1. Summary

Cycle 06 implements the **Validation Suite**, the "Quality Assurance" gate of the system. Before any potential is deployed to the production environment (or even to the next cycle of active learning), it must prove its physical validity.

We implement three critical physical tests:
1.  **Phonon Stability**: Using `phonopy` to calculate the phonon dispersion curves. We check for imaginary frequencies ($\omega^2 < 0$), which indicate dynamic instability.
2.  **Elastic Constants**: Calculating the stiffness matrix ($C_{ij}$) via finite differences and verifying the Born stability criteria (e.g., $C_{11} - C_{12} > 0$).
3.  **Equation of State (EOS)**: Fitting the Energy-Volume curve to the Birch-Murnaghan equation to ensure a correct Bulk Modulus ($B_0$) and equilibrium volume ($V_0$).

This cycle also includes the generation of a comprehensive HTML report summarizing these metrics.

## 2. System Architecture

We populate the `validation/` directory.

### 2.1 File Structure

```ascii
src/mlip_autopipec/
├── config/
│   └── schemas/
│       └── **validation.py**       # Thresholds (max_rmse, stability)
├── **validation/**
│   ├── **__init__.py**
│   ├── **runner.py**               # Orchestrates all tests
│   ├── **phonons.py**              # Phonopy wrapper
│   ├── **elasticity.py**           # Elastic constant calculator
│   ├── **eos.py**                  # EOS calculator
│   └── **report.py**               # HTML Report Generator
└── orchestration/
    └── phases/
        └── **validation.py**       # Validation Phase Logic
```

## 3. Design Architecture

### 3.1 Validation Runner (`validation/runner.py`)

*   **Responsibility**: Run a sequence of tests and aggregate results.
*   **Input**: `potential_path`, `test_structure` (equilibrium).
*   **Output**: `ValidationResult` (Pass/Fail/Conditional, details dict).

### 3.2 Phonon Validator (`validation/phonons.py`)

*   **Logic**:
    *   Create a supercell (e.g., $4 \times 4 \times 4$).
    *   Calculate forces using the ACE potential (using `ase` calculator).
    *   Feed forces to `phonopy`.
    *   Iterate over the band structure data.
    *   **Fail Condition**: If any frequency $\omega < -0.1$ THz (allowing small numerical noise).

### 3.3 Elasticity Validator (`validation/elasticity.py`)

*   **Logic**:
    *   Apply strain tensors ($\pm \delta$) to the unit cell.
    *   Calculate stress.
    *   Fit Stress-Strain curve to get $C_{ij}$.
    *   Check Born criteria based on crystal symmetry (cubic, hexagonal, etc.).

### 3.4 EOS Validator (`validation/eos.py`)

*   **Logic**:
    *   Expand/Compress volume by $\pm 10\%$.
    *   Calculate Energy.
    *   Fit using `ase.eos`.
    *   **Fail Condition**: If $B_0 < 0$ or fitting error is large.

## 4. Implementation Approach

1.  **Step 1: Implement EOS Validator.**
    *   This is the simplest. Use `ase.eos.EquationOfState`.

2.  **Step 2: Implement Elasticity Validator.**
    *   Use `ase.calculators.calculator.Calculator` attached to atoms.
    *   Implement manual deformation logic or use `ase.elasticity` if suitable (manual is often more robust for specific criteria).

3.  **Step 3: Implement Phonon Validator.**
    *   Requires `phonopy` library.
    *   Implement the interface: Atoms -> Supercell -> Forces -> Phonopy Object -> Band Structure.

4.  **Step 4: Report Generation.**
    *   Use `jinja2` or simple string formatting to create an HTML page with plots (using `matplotlib` to save PNGs).

## 5. Test Strategy

### 5.1 Unit Testing
*   **Born Criteria:**
    *   Unit test the logic that checks $C_{ij}$ matrices. Pass a known unstable matrix and assert it returns False.
*   **EOS Fitting:**
    *   Pass a set of (V, E) points. Assert $B_0$ is calculated correctly.

### 5.2 Integration Testing
*   **Phonopy Integration:**
    *   **Mocking**: We might not want to run full phonopy in CI. We can mock the `phonopy` object to return a fake band structure with/without imaginary modes.
    *   **Real Run (if allowed)**: Run on a very simple system (Aluminium) with a Lennard-Jones potential (which we can set up easily).
