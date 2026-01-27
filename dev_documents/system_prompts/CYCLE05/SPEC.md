# Cycle 05 Specification: Validation Framework

## 1. Summary
Cycle 05 introduces the **Validator**, the quality assurance module of PyAcemaker. Before any potential is deployed or marked as "Production Ready", it must pass a battery of physics-based tests. This cycle implements automated checks for **Phonon Dispersion** (dynamic stability), **Elastic Constants** (mechanical stability), and **Equation of State (EOS)** (thermodynamic behavior).

## 2. System Architecture

### 2.1. File Structure
```text
src/mlip_autopipec/
├── validation/                     # [CREATE]
│   ├── __init__.py
│   ├── runner.py                   # [CREATE] Validation Orchestrator
│   ├── phonon.py                   # [CREATE] Phonopy wrapper
│   ├── elastic.py                  # [CREATE] Elasticity calculator
│   └── eos.py                      # [CREATE] Birch-Murnaghan fit
└── config/
    └── validation_config.py        # [CREATE] Thresholds and criteria
```

### 2.2. Component Interaction
- **`ValidationRunner`**: Runs a sequence of tests defined in the config. Aggregates results into a `ValidationReport`.
- **`PhononCalculator`**: Interfaces with `phonopy`. It calculates force constants using the MLIP (fast) and checks for imaginary frequencies in the band structure.
- **`ElasticCalculator`**: Deforms the unit cell ($\pm \delta$), calculates stress, and fits the stiffness tensor $C_{ij}$.
- **`EOSCalculator`**: Compresses/expands the cell and fits the Energy-Volume curve to extract Bulk Modulus ($B_0$).

## 3. Design Architecture

### 3.1. Stability Criteria (The Gatekeeper)
The Validator enforces strict physical rules:
1.  **Dynamic Stability**: No imaginary phonon modes ($\omega^2 < -\epsilon$) across the Brillouin zone.
2.  **Mechanical Stability (Born Criteria)**:
    -   For cubic: $C_{11} - C_{12} > 0$, $C_{11} + 2C_{12} > 0$, $C_{44} > 0$.
3.  **Accuracy**:
    -   $RMSE_{E} < 2$ meV/atom (vs Test Set).
    -   $RMSE_{F} < 0.05$ eV/Å.

### 3.2. Reporting
- Output: A JSON summary (`report.json`) and ideally an HTML report with plots (Phonon bands, Parity plots).

## 4. Implementation Approach

1.  **Phonon**: Use `phonopy` API.
    -   Generate displacements.
    -   Calculate forces using `PacemakerCalculator`.
    -   Compute bands. Check for `frequency < 0`.
2.  **Elastic**: Implement a simple finite-difference method.
    -   Apply strains (Voigt notation).
    -   Measure stress.
    -   Solve linear equations to get $C_{ij}$.
3.  **Runner**: Implement `validate(potential_path)`. Returns `Passed` / `Failed` / `Conditional`.

## 5. Test Strategy

### 5.1. Unit Testing
- **Elastic**: Test the logic on a known Lennard-Jones potential where $C_{ij}$ are known analytically or from literature.
- **Phonon**: Run on a stable crystal (should pass) and an unstable one (e.g., imaginary mode, should fail).

### 5.2. Integration Testing
- **Full Validation Run**:
    -   Input: A dummy potential file.
    -   Action: Run validation runner.
    -   Output: Verify `report.json` is generated and contains "phonon", "elastic", "eos" keys.
