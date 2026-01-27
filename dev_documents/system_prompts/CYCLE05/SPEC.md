# Cycle 05: Validation Framework

## 1. Summary

Cycle 05 implements the "Quality Assurance" layer of the system. While the active learning loop ensures the potential is consistent with the DFT data, it does not guarantee that the potential captures physical properties like phonon stability or elastic moduli correctly. The `Validator` module addresses this by running a battery of physical tests on the trained potential.

We will integrate `phonopy` for vibrational stability analysis and implement custom scripts for Elastic Constants ($C_{ij}$) and Equation of State (EOS) calculations. The Validator acts as a "Gatekeeper": only potentials that pass these physics-based tests are allowed to be promoted to "Production" status.

## 2. System Architecture

### File Structure

**mlip_autopipec/**
├── **validator/**
│   ├── **__init__.py**
│   ├── **runner.py**           # ValidationRunner class
│   ├── **phonons.py**          # Phonopy integration
│   ├── **elastic.py**          # Elastic constants logic
│   └── **eos.py**              # Birch-Murnaghan fitting
└── **config/schemas/**
    └── **validation.py**       # ValidationConfig schema

### Component Description

*   **`validator/runner.py`**: The main entry point. It runs the configured tests in sequence and aggregates the results into a `ValidationReport`.
*   **`validator/phonons.py`**: Interacts with `phonopy`. It generates displacements, calculates forces using the MLIP, computes the force constants, and solves for the phonon band structure. It checks for imaginary frequencies ($\omega^2 < 0$) which indicate dynamic instability.
*   **`validator/elastic.py`**: Deforms the unit cell (strain) and calculates the stress response. It fits the linear relationship $\sigma = C \epsilon$ to obtain the stiffness tensor and checks the Born stability criteria.
*   **`validator/eos.py`**: Performs isotropic compression/expansion and fits the Energy-Volume curve to the Birch-Murnaghan equation to extract the Bulk Modulus ($B_0$) and Equilibrium Volume ($V_0$).

## 3. Design Architecture

### Domain Models

**`ValidationConfig`**
*   **Role**: Defines which tests to run and their tolerances.
*   **Fields**:
    *   `phonons`: `bool`
    *   `elastic`: `bool`
    *   `eos`: `bool`
    *   `max_force_rmse`: `float` (default 0.05 eV/A)
    *   `phonon_supercell`: `List[int]` (e.g., [2, 2, 2])
    *   `elastic_tolerance`: `float` (e.g., 0.15 for 15% error vs DFT)

**`ValidationReport`**
*   **Role**: Summary of results.
*   **Fields**:
    *   `passed`: `bool`
    *   `tests`: `Dict[str, TestResult]`
    *   `metrics`: `Dict[str, float]` (RMSE, B0, C11, etc.)
    *   `plots`: `List[FilePath]` (Paths to generated PNGs)

### Key Invariants
1.  **Strictness**: A potential with imaginary phonons (excluding Gamma point acoustic modes) MUST fail validation.
2.  **Independence**: Validation tests should run independently; failure in one should not stop the others (so the user gets a full report).

## 4. Implementation Approach

1.  **Phonopy Integration**:
    *   Use `phonopy` Python API (`Phonopy` class).
    *   Implement `calculate_phonons(structure, potential)`.
    *   Generate supercells with `phonopy.generate_displacements`.
    *   Compute forces using `pace_calculator`.
    *   Compute bands and check for negative eigenvalues.

2.  **Elastic Constants**:
    *   Implement `calculate_elastic_constants(structure, potential)`.
    *   Apply small strains ($\pm 1\%, \pm 2\%$) in Voigt notation directions (xx, yy, zz, yz, xz, xy).
    *   Compute stress for each strained state.
    *   Solve linear system to get $C_{ij}$.

3.  **EOS**:
    *   Implement `calculate_eos(structure, potential)`.
    *   Scale volume from 0.8 to 1.2.
    *   Fit `ase.eos.EquationOfState`.

4.  **Reporting**:
    *   Generate a simple HTML or Markdown summary showing the plots and Pass/Fail status.

## 5. Test Strategy

### Unit Testing
*   **EOS**: Test the fitting logic with perfect synthetic data ($E = V^2$) to ensure the code recovers the parameters.
*   **Elastic**: Test the Born criteria logic (e.g., pass a stable cubic matrix and an unstable one).

### Integration Testing
*   **Phonon (Mocked)**: Mock the force calculator to return forces corresponding to a simple harmonic oscillator. Verify that `phonons.py` produces a valid band structure object.
*   **End-to-End Validator**: Run `mlip-auto validate` on a provided `potential.yace` and `structure.cif`. (This requires a working potential, perhaps the ZBL baseline itself can be tested).
