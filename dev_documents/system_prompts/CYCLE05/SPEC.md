# Cycle 05: Validation Framework

## 1. Summary

At this stage, we can train a potential (Cycle 04) and run it (Cycle 02). However, "it runs" does not mean "it is correct". A potential can have very low RMSE on training data but fail catastrophically when predicting properties like thermal expansion or vibrational modes. Cycle 05 introduces the **Validation Framework**, acting as the Quality Assurance (QA) department of our autonomous research facility.

This module is responsible for subjecting the trained `.yace` potential to a battery of physical tests. It does not just look at errors; it looks at physics.
1.  **Phonon Dispersion**: We calculate the vibrational frequencies of the crystal lattice. If any frequencies are imaginary ($\omega^2 < 0$), it means the crystal structure is dynamically unstable—the potential predicts that the atoms would spontaneously drift away from their equilibrium positions.
2.  **Elastic Constants**: We apply small strains to the crystal and measure the stress response. This yields the elastic tensor ($C_{ij}$) and bulk moduli. We check if these satisfy the Born stability criteria.
3.  **Equation of State (EOS)**: We compress and expand the crystal to check if the energy-volume curve is smooth and convex.

Finally, this module must generate a **Human-Readable Report**. Since the system runs autonomously, the user needs a dashboard (HTML) to quickly assess the health of the generation.

## 2. System Architecture

We introduce the `physics/validation` package.

### File Structure
Files to be created/modified are in **bold**.

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **validation.py**       # Validation result schemas
│       │   └── config.py               # Update with ValidationConfig
│       ├── physics/
│       │   ├── validation/
│       │   │   ├── **__init__.py**
│       │   │   ├── **runner.py**       # Main validation driver
│       │   │   ├── **phonon.py**       # Phonopy wrapper
│       │   │   ├── **elasticity.py**   # Elastic constant calculator
│       │   │   └── **eos.py**          # Birch-Murnaghan fit
│       │   └── reporting/
│       │       ├── **__init__.py**
│       │       └── **html_gen.py**     # Jinja2 report generator
└── tests/
    └── physics/
        └── validation/
            └── **test_phonon.py**
```

### Component Interaction

1.  **Orchestrator** calls `ValidationRunner.validate(potential_path)`.
2.  **`ValidationRunner`** runs sub-validators in parallel or sequence.
3.  **`PhononValidator`**:
    -   Builds a supercell.
    -   Runs finite displacement calculations using the potential.
    -   Uses `phonopy` to compute the band structure.
    -   Checks for imaginary frequencies.
4.  **`ElasticityValidator`**:
    -   Deforms the unit cell ($\pm 1\%$).
    -   Calculates stress.
    -   Fits the stiffness matrix.
5.  **`ReportGenerator`**:
    -   Aggregates results into `ValidationReport`.
    -   Renders `report.html`.

## 3. Design Architecture

### 3.1. Validation Domain Model (`domain_models/validation.py`)

-   **Class `ValidationMetric`**:
    -   `name`: `str` (e.g., "Bulk Modulus").
    -   `value`: `float`.
    -   `reference`: `Optional[float]` (Experimental/DFT value).
    -   `error`: `Optional[float]`.
    -   `passed`: `bool`.

-   **Class `ValidationResult`**:
    -   `potential_id`: `str`.
    -   `metrics`: `List[ValidationMetric]`.
    -   `plots`: `Dict[str, Path]` (Paths to PNG images).
    -   `overall_status`: `Literal["PASS", "WARN", "FAIL"]`.

### 3.2. Phonon Validator (`physics/validation/phonon.py`)
-   **Dependency**: `phonopy`.
-   **Method**: Finite displacement method (Frozen Phonon).
-   **Logic**:
    -   If `min(frequencies) < -0.1` THz (tolerance), mark as FAIL.
    -   We must distinguish between "soft modes" (phase transitions) and "instability" (bad potential). For now, strict stability is the goal.

## 4. Implementation Approach

### Step 1: Validator Interface
-   Define an abstract base class `BaseValidator` in `runner.py` with a `validate()` method.
-   This ensures all validators return a consistent `ValidationResult`.

### Step 2: EOS Validator (Simplest)
-   Implement `eos.py`.
-   Use `ase.eos.EquationOfState`.
-   Generate 10 structures with volumes from $0.9 V_0$ to $1.1 V_0$.
-   Fit Birch-Murnaghan.
-   Check if Bulk Modulus > 0.

### Step 3: Elasticity Validator
-   Implement `elasticity.py`.
-   Apply 6 strain patterns (Voigt notation).
-   Calculate stress.
-   Solve linear system to get $C_{ij}$.
-   Check Born stability conditions (e.g., $C_{11} - C_{12} > 0$).

### Step 4: Phonon Validator (Complex)
-   Implement `phonon.py`.
-   **Integration**:
    -   Instantiate `Phonopy` object.
    -   Generate displacements.
    -   For each displacement, calculate forces using the ACE potential (fast).
    -   Set forces back to `Phonopy`.
    -   Calculate band structure and density of states (DOS).
    -   Plot using `matplotlib` and save to disk.

### Step 5: Reporting
-   Implement `html_gen.py`.
-   Use `jinja2`. Create a template `templates/report.html`.
-   Embed images as base64 or link to files.

## 5. Test Strategy

### 5.1. Unit Testing
-   **EOS**:
    -   Pass a perfect EOS curve (parabola). Assert it returns correct $B_0$.
    -   Pass a jagged curve. Assert it fails or warns.

### 5.2. Integration Testing (Mocked)
-   **Mocking Phonopy**:
    -   `phonopy` is a heavy dependency. Mock the `Phonopy` class.
    -   Mock `get_band_structure_dict` to return a dictionary with known frequencies.
    -   **Case 1**: Return all positive frequencies. Assert `passed=True`.
    -   **Case 2**: Return some imaginary frequencies (represented as negative numbers or complex). Assert `passed=False`.

### 5.3. Visual Verification (UAT)
-   Generate a report. Open it in a browser. Check if the plots are visible and the tables are formatted correctly.
