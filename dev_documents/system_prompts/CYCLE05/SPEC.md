# Cycle 05 Specification: kMC Integration & Advanced Validation

## 1. Summary

**Goal**: Extend the system's capabilities beyond short-timescale MD and ensure strict physical validation. This cycle integrates **Kinetic Monte Carlo (kMC)** via EON to explore slow processes (e.g., diffusion, ordering) and implements a rigorous **Validator** module. The Validator checks phonon stability and elastic constants, acting as a gatekeeper to prevent faulty potentials from being deployed.

**Key Deliverables**:
1.  **`EONWrapper` (Dynamics)**: An interface to the EON software (aKMC). It manages the setup of kMC simulations and the execution of the `eonclient`.
2.  **`Validator`**: A module that runs physical tests on the trained potential.
    *   **Phonon Stability**: Using `phonopy` to check for imaginary frequencies.
    *   **Elastic Constants**: Calculating bulk/shear moduli and checking Born stability criteria.
3.  **Gatekeeper Logic**: The Orchestrator uses validation results to decide whether to proceed or request more data.

## 2. System Architecture

Files in **bold** are the primary focus of this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **validation.py**     # ValidationResult, ValidationMetric
├── infrastructure/
│   ├── dynamics/
│   │   ├── **__init__.py**
│   │   └── **eon_kmc.py**    # EONWrapper (kMC)
│   └── validator/
│       ├── **__init__.py**
│       ├── **phonons.py**    # Phonopy wrapper
│       └── **elastic.py**    # Elastic constant calculator
└── utils/
    └── **phonopy_utils.py**  # Helper for Phonopy setup
```

## 3. Design Architecture

### 3.1 `EONWrapper` (Dynamics)

*   **Config**: `temperature` (float), `process_search_method` ("dimer", "neb"), `saddle_search_steps` (int).
*   **Logic**:
    1.  Create EON directory structure (`config.ini`, `reactant.con`).
    2.  Generate a Python driver (`pace_driver.py`) that EON calls to compute energy/forces using `potential.yace`.
    3.  **OTF Detection**: The driver script itself monitors $\gamma$. If high, it writes the structure and exits with a specific code (e.g., 100).
    4.  **Orchestrator Logic**: Detect exit code 100 -> Treat as halt -> Extract structure.

### 3.2 `Validator`

*   **Phonon Stability**:
    1.  Create a supercell (e.g., 2x2x2).
    2.  Calculate force constants (finite displacement).
    3.  Compute phonon band structure along high-symmetry paths.
    4.  **Fail Condition**: Imaginary frequencies (negative $\omega^2$) anywhere in the Brillouin zone (ignoring $\Gamma$-point acoustic modes).
*   **Elastic Constants**:
    1.  Apply small strains ($\pm 1\%$) to the unit cell.
    2.  Fit Energy vs Strain curves.
    3.  Calculate Stiffness Tensor ($C_{ij}$).
    4.  **Fail Condition**: Violation of Born stability criteria (e.g., $C_{11} - C_{12} > 0$).

## 4. Implementation Approach

1.  **Implement `EONWrapper`**: Focus on generating the `config.ini` and the driver script.
2.  **Implement `Validator`**: Use `phonopy` API if available, or CLI wrapper. Implement simple finite-difference elasticity calculation in pure Python/NumPy.
3.  **Update Orchestrator**: Call `validator.validate()` after training. If failed, log warning or halt (configurable).

## 5. Test Strategy

### 5.1 Unit Testing
*   **EON Config**: Verify `config.ini` generation.
*   **Elasticity**: Create a dummy potential (e.g., Lennard-Jones) and verify that the calculated elastic constants match analytical values.

### 5.2 Integration Testing
*   **"Mock EON"**: Simulate EON by calling the driver script manually with a few structures.
*   **Validation Pipeline**:
    *   Train a potential on a very small dataset (likely unstable).
    *   Run `Validator`.
    *   Verify it reports "FAILED" (Imaginary phonons).
    *   Train on a good dataset.
    *   Verify it reports "PASSED".
