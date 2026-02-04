# Cycle 06 Specification: Integration, kMC & Validation

## 1. Summary

Cycle 06 is the final piece of the puzzle. We implement the **Validator** to ensure the generated potentials are not just numerically accurate but physically sound. We also extend the Dynamics Engine to support **Adaptive Kinetic Monte Carlo (aKMC)** via EON, enabling the system to simulate long-timescale phenomena like diffusion and ordering.

The `Validator` acts as a quality gate. Before a potential is marked as "Ready", it must pass a suite of physical tests: Phonon dispersion stability (no imaginary modes), Elastic constants check (Born stability criteria), and Equation of State (EOS) smoothness.

Finally, we glue everything together in the `Orchestrator` to support the full complex workflow required for the "Fe/Pt on MgO" Grand Challenge: Training -> MD -> kMC -> Retraining.

## 2. System Architecture

```ascii
src/mlip_autopipec/
├── ...
├── validation/
│   ├── __init__.py
│   ├── **validator.py**    # Main validation logic
│   ├── **phonons.py**      # Phonopy wrapper
│   └── **elastic.py**      # Elastic constants calc
├── dynamics/
│   └── **eon_driver.py**   # EON (kMC) wrapper
└── ...
```

## 3. Design Architecture

### 3.1. Physical Validation (`validation/`)
*   **`PhononCalculator`**: Wraps `phonopy`. Calculates force constants and band structure. Checks for imaginary frequencies ($\omega^2 < 0$) which indicate dynamic instability.
*   **`ElasticCalculator`**: Applies small strains to the unit cell, fits the Energy-Strain curve, and computes the stiffness tensor $C_{ij}$. Checks positive definiteness.
*   **`Validator`**: Orchestrates these tests and generates a `ValidationReport`.

### 3.2. EON Interface (`eon_driver.py`)
*   **`EONWrapper`**:
    *   Prepares `config.ini` for EON.
    *   Implements the Python driver script that EON calls to get Energy/Forces (bridging EON -> Pacemaker).
    *   Monitors EON execution for "high uncertainty" events (similar to MD Watchdog).

## 4. Implementation Approach

1.  **Develop Validator**: Implement the logic to calculate elastic constants using ASE's `ElasticModel` or custom strain implementation.
2.  **Integrate Phonopy**: Implement the interface to Phonopy. **Constraint**: If Phonopy is not installed, the test must skip gracefully.
3.  **EON Driver**: Create the `pace_driver.py` script that EON will use. This script must load the `.yace` potential and return forces to EON via stdout/files.
4.  **Finalize Orchestrator**: Update the main loop to support "kMC Mode" exploration, where the system alternates between MD and kMC based on the user config.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Elastic Check**: Calculate elastic constants for a perfect Lennard-Jones crystal (should match analytical values).
*   **Report Gen**: Verify that a HTML/JSON report is generated summarizing the validation results.

### 5.2. Integration Testing
*   **kMC Loop**: Run EON for 5 steps on a simple adatom diffusion problem. Verify it finds a saddle point and moves the atom.
*   **Grand Challenge CI**: Run the full pipeline on a tiny FePt cluster (Mocked DFT). Verify the sequence:
    1.  Init
    2.  MD (Deposit) -> Halt
    3.  Train
    4.  kMC (Order) -> Find Saddle
    5.  Validate -> Pass
