# CYCLE 06 Specification: Scale-up, Validation & Integration

## 1. Summary

Cycle 06 completes the system by adding two critical capabilities: Long-timescale simulation (Scale-up) and rigorous quality assurance (Validation). We will integrate with `EON` to perform Adaptive Kinetic Monte Carlo (aKMC) simulations, allowing the system to model phenomena like diffusion and ordering that are too slow for MD. Simultaneously, we will implement the `PhysicsValidator`, a module that subjects the trained potential to a battery of physical tests—Phonon stability, Elastic constants, and Equation of State (EOS)—before certifying it for production use. This cycle culminates in the execution of the full "Fe/Pt on MgO" scenario.

## 2. System Architecture

```ascii
src/mlip_autopipec/
├── services/
│   ├── external/
│   │   └── eon_interface.py       # [CREATE] Wrapper for EON (kMC)
│   └── validation/
│       ├── __init__.py
│       ├── validator.py           # [CREATE] Main Validation Logic
│       └── physics_tests.py       # [CREATE] Elasticity, EOS, Phonon logic
```

## 3. Design Architecture

### EON Interface (`eon_interface.py`)
-   **Role**: Manages aKMC simulations.
-   **Integration**: EON is a client-server system. We will wrap the client execution.
-   **Driver**: We must provide a Python script (`pace_driver.py`) that EON calls to get energy/forces from our `.yace` potential.

### Physics Validator (`validator.py`)
-   **Role**: Runs a suite of tests and aggregates results into a `ValidationResult`.
-   **Tests**:
    1.  **EOS**: Fit Energy vs Volume to Birch-Murnaghan equation. Check Bulk Modulus ($B_0$).
    2.  **Elasticity**: Calculate elastic tensor $C_{ij}$ by deforming the cell. Check Born stability criteria.
    3.  **Phonons**: (Optional for V1) Calculate phonon dispersion to check for imaginary frequencies (instability).

## 4. Implementation Approach

1.  **Implement `PhysicsTests`**:
    -   Write `calculate_eos(atoms, potential)`: Scan volumes, calc energy, fit curve.
    -   Write `calculate_elastic_constants(atoms, potential)`: Apply small strains, measure stress, solve linear equations.

2.  **Implement `EonKMC`**:
    -   Generate `config.ini` for EON.
    -   Generate the `pace_driver.py` script that loads `pyacemaker` to evaluate the potential.
    -   Execute `eonclient`.

3.  **Final Integration**:
    -   Update `Orchestrator` to include the Validation step at the end of each cycle (or explicitly requested).
    -   Ensure the "Fe/Pt on MgO" tutorial scenario can utilize these new components.

## 5. Test Strategy

### Unit Testing
-   **EOS Fit**: Provide synthetic E-V data and verify the fitted Bulk Modulus is correct.
-   **Elastic Logic**: Provide synthetic Stress-Strain data and verify $C_{ij}$ calculation.

### Integration Testing
-   **EON Driver**: Verify that the generated `pace_driver.py` can actually load the potential and return forces when fed an atomic configuration via stdin/stdout.
-   **Full Scenario**: Run the "Mock Mode" version of the Fe/Pt tutorial to ensure all components (Orchestrator -> MD -> OTF -> Oracle -> Trainer -> Validation) hand off data correctly.
