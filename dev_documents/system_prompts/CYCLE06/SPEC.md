# CYCLE 06 Specification: Advanced Orchestration (kMC & Validation)

## 1. Summary

In this final cycle, we extend the system's capabilities to "Long Time Scale" and "High Quality Assurance". We integrate **EON** for Adaptive Kinetic Monte Carlo (aKMC) simulations and a comprehensive **Validator** suite that checks physical properties (phonons, elasticity) before deploying potentials.

## 2. System Architecture

Files to be modified/created:

```ascii
src/mlip_autopipec/
├── dynamics/
│   └── **eon_driver.py**       # EON Wrapper
├── validation/
│   ├── **__init__.py**
│   ├── **metrics.py**          # RMSE, Parity Plots
│   └── **stability.py**        # Phonon & Elasticity
└── orchestration/
    └── orchestrator.py         # Final integration
```

## 3. Design Architecture

### 3.1. `EonDynamics`
*   **Role**: Runs kMC simulations to find rare events (diffusion, reactions).
*   **Integration**:
    *   Generates `config.ini` for EON.
    *   Creates a `pace_driver.py` script that EON calls to get energy/forces from the ACE potential.
    *   Handling EON's output (states, processes) and feeding them back into the active learning loop if uncertainty is high.

### 3.2. `Validator`
*   **Suite**:
    1.  **MetricValidator**: Calculates RMSE for Energy, Force, Stress on a held-out test set.
    2.  **StabilityValidator**: Runs `phonopy` (if available) to check for imaginary frequencies.
    3.  **ElasticValidator**: Calculates elastic constants ($C_{ij}$) and checks Born stability criteria.
*   **Reporting**: Generates an HTML report summarizing the quality.

## 4. Implementation Approach

1.  **Validator**:
    *   Implement `src/mlip_autopipec/validation/stability.py`.
    *   Use `ase.phonons` or `phonopy` API.
    *   Use `ase.elasticity` for $C_{ij}$.
2.  **EON Driver**:
    *   Implement `src/mlip_autopipec/dynamics/eon_driver.py`.
    *   Requires writing a standalone Python script (`pace_driver.py`) that loads `potential.yace` and communicates with EON via stdin/stdout.
3.  **Orchestrator**:
    *   Add a "Validation" step at the end of each cycle.
    *   If validation passes, copy the potential to `production/`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_elastic_validation.py`**:
    *   Create a dummy cubic crystal.
    *   Verify that `ElasticValidator` returns correct $C_{11}, C_{12}, C_{44}$.

### 5.2. Integration Testing
*   **`test_eon_interface.py`**:
    *   **Mock Mode**: Mock the EON client call. Verify that the `pace_driver.py` script is generated correctly and works (can load potential and return forces).
*   **`test_full_scenario.py`**:
    *   The "Fe/Pt on MgO" scenario from `FINAL_UAT.md`.
    *   Verify that the Orchestrator can transition from MD (Deposition) to kMC (Ordering).
