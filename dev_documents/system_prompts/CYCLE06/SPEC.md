# Cycle 06 Specification: Validator & Full Orchestration

## 1. Summary

In this final cycle, we close the loop. We implement the **Validator**, the quality assurance gatekeeper that prevents "garbage" potentials from being deployed. It runs physical stability tests (Elastic Constants, Phonons, EOS) to ensure the potential captures the correct physics, not just low RMSE on the training set. We also finalize the **Orchestrator**, enabling it to seamlessly connect all the "Real" components developed in Cycles 02-05 (Generator, Oracle, Trainer, Dynamics) and execute the full active learning pipeline defined in the "Fe/Pt on MgO" scenario.

## 2. System Architecture

Files in **bold** are the focus of this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── **physics.py**      # Physics-based validation suite
│   │   └── **report.py**       # HTML report generator
├── core/
│   ├── **orchestrator.py**     # Finalize logic
└── domain_models/
    ├── **config.py**           # Finalize ValidatorConfig
```

## 3. Design Architecture

### 3.1. PhysicsValidator (`components/validator/physics.py`)
Inherits from `BaseValidator`.
-   **Config**: `elastic_tolerance` (float), `phonon_check` (bool).
-   **Method `validate(potential)`**:
    1.  **Elastic Constants**: Calculates $C_{11}, C_{12}, C_{44}$ for standard phases (fcc/bcc). Checks Born stability criteria ($C_{11} > C_{12}$, etc.).
    2.  **EOS Curve**: Compresses/Expands the cell ($\pm 10\%$) and fits Birch-Murnaghan. Checks if Bulk Modulus $B_0$ is positive and reasonable.
    3.  **Phonon Stability** (Optional): If `phonopy` is present, calculates band structure to check for imaginary frequencies ($\omega^2 < 0$).
    4.  **Result**: Returns `{"passed": bool, "metrics": dict}`.

### 3.2. Full Orchestration Logic
The `Orchestrator` needs to handle the complex state transitions of the "Fe/Pt on MgO" scenario:
-   **Phase 1 (Bulk)**: Loop until bulk validation passes.
-   **Phase 2 (Interface)**: Switch Generator to "Interface Mode", run cycles.
-   **Phase 3 (Production)**: Deploy potential for MD/kMC.

## 4. Implementation Approach

1.  **Update Config**: Add `ValidatorConfig`.
2.  **Implement PhysicsValidator**:
    -   Use `ase.eos` for Equation of State.
    -   Implement a finite-difference elasticity calculator using ASE `StrainFilter` or manual displacement.
3.  **Refine Orchestrator**:
    -   Ensure it properly handles the `Halt` signals from Dynamics (Cycle 05) and triggers the Oracle (Cycle 03).
    -   Implement the "Convergence Check": If Validator passes AND Dynamics runs without Halt, mark as "Converged".
4.  **CLI Polish**: Ensure `mlip-pipeline run` produces clean, informative logs and a final `report.html`.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Elasticity**: Create a dummy potential (LJ) and verify the validator calculates positive elastic constants.
-   **Stability Check**: Feed a potential that produces negative modulus (unstable) and verify `validate()` returns `False`.

### 5.2. Integration Testing (The Grand Finale)
-   **Scenario**: "Fe/Pt Demo"
-   **Config**: Full "Real Mode" configuration (using Mocks where binaries are missing).
-   **Sequence**:
    1.  Gen (Random FePt) -> Oracle (Mock Energy) -> Train (Mock Pacemaker).
    2.  Val (Elasticity Check).
    3.  Dyn (Mock LAMMPS Halt).
    4.  Oracle (New Data) -> Train -> Val -> Dyn (Success).
-   **Success**: The pipeline finishes with "Convergence Reached" after the relearning step.
