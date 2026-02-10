# Cycle 05 User Acceptance Test (UAT)

## 1. Test Scenarios

### **UAT-05: Molecular Dynamics & Uncertainty Halt**
*   **Goal**: Ensure that LAMMPS simulations run correctly with hybrid potentials and stop automatically when the uncertainty threshold ($\gamma$) is exceeded.
*   **Priority**: High (Critical for Active Learning)
*   **Success Criteria**:
    *   The generated LAMMPS script contains `pair_style hybrid/overlay`.
    *   The generated script contains `fix halt`.
    *   When the uncertainty is low, the simulation completes normally.
    *   When the uncertainty is high (mocked/simulated), the simulation halts and returns the failing structure.

## 2. Behavior Definitions (Gherkin)

### Scenario: Hybrid Potential Generation
**GIVEN** a `DynamicsConfig` with `baseline_potential: zbl`
**WHEN** the `LAMMPSDynamics` engine initializes
**THEN** the generated `in.lammps` file should contain:
    *   `pair_style hybrid/overlay pace zbl`
    *   `pair_coeff * * pace`
    *   `pair_coeff * * zbl`

### Scenario: Successful MD Run
**GIVEN** a potential with low uncertainty (well-trained)
**WHEN** the MD runs for 1000 steps
**THEN** the `DynamicsResult` should have `halted=False`
**AND** the `final_structure` should be valid (no exploded atoms)

### Scenario: Uncertainty Halt
**GIVEN** a potential with high uncertainty (poorly-trained)
**OR** a mocked simulation that returns a "Halt" signal at step 500
**WHEN** the MD runs
**THEN** the `DynamicsResult` should have `halted=True`
**AND** the `halt_structure` should correspond to step 500
**AND** the log should indicate "Halt triggered by fix halt"
