# Cycle 07 User Acceptance Test (UAT)

## 1. Test Scenarios

### **UAT-07: Advanced Dynamics Simulation (Deposition + kMC)**
*   **Goal**: Verify that the system can simulate complex physical phenomena: physical vapor deposition (PVD) and subsequent ordering (aKMC), handling long timescales and non-equilibrium events.
*   **Priority**: High (Grand Challenge Requirement)
*   **Success Criteria**:
    *   The LAMMPS deposition script correctly inserts atoms into the vacuum region.
    *   The `hybrid/overlay` potential prevents atoms from exploding upon impact.
    *   The EON driver successfully interfaces with Pacemaker to calculate Energy/Forces.
    *   The system can detect uncertainty during aKMC (e.g., at a saddle point) and trigger a Halt.

## 2. Behavior Definitions (Gherkin)

### Scenario: Deposition Setup
**GIVEN** a substrate structure (MgO) and a `DepositionConfig`
**WHEN** the `DepositionDynamics` engine generates the LAMMPS script
**THEN** the script should contain `fix deposit`
**AND** the deposition region should be defined *above* the substrate surface (Z > Z_surf)
**AND** the `pair_style` should be `hybrid/overlay pace zbl`

### Scenario: EON Integration
**GIVEN** a valid potential and an initial structure
**WHEN** `EONWrapper.run_kmc()` is called
**THEN** an EON working directory should be created with `config.ini`
**AND** `eon_driver.py` should be present in `potentials/`
**AND** `eonclient` should run (mocked) and return a new structure (the product state)

### Scenario: kMC Halt
**GIVEN** a structure at a transition state with high uncertainty ($\gamma > 5.0$)
**WHEN** the `eon_driver.py` evaluates this structure
**THEN** the driver should exit with a non-zero error code (Halt Signal)
**AND** the `EONWrapper` should catch this and return `halted=True`
