# Cycle 07: Advanced Exploration

## 1. Summary
This cycle enhances the system's exploration capabilities beyond simple MD. We implement the **Structure Generator** for systematic sampling of defects (vacancies, interstitials) and strain states. We also integrate **EON** to enable Kinetic Monte Carlo (kMC) simulations, allowing the system to learn rare events and long-timescale phenomena. Finally, we implement the **Adaptive Policy**, which dynamically adjusts exploration parameters based on the current state of the potential.

## 2. System Architecture

We add the `generator` module and `phases/exploration` extensions.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── generator/
│   ├── **__init__.py**
│   ├── **defects.py**           # Defect structure generation
│   ├── **strain.py**            # Strain/Deformation generation
│   └── **policy.py**            # Adaptive Exploration Policy
└── orchestration/
    └── phases/
        └── dynamics/
            └── **eon_wrapper.py**   # EON (kMC) integration
└── tests/
    └── **test_generator.py**
```

## 3. Design Architecture

### Structure Generator (`defects.py`, `strain.py`)
*   **Defects**: systematically remove atoms (vacancy) or add atoms (interstitial) to supercells. Use symmetry (spglib) to avoid generating identical defects.
*   **Strain**: Apply random shear and volumetric strains to create "rattled" structures for training elasticity.

### EON Integration (`eon_wrapper.py`)
*   Similar to LAMMPS driver but controls `eonclient`.
*   Monitors saddle point searches. If the barrier search hits a high-$\gamma$ configuration, it halts and extracts the structure.

### Adaptive Policy (`policy.py`)
*   A decision module.
*   **Inputs**: Current Cycle, Validation Score, Uncertainty Distribution.
*   **Outputs**: Next simulation type (MD vs kMC), Temperature, Pressure.
*   **Logic**:
    *   If validation fails on Elasticity -> Recommend High Strain sampling.
    *   If validation fails on Phonons -> Recommend Low T MD.
    *   If Cycle > 5 -> Switch to kMC for rare events.

## 4. Implementation Approach

1.  **Defect Generator**: Implement classes to generate defects using `pymatgen` or `ase` utilities.
2.  **EON Wrapper**: Implement the EON client runner. Since EON is external, focus on the Python interface that EON calls to get forces (the "Client Pot" driver).
3.  **Policy Engine**: Implement a simple rule-based system first (If X then Y).
4.  **Integration**: Update `DynamicsPhase` to accept an `engine` parameter ("lammps" or "eon") determined by the Policy.

## 5. Test Strategy

### Unit Testing
*   **`test_defects.py`**: Verify that `VacancyGenerator` returns correct number of structures for a simple BCC/FCC lattice.
*   **`test_policy.py`**: Feed dummy stats (e.g., "Elasticity Failed") and verify the recommended action matches.

### Integration Testing
*   **Mock EON**: Simulate an EON run that halts due to high energy/uncertainty. Verify the structure is captured.
