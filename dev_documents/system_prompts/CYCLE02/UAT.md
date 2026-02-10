# Cycle 02 User Acceptance Test (UAT)

## 1. Test Scenarios

### **UAT-02: Structure Generation & Exploration**
*   **Goal**: Verify that the system can generate physically valid and diverse atomic structures (both random and adaptive) based on configuration.
*   **Priority**: High
*   **Success Criteria**:
    *   The `RandomGenerator` can create `n` structures without overlapping atoms (< 1.5 Å).
    *   The `AdaptiveGenerator` can read a `PolicyConfig` and apply different strategies (e.g., strain, vacancies).
    *   Generated structures can be saved to disk (e.g., `candidates.xyz`) and visualized (e.g., using `ase.visualize` or `ovito`).

## 2. Behavior Definitions (Gherkin)

### Scenario: Random Structure Generation
**GIVEN** a `config.yaml` with `generator.type: random` and `generator.n_structures: 10`
**WHEN** the user runs `mlip-auto run-loop` (or unit test)
**THEN** the output directory `candidates/` should contain 10 structure files (or one combined file)
**AND** all structures should have valid cell parameters
**AND** no two atoms in any structure should be closer than 1.5 Å (Collision Check)

### Scenario: Adaptive Policy Selection (Strain)
**GIVEN** a `config.yaml` with `generator.type: adaptive` and `policy.strain_range: 0.1` (10%)
**WHEN** the generator is called with context `iteration=1`
**THEN** the generated structures should exhibit lattice parameters different from the ideal bulk
**AND** the volume change should be within ±10% of the equilibrium volume
