# Cycle 02 User Acceptance Test (UAT)

## 1. Test Scenarios

### Scenario 1: Random Structure Generation
**Priority**: P0 (Critical)
**Description**: Verify that the random structure generator produces diverse atomic configurations.
**Steps**:
1.  Configure `config.yaml` with `strategy: random`.
2.  Run `mlip-runner run config.yaml`.
3.  Inspect the output log.
4.  Verify that `N` structures are generated and written to disk (e.g., `work_dir/iter_000/candidates/001.xyz`).
5.  Check that atoms have `displacement` > 0 compared to the initial structure.

### Scenario 2: Defect Engineering
**Priority**: P1 (High)
**Description**: Verify that the defect generator correctly removes atoms (vacancies) or swaps species (antisites).
**Steps**:
1.  Configure `config.yaml` with `strategy: defects`.
2.  Run `mlip-runner run config.yaml`.
3.  Inspect the generated structure files.
4.  Check the number of atoms. For a 2x2x2 supercell of FCC (32 atoms), a vacancy generator should produce 31 atoms.

### Scenario 3: M3GNet Cold Start (Optional Dependency)
**Priority**: P2 (Medium)
**Description**: Verify that the system can use a pre-trained M3GNet model to relax initial structures if the library is available.
**Steps**:
1.  Install `matgl` and `dgl`.
2.  Configure `config.yaml` with `strategy: m3gnet`.
3.  Run the generator.
4.  Verify that the final energy/forces are populated in the structure's info dictionary (from M3GNet prediction).

## 2. Behavior Definitions (Gherkin)

### Feature: Structure Generation

**Scenario**: Generate Random Rattled Structures
    **Given** a `config.yaml` with `generator.strategy: random` and `rattle_amplitude: 0.1`
    **When** the Orchestrator executes the Exploration phase
    **Then** 10 new structure files should appear in `work_dir/iter_000/candidates/`
    **And** the atomic positions should differ from the perfect crystal by at most 0.1 Angstrom

**Scenario**: Generate Vacancy Defects
    **Given** a perfect FCC crystal of 32 atoms
    **When** the Defect Generator is called with `type: vacancy`
    **Then** the resulting structure should have 31 atoms
    **And** the lattice parameters should remain unchanged
