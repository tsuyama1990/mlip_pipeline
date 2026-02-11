# Cycle 07 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 7.1: Deposition Simulation (Mock MD)
*   **Goal**: Verify that LAMMPS can be configured to deposit atoms at regular intervals.
*   **Action**:
    1.  User updates `config.yaml` to include a `deposition` section.
    2.  User runs `pyacemaker run-loop`.
    3.  User inspects the `dump.lammpstrj` (or the input script in CI mode).
*   **Success Criteria**:
    *   The `in.lammps` file contains `fix deposit`.
    *   The simulation logs show "Atoms deposited: 5/10" (or similar progress).
    *   The trajectory shows new atoms appearing over time.

### Scenario 7.2: EON Connection (Ordering Simulation)
*   **Goal**: Verify the connection between PyAceMaker and the EON (kMC) client.
*   **Action**:
    1.  User installs (or mocks) EON.
    2.  User updates `config.yaml` to include an `eon` section.
    3.  User runs the loop.
*   **Success Criteria**:
    *   The `eon_driver.py` script is generated and executable.
    *   The `config.ini` file for EON is created correctly.
    *   The simulation logs show "Starting EON client...".
    *   The final structure has different atomic positions than the initial one (indicating events occurred).

### Scenario 7.3: Full Fe/Pt Scenario Run (Mock)
*   **Goal**: Verify the complete "Divide & Conquer" workflow described in `FINAL_UAT.md`.
*   **Action**:
    1.  User runs the notebook `02_Deposition_and_Ordering.ipynb` in Mock Mode.
    2.  The notebook executes: `Train MgO` -> `Train FePt` -> `Train Interface` -> `Deposition MD` -> `Ordering kMC`.
*   **Success Criteria**:
    *   All cells execute without error.
    *   The final plot shows a structure with deposited atoms.
    *   The "ordering parameter" calculation returns a plausible value (or a mocked increase).

## 2. Behavior Definitions (Gherkin Style)

### Feature: Deposition
**Scenario**: Adding atoms during MD
  **Given** a deposition configuration with rate R and species S
  **When** the DynamicsEngine starts
  **Then** it should insert S atoms every R steps
  **And** the number of atoms in the system should increase

### Feature: EON Integration
**Scenario**: Running aKMC for long timescales
  **Given** a valid potential and an initial structure
  **When** the EONWrapper is invoked
  **Then** it should launch the EON client
  **And** the client should use `eon_driver.py` to evaluate the potential
  **And** it should find saddle points and evolve the system

### Feature: OTF in kMC
**Scenario**: Handling uncertainty during saddle search
  **Given** the EON driver encounters a high-gamma configuration
  **When** it evaluates the structure
  **Then** it should exit with a specific error code (100)
  **And** the Orchestrator should interpret this as a Halt event
