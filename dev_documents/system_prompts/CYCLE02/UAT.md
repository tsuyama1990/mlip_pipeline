# Cycle 02 UAT: First Light

## 1. Test Scenarios

### 1.1. Scenario: Structure Generation & Visualisation
**ID**: UAT-CY02-001
**Priority**: High
**Description**: Generate a set of perturbed structures from a seed crystal and verify they are physically reasonable.

**Steps:**
1.  **Setup**: Create a Python script `uat_gen.py` that loads a Silicon unit cell (using `ase.build.bulk`).
2.  **Execution**: Use `StructureGenerator` to create 10 perturbed structures with `rattle=0.1`.
3.  **Visualisation**: Use `ase.visualise.write` to save them as `generated.xyz`.
4.  **Verification**: Open `generated.xyz` in visualisation software (OVITO/Vestra) or check manually that atomic positions are slightly disordered but the lattice is intact.

### 1.2. Scenario: The Self-Healing Oracle (Simulated)
**ID**: UAT-CY02-002
**Priority**: Medium
**Description**: Simulate a DFT convergence failure and verify the system retries with new parameters.

**Steps:**
1.  **Setup**: Configure `QEOracle` with a `MockCalculator` that raises a `ConvergenceError` on the first call and succeeds on the second.
2.  **Execution**: Run `oracle.compute(structure)`.
3.  **Observation**: The logs should show: "Calculation failed. Retrying with mixing_beta=0.3...".
4.  **Verification**: The final result should contain energy/forces, and the `provenance` metadata should indicate "Retried: 1".

### 1.3. Scenario: Real DFT Calculation (Requires QE)
**ID**: UAT-CY02-003
**Priority**: Low (Optional if QE not installed)
**Description**: Run a real DFT calculation on a water molecule to verify the binary interface.

**Steps:**
1.  **Setup**: Ensure `pw.x` is in PATH. Set up a valid `config.yaml` with pseudopotentials.
2.  **Execution**: Run the Orchestrator with `type: quantum_espresso` for a single iteration.
3.  **Verification**: Check the `active_learning/iter_001/dft/` folder. It should contain `pw.in`, `pw.out`, and `pw.xml`. The `pw.out` should end with "JOB DONE".

## 2. Behaviour Definitions

**Feature**: Robust DFT Execution

**Scenario**: Automatic Retry on SCF Divergence

**GIVEN** an atomic structure that is difficult to converge (e.g., magnetic iron)
**AND** the Oracle is configured with default `mixing_beta = 0.7`
**WHEN** the DFT calculation fails with "convergence not achieved"
**THEN** the system should catch the error
**AND** the system should update the parameters to `mixing_beta = 0.3`
**AND** the system should re-submit the calculation automatically
**AND** if the second attempt succeeds, the result is returned to the user with a warning log.

**Feature**: Periodic Embedding

**Scenario**: Cutting a cluster for DFT

**GIVEN** a large Supercell (100 atoms) with a defect in the centre
**WHEN** the `Embedding` logic is applied with `radius=5.0`
**THEN** a new, smaller Unit Cell (approx 20 atoms) should be created
**AND** the defect atom should be at the centre of the new cell
**AND** there should be at least 10 Angstroms of vacuum (or buffer) around the cluster
