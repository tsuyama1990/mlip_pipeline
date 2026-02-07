# Cycle 04 UAT: The Watchdog

## 1. Test Scenarios

### 1.1. Scenario: Hybrid Potential Safety
**ID**: UAT-CY04-001
**Priority**: High
**Description**: Verify that the Hybrid Potential prevents atomic fusion.

**Steps:**
1.  **Setup**: Create a `test.data` with two atoms placed very close ($r = 0.5 \AA$).
2.  **Execution**: Run `DynamicsEngine.run_exploration()` with `hybrid=True`.
3.  **Observation**: The MD should run (or explode with high energy), but the atoms should *separate* rapidly due to the ZBL repulsion.
4.  **Negative Test**: Run with `hybrid=False`. If the ACE potential has a "hole", the atoms might stay close or fuse.

### 1.2. Scenario: Uncertainty Halt Trigger
**ID**: UAT-CY04-002
**Priority**: High
**Description**: Verify that the simulation stops when it encounters a structure with high extrapolation grade ($\gamma$).

**Steps:**
1.  **Setup**:
    *   Train a potential on "Liquid Si".
    *   Run MD on "Crystalline Si" (which is locally different) or force a high temperature (5000K).
    *   Set `threshold = 2.0` (very strict).
2.  **Execution**: Start the MD.
3.  **Verification**:
    *   LAMMPS should exit with a non-zero code (or specific error message).
    *   Log should contain `Rule ... unsatisfied`.
    *   The `dump` file should contain the structure at the moment of failure.
    *   The Orchestrator should log "Halt detected at step X. Extracting structure...".

### 1.3. Scenario: The Loop (Halt -> Train -> Resume)
**ID**: UAT-CY04-003
**Priority**: Medium
**Description**: Run a mini-active learning campaign.

**Steps:**
1.  **Setup**: Mock Mode. Config `iterations=5`.
2.  **Execution**: Run `main.py`.
3.  **Observation**:
    *   Iter 1: Dynamics halts.
    *   Iter 2: Dynamics halts.
    *   Iter 3: Dynamics runs to completion (simulating that the model learned).
4.  **Verification**: Check `active_learning/iter_003/report.json` shows `status: "converged"`.

## 2. Behaviour Definitions

**Feature**: Active Learning Watchdog

**Scenario**: Detecting Unsafe Extrapolation

**GIVEN** an MD simulation running with a Pacemaker potential
**AND** the `fix halt` command is active with threshold $\gamma_{max} = 5.0$
**WHEN** the system evolves into a configuration where $\gamma = 5.1$
**THEN** the LAMMPS process should terminate immediately
**AND** the Python wrapper should detect the "Halt" condition
**AND** the wrapper should retrieve the final atomic configuration
**AND** this configuration should be flagged for labelling by the Oracle
