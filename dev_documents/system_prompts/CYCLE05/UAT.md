# Cycle 05 UAT: The Dynamics Engine (Inference & OTF)

## 1. Test Scenarios

### Scenario 01: "The Explorer" (Stable MD Run)
**Priority**: Critical
**Description**: Verify that the system can run a stable MD simulation using a Hybrid Potential (ACE + ZBL) without crashing due to atomic overlap.

**Gherkin Definition**:
```gherkin
GIVEN a trained potential "potential.yace"
AND a starting structure (Al bulk)
AND a target temperature of 300K
WHEN I execute the dynamics engine for 1000 steps
THEN the simulation should complete successfully (CONVERGED)
AND the final temperature should be approximately 300K
AND the trajectory file should be created
```

### Scenario 02: "The Watchdog" (Uncertainty Halt)
**Priority**: Critical
**Description**: Verify that the simulation halts immediately when the model encounters an unknown configuration (high gamma).

**Gherkin Definition**:
```gherkin
GIVEN a potential trained only on low-energy structures
AND a simulation set to very high temperature (2000K, melting)
AND a gamma threshold of 2.0
WHEN I execute the dynamics engine
THEN the simulation should halt before completing all steps (HALTED)
AND the returned structure should have a max gamma > 2.0
AND the logs should indicate "Uncertainty limit exceeded"
```

## 2. Verification Steps

1.  **Environment Check**: Ensure `lammps` binary is available or mocked.
2.  **Run Script**: Create `scripts/test_dynamics.py`.
    *   Run Case 1 (Stable): assert `status == CONVERGED`.
    *   Run Case 2 (Unstable): assert `status == HALTED`.
3.  **Validate Output**: Check `log.lammps` for "Fix halt condition met".
