# Cycle 06 UAT: Validation, Integration & Orchestration

## 1. Test Scenarios

### Scenario 01: "The Auditor" (Physics Validation)
**Priority**: Critical
**Description**: Verify that the generated potential is physically stable (no imaginary phonons, positive elastic constants).

**Gherkin Definition**:
```gherkin
GIVEN a trained potential for "Al bulk"
WHEN I execute the validation suite
THEN the phonon spectrum should have no imaginary modes (stable)
AND the Bulk Modulus should be approximately 76 GPa (within 10%)
AND the validation report "validation_report.html" should be generated
```

### Scenario 02: "The Grand Finale" (Full System E2E)
**Priority**: Critical
**Description**: Execute the full Fe/Pt on MgO pipeline from `config.yaml` to deployment.

**Gherkin Definition**:
```gherkin
GIVEN a complete "config_fe_pt_mgo.yaml"
AND execution mode is set to "CI Mode" (Small system)
WHEN I execute "mlip-pipeline run config.yaml"
THEN the system should run for N cycles
AND the "active_learning" directory should contain iteration folders
AND the final "potential.yace" should exist
AND the final validation should pass
```

## 2. Verification Steps

1.  **Environment Check**: Ensure all dependencies (`phonopy`, etc.) are installed.
2.  **Run Script**: Execute `uv run mlip-pipeline run config_fe_pt_mgo.yaml`.
3.  **Validate Output**: Check `final_report.json` for success status.
