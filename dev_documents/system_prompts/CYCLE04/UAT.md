# Cycle 04 UAT: The Trainer (Pacemaker Integration)

## 1. Test Scenarios

### Scenario 01: "The Apprentice" (Model Training)
**Priority**: Critical
**Description**: Verify that the system can train a simple ACE potential from a small dataset.

**Gherkin Definition**:
```gherkin
GIVEN a dataset with 10 structures (Al bulk)
AND a configured Trainer with "type=pacemaker"
WHEN I execute the training command
THEN the system should generate "input.yaml"
AND the system should execute "pace_train"
AND the output directory should contain "potential.yace"
AND the log should show "Training completed successfully"
```

### Scenario 02: "The Specialist" (Active Set Selection)
**Priority**: High
**Description**: Verify that the system selects a subset of structures using Active Set optimization.

**Gherkin Definition**:
```gherkin
GIVEN a dataset with 100 structures
AND an Active Set target of 10
WHEN I execute the selection process
THEN the system should return exactly 10 structures
AND the selected structures should be the most diverse (highest D-optimality)
```

## 2. Verification Steps

1.  **Environment Check**: Ensure `pacemaker` (or `pace_train` in PATH) is available.
2.  **Run Script**: Create `scripts/test_trainer.py`.
    *   Load dataset.
    *   Init `PacemakerTrainer`.
    *   Train.
    *   Check `potential.yace` existence.
3.  **Validate Output**: Inspect `training.log` for loss convergence (if real training runs).
