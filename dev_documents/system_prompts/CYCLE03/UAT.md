# Cycle 03 UAT: The Oracle (DFT Automation)

## 1. Test Scenarios

### Scenario 01: "The Physicist" (DFT Calculation)
**Priority**: High
**Description**: Verify that the system can run a single-point SCF calculation on a simple structure (e.g., Bulk Si) using Quantum Espresso (or a realistic mock).

**Gherkin Definition**:
```gherkin
GIVEN a valid structure (Bulk Si, 2 atoms)
AND a configured Oracle with "type=qe"
WHEN I execute the computation
THEN the system should generate a valid "pw.in" file
AND the system should execute "pw.x"
AND the returned structure should have "energy", "forces", and "stress" properties
AND the energy should be negative (physically meaningful)
```

### Scenario 02: "The Healer" (Self-Correction)
**Priority**: Medium
**Description**: Simulate a convergence failure and verify the system attempts to fix it.

**Gherkin Definition**:
```gherkin
GIVEN a difficult structure (e.g., surface slab with vacuum)
AND a simulated "SCF Convergence Error" (via mock or tough settings)
WHEN I execute the computation
THEN the system should detect the failure
AND the system should retry with a lower "mixing_beta"
AND the logs should show "Retrying with modified parameters..."
```

## 2. Verification Steps

1.  **Environment Check**: Ensure `quantum-espresso` (or `pw.x` in PATH) is available, OR configure the `MockQE` script.
2.  **Run Script**: Create `scripts/test_oracle.py` that:
    *   Creates a `Structure` (Si).
    *   Initializes `QEOracle`.
    *   Calls `compute()`.
    *   Prints Energy and Max Force.
3.  **Validate Output**: Check `pw.out` (or temp file) for "JOB DONE".
