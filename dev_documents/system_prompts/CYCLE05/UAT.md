# Cycle 05 UAT: Dynamics Engine (MD & Uncertainty)

## 1. Test Scenarios

| ID | Scenario Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **05-1** | **Hybrid Potential Stability** | Verify that `pair_style hybrid/overlay` is correctly applied, preventing atoms from overlapping (nuclear fusion) even at high temperatures. | High |
| **05-2** | **Uncertainty Watchdog Trigger** | Verify that the simulation halts (Active Learning Trigger) when the extrapolation grade ($\gamma$) exceeds the threshold. | Critical |
| **05-3** | **MD Trajectory Output** | Verify that a successful MD run produces a valid trajectory file (dump) readable by ASE/OVITO. | Medium |

## 2. Behavior Definitions (Gherkin)

### Scenario 05-1: Hybrid Potential Stability
```gherkin
GIVEN a configuration with "dynamics.hybrid_potential: zbl"
AND a simulation temperature of 5000K (very high)
WHEN the Dynamics Engine runs
THEN the minimum distance between atoms should never drop below 0.5 Angstrom
(Because the ZBL core repulsion should dominate)
```

### Scenario 05-2: Uncertainty Watchdog Trigger
```gherkin
GIVEN a configuration with "dynamics.uncertainty_threshold: 5.0"
AND a structure that is chemically very different from the training set
WHEN the Dynamics Engine starts the simulation
THEN it should halt before completing all steps
AND it should return a "HaltEvent" status
AND the log should indicate "Fix halt condition met"
```
