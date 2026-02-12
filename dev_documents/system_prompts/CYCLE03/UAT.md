# Cycle 03 UAT: Oracle (DFT Automation)

## 1. Test Scenarios

| ID | Scenario Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **03-1** | **Automated QE Calculation** | Verify the system can generate a valid Quantum Espresso input file, run it, and parse the energy/forces. | High |
| **03-2** | **Self-Healing Recovery** | Verify the system automatically recovers from a simulated SCF convergence failure. | Critical |
| **03-3** | **Cluster Embedding** | Verify that a local environment cut from a large structure is correctly embedded into a periodic box for DFT. | Medium |

## 2. Behavior Definitions (Gherkin)

### Scenario 03-1: Automated QE Calculation
```gherkin
GIVEN a configuration with "oracle.code: quantum_espresso"
AND a structure of "Bulk Si"
WHEN the Orchestrator runs the "Oracle" phase
THEN a "pw.in" file should be generated
AND the file should contain "tprnfor = .true."
AND the calculation should complete successfully
AND the output dataset should contain Forces and Stress tensors
```

### Scenario 03-2: Self-Healing Recovery
```gherkin
GIVEN a DFT calculation that fails with "convergence not achieved"
WHEN the Self-Healing logic is triggered
THEN it should retry the calculation with "mixing_beta" reduced by 50%
AND if it fails again, it should retry with "smearing" increased
AND if it succeeds, the final result should be saved
```
