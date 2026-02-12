# Cycle 07 UAT: aKMC Integration

## 1. Test Scenarios

| ID | Scenario Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **07-1** | **EON Configuration Gen** | Verify that `EONDriver` creates valid `config.ini` and structure files (`.con`) from an ASE atoms object. | High |
| **07-2** | **Potential Driver Interface** | Verify that the Python script called by EON correctly computes Energy/Forces and handles the Uncertainty check. | Critical |
| **07-3** | **Saddle Point Halt** | Verify that if a saddle point search enters a high-uncertainty region, the system Halts and triggers Active Learning. | Medium |

## 2. Behavior Definitions (Gherkin)

### Scenario 07-2: Potential Driver Interface
```gherkin
GIVEN a running "potential_server.py" process
WHEN I send atomic coordinates via Standard Input
THEN it should output the Energy and Forces in the format expected by EON
AND it should log the maximum extrapolation grade
```

### Scenario 07-3: Saddle Point Halt
```gherkin
GIVEN an aKMC simulation where a transition state has high uncertainty
WHEN the "potential_server.py" detects gamma > threshold
THEN it should exit with code 100
AND the "EONDriver" should catch this exit code
AND return a "HaltEvent" to the Orchestrator
```
