# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 06-01: kMC Saddle Point Search
- **Priority**: High
- **Description**: Verify that the system can perform a KMC step to find a reaction barrier.
- **Steps**:
    1. Setup a vacancy diffusion case.
    2. Run the kMC module.
    3. **Expected Result**: EON finds the saddle point (energy barrier) for the vacancy jump.

### Scenario 06-02: Adaptive Temperature Ramping
- **Priority**: Medium
- **Description**: Verify that the system automatically increases simulation temperature as the potential becomes more robust.
- **Steps**:
    1. Start a fresh run. Monitor the config logs.
    2. Allow 3 iterations to pass successfully.
    3. **Expected Result**: Iteration 1 uses T=300K. Iteration 2 uses T=600K. Iteration 3 uses T=1000K (or similar logic defined in policy).

### Scenario 06-03: Full System "Zero-Config" Demonstration
- **Priority**: Critical
- **Description**: The ultimate test. Give the system a chemical formula and wait.
- **Steps**:
    1. Provide `config.yaml` with only `elements: [Ti, O]`.
    2. Run `mlip-auto run`.
    3. Wait (possibly hours/days in real life, mocked here).
    4. **Expected Result**: The system proceeds through generation, training, active learning loops, and finally produces a potential that passes Validation.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Advanced Exploration

  Scenario: Long-timescale exploration with kMC
    GIVEN a stable potential
    WHEN I enable the "EON" engine
    THEN the system should search for rare events
    AND if a saddle point has high uncertainty
    THEN it should be added to the training set

  Scenario: Adaptive Policy execution
    GIVEN the system has not halted for 100,000 steps
    WHEN the Policy Engine evaluates the state
    THEN it should decide to "Increase Aggression"
    AND the next MD run should have higher Temperature or Pressure
```
