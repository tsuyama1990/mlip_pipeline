# Cycle 02 UAT: Structure Generator

## 1. Test Scenarios

| ID | Scenario Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **02-1** | **Random Generation** | Verify `RandomGenerator` produces distorted structures from a seed crystal. | High |
| **02-2** | **Adaptive Temperature Schedule** | Verify `AdaptiveGenerator` increases simulation temperature across cycles as defined in config. | Critical |
| **02-3** | **Policy Switching** | Verify that the system can switch between "Random" and "Adaptive" strategies via config. | Medium |

## 2. Behavior Definitions (Gherkin)

### Scenario 02-1: Random Generation
```gherkin
GIVEN a configuration with "generator.strategy: random"
AND a seed structure of "Bulk MgO"
WHEN the Orchestrator runs the "Exploration" phase
THEN it should produce 10 candidate structures
AND each structure should have different lattice parameters (Strain)
AND each structure should have non-zero atomic displacements
```

### Scenario 02-2: Adaptive Temperature Schedule
```gherkin
GIVEN a configuration with "generator.strategy: adaptive"
AND a schedule "temperature_schedule: [300, 1000]"
WHEN the Orchestrator runs Cycle 0
THEN the generator should use T=300K
WHEN the Orchestrator runs Cycle 1
THEN the generator should use T=1000K
```
