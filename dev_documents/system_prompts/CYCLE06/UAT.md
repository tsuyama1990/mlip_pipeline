# Cycle 06 UAT: Local Learning Loop

## 1. Test Scenarios

| ID | Scenario Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **06-1** | **Local Candidate Generation** | Verify that the system generates diverse but physically reasonable perturbations around a "Halt" structure. | High |
| **06-2** | **Full Active Learning Cycle** | Verify the complete chain: Halt -> Generate Candidates -> Select -> DFT -> Train -> Resume. | Critical |
| **06-3** | **Fine-Tuning Efficiency** | Verify that the "Update" step uses `initial_potential` to fine-tune rapidly (e.g., 10 epochs) rather than training from scratch. | Medium |

## 2. Behavior Definitions (Gherkin)

### Scenario 06-1: Local Candidate Generation
```gherkin
GIVEN a structure where atom 10 has high uncertainty
WHEN the Candidate Generator runs with "n_candidates=20"
THEN it should return 20 new structures
AND each structure should have atom 10 displaced by max 0.1 Angstrom
AND the topology should remain consistent (no bond breaking)
```

### Scenario 06-2: Full Active Learning Cycle
```gherkin
GIVEN a simulation that Halts at step 500
WHEN the Active Learner is triggered
THEN it should produce a new potential version "v2"
AND the simulation should resume from step 500 using "v2"
AND the simulation should proceed past step 501 (assuming the hole is filled)
```
