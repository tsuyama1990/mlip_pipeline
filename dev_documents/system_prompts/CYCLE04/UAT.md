# Cycle 04 UAT: Trainer (Pacemaker Integration)

## 1. Test Scenarios

| ID | Scenario Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **04-1** | **Delta Learning Configuration** | Verify that the system correctly configures a ZBL reference potential for "Fe-Pt" and subtracts it during training. | High |
| **04-2** | **Active Set Selection** | Verify that `pace_activeset` reduces the dataset size while retaining diversity (e.g., selecting 100 representative structures from 1000). | Medium |
| **04-3** | **End-to-End Training** | Verify that a complete training loop (Data -> Config -> Train -> Potential) produces a valid `.yace` file. | Critical |

## 2. Behavior Definitions (Gherkin)

### Scenario 04-1: Delta Learning Configuration
```gherkin
GIVEN a configuration with "trainer.delta_learning: zbl"
AND elements "Fe", "Pt"
WHEN the Trainer prepares the "input.yaml" for Pacemaker
THEN the file should contain a "potential" section
AND the "potential" section should define a "pair_style zbl" reference
AND the ZBL parameters should match the atomic numbers of Fe (26) and Pt (78)
```

### Scenario 04-3: End-to-End Training
```gherkin
GIVEN a dataset of 50 DFT-calculated structures
WHEN the Trainer executes "train()"
THEN "pace_collect" should run successfully
AND "pace_train" should run for the specified number of epochs
AND a "potential.yace" file should be created in the output directory
```
