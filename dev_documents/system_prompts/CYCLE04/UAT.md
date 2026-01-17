# Cycle 04: User Acceptance Testing (UAT)

## 1. Test Scenarios

Cycle 04 is about efficiency. We test if the system can filter out the noise and select only the signal.

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-04-01** | High | MACE Screening | Verify that the system can use a surrogate model to estimate properties of generated structures and reject unphysical ones (forces > 100 eV/A). |
| **UAT-04-02** | High | Diversity Selection (FPS) | Verify that the FPS algorithm selects a diverse subset of structures from a larger pool. Visual verification via PCA plot is key here. |

### Recommended Notebooks
*   `notebooks/UAT_04_Surrogate.ipynb`:
    1.  Generate a "cluster" of similar structures and a few "outliers" (manually or via generator).
    2.  Compute descriptors (or use dummy vectors).
    3.  Run FPS.
    4.  Visualize: Plot the structures in 2D (PCA of descriptors) and highlight the selected ones. The selected ones should be spread out.

## 2. Behavior Definitions

### UAT-04-01: MACE Inference

**Narrative**:
The Generator produced 1000 structures. Some are highly compressed (unphysical) due to random strain. We don't want to waste DFT time on them. The MACE client evaluates all 1000. It flags 50 of them as having forces > 100 eV/A. These are discarded. The remaining 950 are annotated with their estimated energy.

```gherkin
Feature: Surrogate Screening

  Scenario: Evaluating Candidate Structures
    GIVEN a set of candidate structures including some with overlapping atoms
    WHEN the MACE Client processes them
    THEN each valid structure should have "mace_energy" and "mace_forces" attached
    AND structures with extreme forces (e.g., > 100 eV/A) should be flagged as invalid and removed from the list
    AND the processing time per structure should be significantly less than DFT (milliseconds vs minutes)
```

### UAT-04-02: FPS Selection

**Narrative**:
We have 950 valid candidates. 900 of them are just slight rattles of the ground state. 50 of them are high-energy liquid-like structures. We can only afford 10 DFT calculations. Random sampling might pick 10 ground-state-like structures. FPS should pick the outliers. We run FPS. The selected indices include the liquid-like structures because they are "far" in descriptor space.

```gherkin
Feature: Diversity Selection

  Scenario: Selecting Diverse Candidates
    GIVEN a pool of 100 structures containing 90 similar ones (perturbations of ground state) and 10 distinct ones (liquids/defects)
    WHEN FPS is used to select 10 structures using SOAP descriptors
    THEN the selection should include the distinct ones (outliers) as they maximize diversity
    AND the selection should not be dominated by the 90 similar structures
    AND the order of selection should be deterministic (given the same start)
```
