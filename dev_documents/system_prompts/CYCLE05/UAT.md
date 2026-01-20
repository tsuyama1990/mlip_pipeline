# Cycle 05: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 5.1: Stable MD Simulation
-   **Priority**: High
-   **Description**: Verify that the generated potential can run a stable MD simulation.
-   **Steps**:
    1.  Load the potential trained in Cycle 4.
    2.  Set up an NVT simulation at 300K for 10,000 steps.
    3.  Run `LammpsRunner`.
-   **Success Criteria**:
    -   Simulation completes without error.
    -   Temperature fluctuates around 300K.
    -   Total Energy drifts less than 1 meV/atom/ps (in NVE check).

### Scenario 5.2: Periodic Embedding Extraction
-   **Priority**: Critical
-   **Description**: Verify that we can cut a small training box out of a large simulation.
-   **Steps**:
    1.  Take a large structure (500 atoms).
    2.  Pick atom #100 as the center.
    3.  Run `extract_cluster(radius=4.0, buffer=3.0)`.
    4.  Visualize the result.
-   **Success Criteria**:
    -   Resulting structure has ~50-100 atoms.
    -   Atoms near the center have `force_mask=1`.
    -   Atoms near the edge have `force_mask=0`.
    -   The structure is a valid periodic box (no atoms overlapping when tiled).

### Scenario 5.3: Uncertainty Detection (Mock)
-   **Priority**: Medium
-   **Description**: Verify the system detects high extrapolation grades.
-   **Steps**:
    1.  Since we can't easily force high gamma without a bad potential, we mock the log file.
    2.  Create a `log.lammps` with a line `Step 500 MaxGamma 15.2`.
    3.  Run `UncertaintyChecker`.
-   **Success Criteria**:
    -   Checker identifies Step 500 as a failure point.
    -   Checker returns the snapshot index corresponding to Step 500.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Inference and Active Learning Hook

  Scenario: Running Molecular Dynamics
    Given a valid .yace potential
    And an initial structure
    When the MD engine runs for 1000 steps
    Then a trajectory file should be produced
    And a log file with thermodynamic data should be generated

  Scenario: Extracting a local environment
    Given a large atomic system
    And a target atom index
    When the embedding extractor runs with radius R and buffer B
    Then a new smaller unit cell should be created
    And atoms within R should be unmasked
    And atoms between R and R+B should be masked
```
