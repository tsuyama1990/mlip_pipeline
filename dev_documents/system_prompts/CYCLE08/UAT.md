# User Acceptance Testing (UAT): Cycle 08

## 1. Test Scenarios

Cycle 08 extends the system to long timescales and finalises the product.

### Scenario 8.1: Kinetic Monte Carlo (aKMC) Run
-   **ID**: UAT-C08-01
-   **Priority**: High
-   **Description**: Run aKMC to find diffusion barriers.
-   **Success Criteria**:
    -   Orchestrator launches EON.
    -   EON finds saddle points.
    -   The system learns from these saddle points (high energy states).

### Scenario 8.2: kMC Halt and Learn
-   **ID**: UAT-C08-02
-   **Priority**: Critical
-   **Description**: The aKMC search enters a weird geometry. The driver halts it.
-   **Success Criteria**:
    -   `pace_driver.py` detects High Gamma.
    -   EON stops.
    -   Orchestrator picks up the structure.
    -   DFT runs on the saddle point.
    -   Potential is updated.
    -   EON resumes.

### Scenario 8.3: Production Release
-   **ID**: UAT-C08-03
-   **Priority**: Medium
-   **Description**: The user wants to share the potential.
-   **Success Criteria**:
    -   Command `mlip-auto deploy` is run.
    -   A zip file `Ti_O_potential_v1.0.zip` is created.
    -   It contains `potential.yace`, `metadata.json`, and `validation_report.html`.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Long-Timescale Exploration and Deployment

  Scenario: Run aKMC to find rare events
    Given a configured EON environment
    When I start the KMC phase
    Then the system should identify saddle points
    And calculate reaction rates

  Scenario: Package for distribution
    Given a validated potential
    When I run the deploy command
    Then a distributable archive should be created
    And it should contain the license information
```
