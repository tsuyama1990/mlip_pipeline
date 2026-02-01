# User Acceptance Test (UAT): Cycle 06

## 1. Test Scenarios

### Scenario 06-01: The Long Game (Priority: Low)
**Objective**: Verify that the system can perform Adaptive Kinetic Monte Carlo (aKMC) exploration.

**Description**:
The user wants to find diffusion paths that happen on the second timescale, which MD cannot see.

**User Journey**:
1.  User sets `exploration_method: akmc` in config.
2.  System starts EON.
3.  System logs "EON: Starting Process Search...".
4.  System logs "EON: Found Saddle Point (Barrier 0.5 eV)".
5.  System logs "EON: Moving to new state."
6.  The `active_learning/iter_XXX/kmc_run/` directory contains EON output files (`procdata/`, `client.log`).

**Success Criteria**:
*   EON binary is invoked successfully.
*   The system does not crash immediately.

### Scenario 06-02: The Final Product (Priority: High)
**Objective**: Verify that the system packages the final potential for distribution.

**Description**:
The project is done. The user wants to send the potential to a collaborator.

**User Journey**:
1.  The active learning loop finishes (max iterations reached).
2.  The system logs "Deploying Production Release...".
3.  A file `release_v1.0.0.zip` appears in the root.
4.  User unzips it.
5.  It contains: `potential.yace`, `manifest.json`, `report.html`, `LICENSE`.

**Success Criteria**:
*   The zip file is valid.
*   The manifest contains the correct version and author info.

## 2. Behavior Definitions (Gherkin)

### Feature: Advanced Exploration & Deployment

```gherkin
Feature: EON Integration

  Scenario: Run aKMC Search
    GIVEN a trained potential
    AND an EON configuration
    WHEN the Orchestrator executes the aKMC phase
    THEN it should generate the "config.ini" for EON
    AND it should execute the EON client
    AND it should use the "pace_driver.py" for force calculations

  Scenario: Create Production Package
    GIVEN a completed workflow
    WHEN the ProductionDeployer runs
    THEN it should create a ZIP archive
    AND the archive should contain the final potential file
    AND the archive should contain a "manifest.json" with metadata
```
