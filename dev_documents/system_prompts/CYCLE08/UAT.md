# Cycle 08 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 8.1: Periodic Embedding Accuracy
**Priority**: Critical
**Description**: Verify that the "cutting" process doesn't introduce physical artifacts that corrupt the training.
**Steps**:
1.  Take a perfect Bulk Silicon supercell.
2.  Run `embed_cluster()` on one atom.
3.  Calculate DFT forces on the original supercell and the small embedded cell.
4.  **Expectation**: The forces on the *central* atoms should be nearly identical (within 0.05 eV/A).
5.  **Expectation**: The forces on the *boundary* atoms will differ (which is why we mask them).

### Scenario 8.2: aKMC Execution (Scale-Up)
**Priority**: High
**Description**: Verify that the system can interface with EON to find rare events.
**Steps**:
1.  Setup an EonExplorer with a simple adatom diffusion task.
2.  Run `explore()`.
3.  **Expectation**: EON finds a saddle point and a new minimum (the adatom jumped to the next site).
4.  **Expectation**: The system returns these new structures as candidates.

### Scenario 8.3: Phonon Stability Check
**Priority**: Medium
**Description**: Verify that the Validator catches dynamically unstable potentials.
**Steps**:
1.  Train a potential on very little data (likely unstable).
2.  Run `validator.validate_phonons()`.
3.  **Expectation**: The system reports "Imaginary Frequencies Detected" and marks the potential as `passed=False`.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Scale-Up and Advanced Validation

  Scenario: Extracting Local Clusters
    Given a large MD snapshot (10,000 atoms)
    When I request a local cluster for atom ID 500
    Then I should get a small periodic cell (approx 60 atoms)
    And the local environment of atom 500 should be preserved

  Scenario: Phonon Validation
    Given a potential that predicts negative elastic constants
    When I run the phonon validation
    Then it should detect imaginary modes
    And fail the validation step
```
