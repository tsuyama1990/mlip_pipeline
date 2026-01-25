# Cycle 05: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Embedding Logic Verification
**Priority**: High
**Goal**: Verify that the embedded small cell mimics the bulk environment.
**Procedure**:
1.  Create a large $10 \times 10 \times 10$ supercell of Al.
2.  Displace one atom in the center.
3.  Run `create_embedded_supercell` targeting that atom.
4.  Compare the neighbor list of the central atom in the small cell vs. the large cell.
**Success Criteria**:
*   The neighbor lists match exactly within $R_{cut}$.
*   The generated cell is significantly smaller (e.g., $3 \times 3 \times 3$).

### Scenario 2: Active Set Selection
**Priority**: Medium
**Goal**: Verify that redundant structures are filtered out.
**Procedure**:
1.  Generate 10 identical structures (with tiny noise).
2.  Run `ActiveSetSelector` asking for 2 structures.
3.  Check the output count.
**Success Criteria**:
*   Returns exactly 2 structures.
*   The selection includes the most distinct ones.

## 2. Behavior Definitions

```gherkin
Feature: Selection Module

  Scenario: Periodic Embedding
    GIVEN a large MD frame with a local defect
    WHEN the system applies Periodic Embedding
    THEN it should produce a small supercell
    AND the local environment of the defect should be preserved
    AND the new cell should be periodic (no vacuum)

  Scenario: D-Optimality Filter
    GIVEN a set of candidate structures
    WHEN the Active Set algorithm is applied
    THEN it should return a subset of structures
    AND the subset should maximize the information content (Determinant)
```
