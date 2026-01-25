# Cycle 07: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Symmetry-Aware Defect Generation
**Priority**: Medium
**Goal**: Verify that we don't generate redundant defects.
**Procedure**:
1.  Input a BCC structure (identical sites).
2.  Request single vacancies.
3.  Check output count.
**Success Criteria**:
*   Should return exactly 1 structure (since all sites are equivalent).

### Scenario 2: Adaptive Policy Triggers
**Priority**: High
**Goal**: Verify the system adapts to failure.
**Procedure**:
1.  Manually set the Validation Report to "FAIL: Elasticity".
2.  Run the Policy engine.
3.  Check the recommended next step.
**Success Criteria**:
*   Recommendation is `StrainSampling` or `DeformationScan`.

## 2. Behavior Definitions

```gherkin
Feature: Advanced Exploration

  Scenario: Generate Defects
    GIVEN a pristine crystal structure
    WHEN the Defect Generator runs
    THEN it should produce unique vacancy and interstitial configurations
    AND it should ignore symmetric duplicates

  Scenario: Policy Adaptation
    GIVEN a workflow where elastic validation failed
    WHEN the Policy Engine is queried
    THEN it should prioritize strain-based sampling in the next cycle
```
