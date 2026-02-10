# Cycle 06 User Acceptance Test (UAT)

## 1. Test Scenarios

### **UAT-06: Self-Correction Loop (Halt & Fix)**
*   **Goal**: Demonstrate that the system can autonomously recover from a "Dynamics Halt" by generating new training data and refining the potential.
*   **Priority**: Critical (Core Value Proposition)
*   **Success Criteria**:
    *   The loop detects the halt signal.
    *   The generator creates local candidates around the problematic structure.
    *   The Oracle labels these candidates.
    *   The Trainer updates the potential.
    *   The simulation resumes and proceeds further than the initial halt point.

## 2. Behavior Definitions (Gherkin)

### Scenario: Halt Recovery
**GIVEN** a potential that fails (high uncertainty) at MD step 100
**WHEN** the Orchestrator runs the loop
**THEN** the Dynamics Engine should return `halted=True`
**AND** the Orchestrator should trigger the "Refinement Phase"
**AND** the Generator should produce 10+ local candidates based on the halted structure
**AND** the Oracle should compute energies for these candidates
**AND** the Trainer should produce a new potential version (e.g., `v2.yace`)
**AND** the Dynamics Engine should be called again with `v2.yace`
**AND** the simulation should proceed past step 100 (assuming the fix worked)

### Scenario: Convergence
**GIVEN** a potential that is stable for the full MD duration
**WHEN** the Orchestrator runs
**THEN** the loop should complete without triggering the Refinement Phase
**AND** the final state should be "Converged"
