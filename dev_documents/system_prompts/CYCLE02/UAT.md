# Cycle 02 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 2.1: Random Structure Generation (Baseline)
*   **Goal**: Verify that the system can generate a set of random atomic structures (perturbed positions and strained cells) without external tools.
*   **Action**:
    1.  User updates `config.yaml` to set `generator.type: random` and `generator.n_structures: 10`.
    2.  User runs `pyacemaker run-loop`.
    3.  User inspects the output.
*   **Success Criteria**:
    *   The `logs/` directory contains entries for "Generated 10 random structures".
    *   The orchestrator passes these structures to the Oracle (Mock Oracle in this cycle).
    *   The `workflow_state.json` reflects the structure count.

### Scenario 2.2: Cold Start with M3GNet (Advanced Exploration)
*   **Goal**: Verify that the system can leverage a pre-trained universal potential (M3GNet) to generate physically plausible initial structures.
*   **Action**:
    1.  User ensures `matgl` is installed (or uses the mock environment).
    2.  User updates `config.yaml` to set `generator.type: m3gnet`.
    3.  User runs `pyacemaker run-loop`.
*   **Success Criteria**:
    *   The logs show "Initializing M3GNet model...".
    *   The generated structures have significantly lower energy/forces (when evaluated by M3GNet) compared to random noise.
    *   The system does not crash if `matgl` is missing (it should fallback or error gracefully).

### Scenario 2.3: Adaptive Policy Switching
*   **Goal**: Verify that the `AdaptiveGenerator` correctly selects strategies based on the configuration.
*   **Action**:
    1.  User sets `generator.type: adaptive` and `generator.policy.initial_exploration_ratio: 0.5`.
    2.  User runs `pyacemaker run-loop`.
*   **Success Criteria**:
    *   The logs show a mix of "Using Random Strategy" and "Using M3GNet Strategy".
    *   The total number of generated structures matches the requested count.

## 2. Behavior Definitions (Gherkin Style)

### Feature: Random Structure Generation
**Scenario**: Generating perturbed structures
  **Given** a `config.yaml` with `generator.type: random`
  **When** the user runs the orchestrator
  **Then** the generator should produce structures with random atomic displacements
  **And** the cell vectors should be strained within the specified range

### Feature: M3GNet Integration
**Scenario**: Generating relaxed structures with M3GNet
  **Given** `matgl` is installed
  **And** a `config.yaml` with `generator.type: m3gnet`
  **When** the user runs the orchestrator
  **Then** the generator should load the pre-trained model
  **And** the generated structures should be relaxed (local minima)

### Feature: Adaptive Strategy
**Scenario**: Switching between strategies
  **Given** a `config.yaml` with `generator.type: adaptive`
  **When** the generator is called with `n=100` and `ratio=0.8`
  **Then** approximately 80 structures should come from the primary strategy
  **And** approximately 20 structures should come from the secondary strategy
