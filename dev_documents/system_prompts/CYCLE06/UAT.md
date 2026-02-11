# Cycle 06 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 6.1: Mock Halt & Resume Loop (Full Integration)
*   **Goal**: Verify that the Orchestrator can detect a Halt, diagnose it, generate new training data, update the potential, and resume the simulation without user intervention.
*   **Action**:
    1.  User runs `pyacemaker run-loop` with `dynamics.mock.halt_prob: 1.0` and `otf.max_retries: 1`.
    2.  User observes the Orchestrator logs for the complete sequence: "Simulating...", "HALT Detected!", "Generating Candidates...", "Running DFT...", "Training...", "Resuming...".
*   **Success Criteria**:
    *   The simulation finishes successfully after at least one retraining event.
    *   A new potential file `generation_XXX.yace` is created and used in the resumed run.
    *   The final log shows "Simulation Completed".

### Scenario 6.2: Local Candidate Generation
*   **Goal**: Verify that generating local variations around a halted structure produces physically meaningful candidates.
*   **Action**:
    1.  User creates a script to test `LocalCandidateGenerator`.
    2.  Input: A single H2 molecule structure with bond length 0.5 A (too short).
    3.  User requests 10 candidates with `perturbation_scale=0.1`.
*   **Success Criteria**:
    *   The output is a list of 10 structures.
    *   The bond lengths vary around 0.5 A but are not identical.
    *   The cell vectors are preserved.

### Scenario 6.3: Max Retries Limit
*   **Goal**: Verify the system gives up gracefully if the potential cannot be improved after N retries.
*   **Action**:
    1.  User sets `otf.max_retries: 2` and forces the Mock Oracle to return high energy/forces (simulating a tough problem).
    2.  User runs the loop.
*   **Success Criteria**:
    *   The system halts 2 times and attempts retraining.
    *   After the 3rd failure (or max retries reached), it logs "Max OTF retries reached. Aborting."
    *   It saves the state and exits with a specific error code.

## 2. Behavior Definitions (Gherkin Style)

### Feature: OTF Loop Logic
**Scenario**: Detecting and handling a Halt
  **Given** a running MD simulation
  **When** the extrapolation grade exceeds the threshold
  **Then** the simulation should pause
  **And** the Orchestrator should extract the problematic structure
  **And** it should initiate the retraining workflow

### Feature: Local Candidate Generation
**Scenario**: Generating training data for a hole in the PES
  **Given** a structure where the potential failed ($S_{halt}$)
  **When** the generator is invoked
  **Then** it should return a set of perturbed structures $\{S_i\}$ around $S_{halt}$
  **And** these structures should probe the local curvature

### Feature: Resume Functionality
**Scenario**: Restarting simulation after training
  **Given** a newly trained potential file
  **And** a checkpoint from the halted simulation
  **When** the Orchestrator resumes the run
  **Then** it should use the new potential
  **And** it should continue from the last valid timestep
