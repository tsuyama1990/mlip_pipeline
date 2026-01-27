# Cycle 06: Advanced Dynamics (kMC) & Adaptive Policy

## 1. Summary

Cycle 06 marks the final stage of development, introducing advanced capabilities that distinguish PyAcemaker from simple MLIP fitting scripts. We implement two major features:
1.  **Kinetic Monte Carlo (kMC) Integration**: Enabling the system to simulate long-timescale phenomena (diffusion, reaction barriers) by interfacing with the EON software.
2.  **Adaptive Exploration Policy**: Replacing simple random exploration with an intelligent agent that dynamically adjusts simulation parameters (temperature, strain, defect density) based on the state of the learning process.

This cycle also includes the "Final Polish", ensuring the system is truly "Zero-Config" and robust for general users.

## 2. System Architecture

### File Structure

**mlip_autopipec/**
├── **dynamics/**
│   └── **eon/**
│       ├── **__init__.py**
│       ├── **wrapper.py**      # EONWrapper class
│       └── **client.py**       # Python client for EON
└── **generator/**
    └── **policy.py**           # AdaptivePolicy class

### Component Description

*   **`dynamics/eon/wrapper.py`**: Manages the EON simulation. Unlike LAMMPS, EON is typically a Client-Server architecture. We will implement a wrapper that runs EON in a local "Client" mode to perform Saddle Point searches (using Dimer or NEB methods) driven by the ACE potential.
*   **`generator/policy.py`**: The "Strategist". It analyzes the history of the workflow (e.g., "We have plenty of liquid data but low validation score on phonons") and dictates the next Exploration Strategy (e.g., "Run low-temperature MD with defects").

## 3. Design Architecture

### Domain Models

**`EONConfig`**
*   **Role**: kMC parameters.
*   **Fields**:
    *   `temperature`: `float`
    *   `process_search_method`: `Literal["dimer", "neb"]`
    *   `prefetch_clients`: `int`

**`PolicyState`**
*   **Role**: Internal memory for the adaptive policy.
*   **Fields**:
    *   `phase_coverage`: `Dict[str, float]` (e.g., {"liquid": 0.8, "solid": 0.2})
    *   `uncertainty_history`: `List[float]`
    *   `current_strategy`: `str`

### Key Invariants
1.  **Interface Consistency**: The `EONWrapper` must expose the same `run()` interface as `LammpsRunner` (returning a status object with `halted` flag), allowing the Orchestrator to switch between MD and kMC transparently.
2.  **Convergence**: The Adaptive Policy must eventually reduce the exploration aggressiveness (cooling schedule) to allow the potential to converge.

## 4. Implementation Approach

1.  **EON Integration**:
    *   Create a `pace_driver.py` script that EON calls to get Energy/Forces from the `potential.yace`.
    *   Implement `EONWrapper.run()`: Setup the `config.ini` for EON, place the `pace_driver.py`, and execute `eonclient`.
    *   Implement Uncertainty Halt in kMC: The `pace_driver.py` itself must calculate $\gamma$. If $\gamma > threshold$, it should return a specific exit code or error string that `EONWrapper` detects as a Halt.

2.  **Adaptive Policy Logic**:
    *   Implement `decide_next_exploration(state)`:
        *   IF `cycle < 2`: Return "Random Exploration" (High T).
        *   IF `validation_failed(phonons)`: Return "Low T MD".
        *   IF `uncertainty_high_in_expansion`: Return "High T + Volume Expansion".

3.  **Final Polish**:
    *   Review all log messages for clarity.
    *   Ensure all temporary files are cleaned up.
    *   Finalize the `README.md` and documentation.

## 5. Test Strategy

### Unit Testing
*   **Policy**: Feed various `WorkflowState` scenarios to the `AdaptivePolicy` and assert that it returns the expected `ExplorationConfig`.
*   **EON Driver**: Test `pace_driver.py` independently by piping XYZ data into stdin and checking the formatted output.

### Integration Testing
*   **kMC Loop**: Run a short EON job (finding one saddle point) using the Orchestrator. Verify that if the saddle point traverses a high-uncertainty region, the job halts and produces a candidate structure.

### System Testing
*   **Grand Challenge**: Run the full "Zero-Config" pipeline on a simple system (e.g., Copper) from scratch.
    1.  Start with no data.
    2.  System generates random structures.
    3.  System trains Gen 0.
    4.  System runs MD -> Halts.
    5.  System learns.
    6.  System runs kMC -> Halts.
    7.  System converges.
    8.  Validation passes.
