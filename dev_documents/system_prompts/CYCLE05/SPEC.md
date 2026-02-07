# Cycle 05 Specification: Scale-Up (kMC & Adaptive Policy)

## 1. Summary

Cycle 05 addresses the limitations of standard MD: timescale and sampling efficiency.

We introduce **Adaptive Exploration Policies**. Instead of just running random MD, the system will now decide *how* to explore based on the current knowledge state. For example, if the uncertainty is generally low, it might switch to "High Temperature MD" to find rare events. If the system is a multicomponent alloy, it might trigger "Monte Carlo Swap" steps to explore chemical ordering.

We also integrate **EON (Eon Client/Server)** for **Adaptive Kinetic Monte Carlo (aKMC)**. This allows the system to find saddle points and evolve the system over seconds or hours, capturing diffusion and phase transitions that MD would miss. The integration requires a two-way bridge: Python calling EON to start a search, and EON calling back into Python (via a driver script) to get Potentials and Forces from our ACE model.

## 2. System Architecture

Files to be created/modified are marked in **bold**.

```
PYACEMAKER/
├── src/
│   └── mlip_autopipec/
│       ├── **structure_generator/**
│       │   └── **policy_engine.py**        # Adaptive Policy Logic
│       ├── **dynamics/**
│       │   ├── **eon_client.py**           # Wrapper for EON
│       │   └── **drivers/**
│       │       └── **pace_driver.py**      # EON-compatible potential script
│       └── config/
│           └── **policy_config.py**
└── tests/
    └── **integration/**
        └── **test_eon_binding.py**
```

## 3. Design Architecture

### 3.1. Adaptive Policy Engine
A decision tree or simple rule-based system.
*   **Inputs**:
    *   Current Potential Validation Error (RMSE).
    *   History of Halts (Where did we fail last?).
    *   Material Properties (Is it a metal? An insulator?).
*   **Outputs**: A `Strategy` object (e.g., `Strategy(method="MD", temp=1000K)` or `Strategy(method="aKMC", temp=300K)`).

### 3.2. EON Integration
*   **The Driver (`pace_driver.py`)**: EON is written in C++ but calls external potentials via stdin/stdout. We must provide a script that reads coordinates and prints Energy/Forces using our `Pacemaker` potential.
    *   **Crucial**: This driver must also implement the **Watchdog**. If EON explores a high-$\gamma$ region (on a saddle point), the driver must exit with a specific code to trigger a Halt, just like LAMMPS.
*   **The Client (`eon_client.py`)**: Manages the EON simulation directory (`config.ini`, `reactant.con`), runs `eonclient`, and parses the results (`process.dat`).

## 4. Implementation Approach

1.  **Policy Engine**: Implement a class `ExplorationPolicy`. Start with a simple rule: "If Iteration < 5, use High-T MD. If Iteration >= 5, use kMC".
2.  **EON Driver**: Write `pace_driver.py`. This is a standalone script that imports `pyacemaker`.
3.  **EON Wrapper**: Implement the logic to set up an EON run.
    *   **Config**: Generate `config.ini` specifying `potential = script`.
4.  **Integration**: Update `Orchestrator` to consult the `PolicyEngine` before choosing the Dynamics method.

## 5. Test Strategy

### 5.1. Unit Testing Approach
*   **Policy Logic**: Mock the system state and assert that the Policy returns the expected strategy.
*   **Driver**: Pipe a sample structure (XYZ format) into `pace_driver.py` stdin and verify it prints the correct Energy/Forces (compared to direct calculation).

### 5.2. Integration Testing Approach
*   **Mock EON**: Create a dummy `eonclient` executable that reads `config.ini` and writes a dummy `process.dat` (simulating a found saddle point). Verify the wrapper parses it correctly.
*   **Full Cycle with kMC**: Run the orchestrator with a policy that forces kMC. Verify that the "Dynamics" step executes the EON logic and returns a new structure (the product state of the event).
