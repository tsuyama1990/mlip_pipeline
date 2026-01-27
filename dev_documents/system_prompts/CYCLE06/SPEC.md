# Cycle 06 Specification: Advanced Dynamics, Adaptive Policy & Final Integration

## 1. Summary

Cycle 06 represents the final evolution of PyAcemaker. It introduces two advanced capabilities: "Adaptive Exploration Policy" and "Long-Timescale Simulation" via Adaptive Kinetic Monte Carlo (aKMC). The Adaptive Policy replaces static rules with logic that dynamically decides how to explore the configuration space (e.g., "Switch to high temperature," "Run kMC," "Add vacancies") based on the material's state. The aKMC integration (using EON) allows the system to find reaction barriers and diffusion mechanisms that standard MD cannot reach. This cycle also includes the final polish of the CLI and documentation.

## 2. System Architecture

Files to be added/modified (in bold):

```ascii
mlip_autopipec/
├── **dynamics/**
│   └── **eon.py**             # EON Wrapper
├── **generator/**
│   └── **policy.py**          # Adaptive Exploration Policy
└── **scripts/**
    └── **pace_driver.py**     # Driver script for EON to call Pacemaker
```

## 3. Design Architecture

### 3.1 Adaptive Exploration Policy

**`AdaptivePolicy` (in `generator/policy.py`)**
-   **Responsibilities**:
    -   Analyze the current state of the project (e.g., "How many holes in the potential?", "Is the melting point converged?").
    -   Decide the next `ExplorationTask`.
-   **Logic**:
    -   If `uncertainty_distribution` is wide -> Run `Cautious Exploration` (Low T).
    -   If `material_type` is Metal and `diffusion` is slow -> Run `High-MC Policy` (Swap atoms).
    -   If `cycle_count` > 5 -> Trigger `EON` to find rare events.

### 3.2 EON Integration (kMC)

**`EONWrapper` (in `dynamics/eon.py`)**
-   **Responsibilities**:
    -   Configure and run the EON client.
    -   Manage the `pace_driver.py` interface.
-   **Driver Script (`pace_driver.py`)**:
    -   A standalone script called by EON.
    -   Reads coordinates from stdin.
    -   Calculates Energy/Forces/Gamma using `pacemaker`.
    -   If $\gamma > threshold$, returns a special exit code to signal "Halt".

## 4. Implementation Approach

1.  **Policy Engine**:
    -   Implement a rule-based engine first. (e.g., `if cycle < 3: return MD_RAMP; else: return KMC`).
    -   Define the `ExplorationTask` data class (params for MD/kMC).

2.  **EON Driver**:
    -   Write `pace_driver.py` which loads the `.yace` file.
    -   Ensure it conforms to EON's communication protocol (usually simple text I/O).

3.  **Final Integration**:
    -   Update `Orchestrator` to consult `AdaptivePolicy` before starting the Exploration phase.
    -   Update `Orchestrator` to handle `EON` halts (similar to LAMMPS halts).

## 5. Test Strategy

### 5.1 Unit Testing
-   **Policy Logic**: Feed fake status data to `AdaptivePolicy`. Assert it returns the expected strategy (e.g., "Metals should trigger High-MC").

### 5.2 Integration Testing
-   **kMC Interface**:
    -   Run EON with `pace_driver.py` on a simple test saddle point search.
    -   Verify that EON correctly receives forces from the driver.
-   **Full System Test**:
    -   Run the complete pipeline on a multi-element system (e.g., Ag-Pd).
    -   Verify that it switches modes (MD -> kMC) if configured to do so.
