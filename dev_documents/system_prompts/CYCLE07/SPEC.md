# Cycle 07 Specification: Advanced Dynamics (kMC / EON)

## 1. Summary
**Goal**: Implement the `DynamicsEngine` for Adaptive Kinetic Monte Carlo (aKMC) using EON. This cycle enables the system to explore time scales far beyond MD (seconds to years) and identify rare events like diffusion and reactions.

**Key Features**:
*   **EON Integration**: Interface with `eonclient` (C++) via Python.
*   **Bridge Driver**: A Python script (`driver.py`) that EON calls to compute energy/forces using Pacemaker.
*   **Saddle Point Search**: Use Dimer/NEB methods to find transition states.
*   **On-the-Fly Learning**: Halt kMC if a saddle point has high uncertainty ($\gamma$).

## 2. System Architecture

Files to be implemented/modified are **bold**.

```ascii
src/mlip_autopipec/
├── dynamics/
│   ├── **kmc_driver.py**       # EON Interface
│   └── eon/
│       ├── **__init__.py**
│       ├── **wrapper.py**      # Config Generation
│       └── **driver.py**       # Python Hook for EON
└── tests/
    └── **test_dynamics/**
        └── **test_kmc.py**
```

## 3. Design Architecture

### 3.1. Dynamics Component (`src/mlip_autopipec/dynamics/`)

#### `kmc_driver.py`
*   **`KMCDriver`**:
    *   Generates `config.ini` for EON.
    *   Sets up the directory structure (`client/`, `server/` mocked locally).
    *   **Crucial Config**:
        ```ini
        [Potential]
        potential = script
        script_path = ./potentials/driver.py
        ```
    *   Runs `eonclient` via `subprocess`.
    *   Parses output for "Process Found" or "Halt" (exit code 100).

#### `eon/driver.py` (The Hook)
*   **Script Logic**:
    1.  Read atomic coordinates from `stdin` (EON format).
    2.  Load `potential.yace` via `pyace` or `lammps` interface.
    3.  Compute Energy, Forces, and **Max Gamma**.
    4.  If `max_gamma > threshold`:
        *   Write "bad_structure.cfg".
        *   Exit with code 100 (Halt).
    5.  Else:
        *   Print Energy and Forces to `stdout`.
        *   Exit with code 0.

### 3.2. Orchestrator Update

*   **`workflow.py`**: Add support for `run_kmc()` alongside `run_md()`.
*   **Logic**:
    *   If `MD` finds a stable state, launch `kMC` from that state to find transitions.
    *   If `kMC` halts, learn the saddle point region (high curvature).

## 4. Implementation Approach

1.  **Implement EON Wrapper**: Create `config.ini` generator.
2.  **Implement Python Hook**: Critical for connecting C++ EON to Python Pacemaker.
3.  **Implement Halt Logic**: Define specific exit codes for uncertainty.
4.  **Mock EON**: If `eonclient` is missing, simulate a process search that finds a saddle point or halts.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_kmc.py`**:
    *   **Hook Test**: Feed dummy coordinates to `driver.py` (import as module) and verify it returns energy/forces in correct format.
    *   **Halt Test**: Mock `calc.get_gamma` to return high value, assert exit code 100.

### 5.2. Integration Testing
*   **Mock Workflow**:
    *   Run `Orchestrator` configured for kMC.
    *   Simulate a halt event.
    *   Verify "bad_structure.cfg" is picked up and added to candidates.
