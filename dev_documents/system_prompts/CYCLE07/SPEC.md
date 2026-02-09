# Cycle 07 Specification: Advanced Dynamics (EON/kMC)

## 1. Summary
This cycle extends the capabilities of PyAceMaker beyond standard Molecular Dynamics (MD) to handle "Rare Events" using Adaptive Kinetic Monte Carlo (aKMC). By integrating the EON software suite, the system can simulate timescales of seconds, hours, or even years, which are inaccessible to MD. A key innovation here is the development of a custom `pace_driver.py` that allows EON to call the ACE potential while simultaneously monitoring uncertainty ($\gamma$), triggering an active learning loop if a transition state (saddle point) enters an unexplored region of the Potential Energy Surface.

## 2. System Architecture

Files in **bold** are to be created or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── **eon_driver.py**       # EON Interface
│   │   ├── **pace_driver_template.py** # Bridge Script Template
│   │   └── **__init__.py**
│   └── ...
└── core/
    └── **orchestrator.py**         # Update to use EON
```

## 3. Design Architecture

### 3.1 EON Driver (`eon_driver.py`)

This class manages the configuration and execution of the `eonclient` (or `eon_server`).

**Responsibilities:**
*   **Input Generation**: Creating `config.ini` (specifying temperature, pre-exponential factor, etc.) and `reactant.con` (initial structure).
*   **Driver Script Creation**: Writing a `pace_driver.py` into the working directory. This script acts as the "Potential" for EON.
*   **Execution**: Running the `eonclient` subprocess.
*   **Halt Detection**: Checking the return code of `eonclient`. If the driver script exits with a specific code (e.g., 100), it signals a "Halt Event".

### 3.2 The Bridge Script (`pace_driver_template.py`)

This is a standalone Python script that EON calls for every energy/force evaluation.

**Logic:**
1.  **Read Structure**: EON passes coordinates via standard input or a temporary file.
2.  **Evaluate Potential**: Load `potential.yace` (using `pyacemaker.calculator`) and compute energy and forces.
3.  **Check Uncertainty**: Compute max $\gamma$.
    *   If $\gamma > threshold$: Write the structure to `halted.con` and exit with code 100.
4.  **Output**: Print energy and forces to standard output in EON's format.

### 3.3 Handover Logic

The Orchestrator can now switch between engines.
*   **MD Phase**: Run high-temp MD to explore liquid/disordered states.
*   **Quench**: Cool down to 0K.
*   **kMC Phase**: Pass the quenched structure to EON to explore long-term diffusion/ordering.

## 4. Implementation Approach

1.  **Develop Bridge Template**: Create `pace_driver_template.py`. This must be robust and self-contained (importing only necessary libraries).
2.  **Implement EON Driver**: Write `EONDynamics`. Ensure it sets up the directory structure correctly (`potentials/` folder, etc.).
3.  **Halt Logic**: Define the exit code convention (e.g., 100 = OTF Halt).
4.  **Integration**: Test `EONDynamics` with a mock potential.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Config Generation**: Verify `EONDynamics` creates a valid `config.ini`.
*   **Driver Template**:
    *   Mock the `pace_calculator`.
    *   Run the template script with dummy input.
    *   Verify it prints energy/forces in the correct format.
    *   Verify it exits with 100 if gamma is high.

### 5.2 Integration Testing
*   **EON Execution (Mock)**:
    *   Install EON (or mock its binary).
    *   Run `EONDynamics.explore()`.
    *   Trigger a halt in the driver.
    *   Verify `EONDynamics` catches the halt and returns the path to `halted.con`.

### 5.3 System Testing
*   **Reaction Path**:
    *   Set up a simple adatom diffusion case (e.g., Al on Al(100)).
    *   Run kMC.
    *   Verify it finds a saddle point and moves to a new basin.
