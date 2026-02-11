# Cycle 05 Specification: Dynamics Engine (LAMMPS Integration)

## 1. Summary

Cycle 05 implements the **Dynamics Engine**, which is responsible for running Classical Molecular Dynamics (MD) simulations using the potentials trained in previous cycles. This module is the "workhorse" of the system, exploring the phase space and validating the potential's stability.

The critical features introduced in this cycle are:
1.  **Hybrid Potential Application**: To ensure physical robustness, the engine automatically configures LAMMPS to use a `hybrid/overlay` pair style. This superimposes the machine learning potential (ACE) onto a physics-based baseline (ZBL/LJ), preventing unphysical atomic overlap in extrapolation regions.
2.  **Uncertainty Quantification (Watchdog)**: The engine monitors the extrapolation grade ($\gamma$) of the potential in real-time. If $\gamma$ exceeds a safety threshold, the simulation is halted immediately (`fix halt`), triggering the active learning loop.

## 2. System Architecture

We expand the `components/dynamics` module and introduce MD-specific configurations.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```
.
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **config.py**      # Add DynamicsConfig (MD parameters)
│       │   └── **results.py**     # Add DynamicsResult (Trajectory path, Halt info)
│       └── components/
│           ├── **dynamics/**
│           │   ├── **__init__.py**
│           │   ├── **base.py**        # BaseDynamics (Abstract)
│           │   ├── **lammps.py**      # LAMMPS Interface (MD)
│           │   ├── **input_gen.py**   # LAMMPS Input Script Generator
│           │   └── **parsers.py**     # Log/Dump Parsers
│           └── **factory.py**         # Update to include LAMMPSDynamics
```

### Key Components
1.  **MDInterface (`src/mlip_autopipec/components/dynamics/lammps.py`)**: The main controller. It can use the `lammps` Python module (if available) or call the binary via `subprocess`. It manages the simulation lifecycle: setup -> run -> monitor -> finalize.
2.  **InputGenerator (`src/mlip_autopipec/components/dynamics/input_gen.py`)**: Responsible for writing the `in.lammps` file. It handles the complex logic of `pair_style hybrid/overlay`, setting up the `compute pace` for uncertainty, and configuring the `fix halt` command.
3.  **Parsers (`src/mlip_autopipec/components/dynamics/parsers.py`)**: Parses LAMMPS log files to extract thermodynamic data (T, P, E) and identifies the reason for termination (Completed vs. Halted).

## 3. Design Architecture

### 3.1. Domain Models
*   **DynamicsConfig**:
    *   `timestep`: Float (fs, e.g., 1.0).
    *   `n_steps`: Int (e.g., 100,000).
    *   `temperature`: Float (K).
    *   `pressure`: Float (Bar, optional).
    *   `uncertainty_threshold`: Float (gamma value, e.g., 5.0).
    *   `hybrid_baseline`: Config for the ZBL/LJ part.
*   **DynamicsResult**:
    *   `trajectory_path`: Path to `dump.lammpstrj`.
    *   `final_structure`: ASE Atoms object.
    *   `halted`: Boolean.
    *   `halt_step`: Integer (if halted).
    *   `max_gamma`: Float (recorded maximum uncertainty).

### 3.2. Hybrid Potential Logic
The generator must produce lines like:
```lammps
pair_style hybrid/overlay pace zbl 0.5 2.0
pair_coeff * * pace potential.yace Element1 Element2
pair_coeff * * zbl 14 6
```
This ensures that the ZBL potential (which is purely repulsive at short range) is *added* to the ACE potential.

### 3.3. Uncertainty Watchdog
Using the `USER-PACE` package in LAMMPS:
```lammps
compute pace_gamma all pace potential.yace ... gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt 10 v_max_gamma > 5.0 error hard
```
The `MDInterface` must detect if LAMMPS exited with "error hard" and interpret it as a valid "Halt" event for Active Learning, not a crash.

## 4. Implementation Approach

1.  **Dependencies**: `lammps` (Python module) or a binary path.
2.  **Input Generation**: Implement `input_gen.py`. Use Jinja2 templates or string formatting to build the `in.lammps` file. Focus on the `hybrid/overlay` logic.
3.  **Execution**: Implement `lammps.py`. Use `subprocess` primarily as it's more robust to environment differences than the Python module in some CI contexts.
4.  **Parsing**: Implement log parsing to find the "Halt" message.
5.  **Integration**: Link the `DynamicsEngine` to the `Orchestrator`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Input Generation**: Verify that `generate_input_script(config)` produces a string containing the correct `pair_style` and `fix halt` commands.
*   **Parser**: Feed a dummy LAMMPS log (with and without halt) to the parser and verify the `halted` flag is correctly set.

### 5.2. Integration Testing (Mock LAMMPS)
*   **Goal**: Verify the loop handles "Halt" events correctly.
*   **Mock Implementation**:
    *   Create a `MockDynamics` class.
    *   It accepts a `halt_probability` in its config.
    *   If it decides to "halt", it returns a `DynamicsResult(halted=True, halt_step=500, max_gamma=10.0)`.
    *   If not, it returns a completed result.
*   **Orchestrator**: Verify that when `halted=True` is returned, the Orchestrator (in Cycle 06) will trigger the specific handling logic.
