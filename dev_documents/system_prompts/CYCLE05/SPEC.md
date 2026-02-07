# Cycle 05 Specification: The Dynamics Engine (Inference & OTF)

## 1. Summary
This cycle implements the `Dynamics Engine`, the component responsible for running molecular dynamics (MD) simulations and performing "On-the-Fly" (OTF) learning. It uses `LAMMPS` as the backend engine, wrapped with Python logic to monitor uncertainty. The key innovation is the "Hybrid Potential" approach (overlaying ACE with ZBL/LJ for safety) and the "Watchdog" mechanism that halts simulation when the extrapolation grade ($\gamma$) exceeds a threshold, triggering the active learning loop.

## 2. System Architecture

### 2.1. File Structure
The following files must be created or modified. **Bold** files are the focus of this cycle.

src/mlip_autopipec/
├── domain_models/
│   ├── **potential.py**            # ExplorationResult
│   ├── **config.py**               # DynamicsConfig (LammpsConfig)
├── interfaces/
│   ├── **dynamics.py**             # Enhanced BaseDynamics
├── infrastructure/
│   ├── **dynamics/**
│   │   ├── **__init__.py**
│   │   ├── **lammps_interface.py** # Wrapper for LAMMPS execution
│   │   ├── **hybrid_potential.py** # Logic for pair_style hybrid/overlay
│   │   └── **watchdog.py**         # Halt trigger logic (fix halt)
└── orchestrator/
    └── **simple_orchestrator.py**  # Update logic to handle HALTED status

### 2.2. Class Diagram
*   `LammpsDynamics` implements `BaseDynamics`.
*   `Watchdog` configures the `fix halt` command in LAMMPS based on `gamma_threshold`.

## 3. Design Architecture

### 3.1. Dynamics Logic (`infrastructure/dynamics/lammps_interface.py`)
*   **Input**: `Potential`, `Start Structure`, `Temperature`, `Steps`.
*   **Process**:
    1.  **Input Generation**: Create `in.lammps` and `data.lammps`.
    2.  **Hybrid Setup**:
        *   `pair_style hybrid/overlay pace zbl 1.0 2.0`
        *   `pair_coeff * * pace potential.yace Element1 Element2`
        *   `pair_coeff * * zbl Element1 Element2`
    3.  **Watchdog Setup**:
        *   `compute gamma all pace potential.yace gamma_mode=1`
        *   `fix halt all halt 10 v_max_gamma > ${threshold} error hard`
    4.  **Execution**: Run LAMMPS via `subprocess` or `lammps` python module.
    5.  **Output Parsing**:
        *   If exit code is 0 -> `CONVERGED`.
        *   If exit code indicates Halt -> `HALTED`. Extract the frame with max gamma.
*   **Output**: `ExplorationResult` (Status, Trajectory, Final Structure).

### 3.2. Halt & Diagnose
*   When halted, the engine must return the exact structure that caused the halt.
*   This structure becomes a high-priority candidate for the Oracle.

## 4. Implementation Approach

1.  **Dependencies**: Ensure `lammps` is installed (or mocked).
2.  **Implement Hybrid Potential**: Logic to generate `pair_style` commands.
3.  **Implement Watchdog**: Logic to generate `fix halt` commands.
4.  **Implement LammpsDynamics**: The main runner class.
5.  **Update Config**: Add `DynamicsConfig` (timestep, temperature, gamma_threshold).

## 5. Test Strategy

### 5.1. Unit Testing
*   **Input Gen**: Verify `in.lammps` contains correct hybrid pair styles and fix halt commands.
*   **Parsing**: Verify log parser correctly identifies "Halt" vs "Normal" termination.

### 5.2. Integration Testing
*   **Mock MD**:
    *   Run `LammpsDynamics` with a mock script that simulates a halt (exit 1).
    *   Verify `ExplorationResult.status` is `HALTED`.
*   **Real MD (if available)**:
    *   Run a short MD with a dummy potential.
    *   Verify atoms move (trajectory file grows).
