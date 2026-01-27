# Cycle 03 Specification: Dynamics Engine & Uncertainty Monitoring

## 1. Summary

Cycle 03 builds the "Dynamics Engine", the component responsible for exploring the potential energy surface. This engine uses LAMMPS to run Molecular Dynamics (MD) simulations. Crucially, it must support "On-the-Fly" (OTF) uncertainty monitoring. Using the `compute pace` command, it calculates the extrapolation grade ($\gamma$) for every atom at every step. If this value exceeds a threshold, the simulation must strictly "Halt" to prevent unphysical results and to signal the Orchestrator that new data is needed. Additionally, to ensure physical robustness, this cycle implements the "Hybrid Potential" logic, overlaying the Machine Learning potential with a physical baseline (ZBL/LJ).

## 2. System Architecture

Files to be added/modified (in bold):

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       └── **inference.py**   # MD settings (Temp, Steps, Thresholds)
├── **dynamics/**
│   ├── **__init__.py**
│   ├── **lammps.py**          # LammpsRunner class
│   └── **log_parser.py**      # Tool to parse LAMMPS logs
└── orchestration/
    └── ...
```

## 3. Design Architecture

### 3.1 Inference Configuration

**`InferenceConfig` (in `schemas/inference.py`)**
-   **Responsibilities**: Defines MD parameters.
-   **Fields**:
    -   `temperature`: float (K)
    -   `pressure`: float (Bar)
    -   `n_steps`: int
    -   `uncertainty_threshold`: float (The $\gamma$ limit, e.g., 5.0)
    -   `baseline_potential`: Enum (ZBL, LJ)

### 3.2 LAMMPS Runner

**`LammpsRunner` (in `dynamics/lammps.py`)**
-   **Responsibilities**: Manage the LAMMPS process.
-   **Key Logic**:
    -   **Input Generation**: Must dynamically generate `in.lammps`.
    -   **Hybrid Overlay**:
        ```lammps
        pair_style hybrid/overlay pace zbl 1.0 2.0
        pair_coeff * * pace potential.yace ...
        pair_coeff * * zbl ...
        ```
    -   **Watchdog**:
        ```lammps
        compute gamma all pace ...
        variable max_gamma equal max(c_gamma)
        fix watchdog all halt 10 v_max_gamma > ${threshold} error hard
        ```
-   **Output**: Returns a `SimulationResult` object indicating if the run finished or halted, and providing the path to the final structure.

### 3.3 Log Parser

**`LogParser` (in `dynamics/log_parser.py`)**
-   **Responsibilities**: Read `log.lammps` to determine *why* the simulation stopped. Differentiates between normal completion and "Halt by Watchdog".

## 4. Implementation Approach

1.  **LAMMPS Interface**:
    -   Use the `lammps` Python bindings if available (`from lammps import lammps`) for tighter control, or `subprocess` calling `lmp_serial`/`lmp_mpi`.
    -   The `subprocess` approach is often more robust against environment segfaults.

2.  **Input Script Templating**:
    -   Use `jinja2` to create flexible `in.lammps` templates.
    -   Ensure the `hybrid/overlay` logic is correctly parameterized for different element pairs.

3.  **Halt Handling**:
    -   When LAMMPS exits (likely with a specific error code due to `fix halt`), the Python wrapper must catch this.
    -   It must then locate the `dump` file and extract the last frame (the high-uncertainty structure).

## 5. Test Strategy

### 5.1 Unit Testing
-   **Template Rendering**: Verify `in.lammps` contains `pair_style hybrid/overlay` and `fix halt`.
-   **Log Parsing**: Create a dummy log file containing "Fix halt condition met". Verify `LogParser` identifies this as a Halt event.

### 5.2 Integration Testing
-   **Synthetic Halt**:
    -   Create a dummy potential (or use a blank one).
    -   Run MD on a structure that is known to be far from equilibrium.
    -   Set a very low threshold ($\gamma=0.1$).
    -   Assert that the simulation terminates early (e.g., at step 10 instead of 1000).
    -   Assert that the runner reports `halted=True`.
