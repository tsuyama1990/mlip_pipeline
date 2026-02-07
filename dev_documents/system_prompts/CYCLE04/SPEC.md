# Cycle 04 Specification: Dynamics Engine & Uncertainty

## 1. Summary

Cycle 04 transforms the system from a passive learner into an **Active** one. We integrate the **Dynamics Engine**, powered by **LAMMPS**, to perform Molecular Dynamics (MD) simulations.

The critical innovation here is the "Safety-First" **Hybrid Potential** and the **Uncertainty Watchdog**.
1.  **Hybrid Potential**: We will implement logic to automatically combine the learned ACE potential with a baseline ZBL/LJ potential using `pair_style hybrid/overlay`. This ensures that even if the neural network predicts physical nonsense (e.g., attractive forces at r=0), the physics-based baseline dominates and prevents the simulation from exploding.
2.  **Uncertainty Watchdog**: We utilize Pacemaker's ability to output an extrapolation grade $\gamma$ for every atom at every timestep. We will configure LAMMPS with a `fix halt` command that monitors the maximum $\gamma$ in the system. If it exceeds a safe threshold (e.g., 5.0), the simulation aborts immediately.

This "Halt" event triggers the **Active Learning Loop**: the "strange" structure that caused the halt is extracted, sent to the Oracle for labelling, and added to the training set. This allows the potential to learn exactly what it needs to know to survive the simulation (Active Learning).

## 2. System Architecture

Files to be created/modified are marked in **bold**.

```
PYACEMAKER/
├── src/
│   └── mlip_autopipec/
│       ├── **dynamics/**
│       │   ├── **__init__.py**
│       │   ├── **lammps_engine.py**        # Implements BaseDynamics
│       │   └── **input_generator.py**      # Generates in.lammps
│       ├── orchestrator/
│       │   └── **active_learner.py**       # Logic to handle Halt events
│       └── config/
│           └── **dynamics_config.py**
└── tests/
    ├── **unit/**
    │   └── **test_lammps_gen.py**
    └── **integration/**
        └── **test_active_loop_mock.py**    # Test the Halt-Retrain cycle
```

## 3. Design Architecture

### 3.1. LAMMPS Input Generator
Responsible for creating robust `in.lammps` files.
*   **Hybrid Pair Style**:
    ```
    pair_style hybrid/overlay pace zbl 1.0 2.0
    pair_coeff * * pace potential.yace Element1 Element2
    pair_coeff * * zbl 10 20
    ```
*   **Watchdog Logic**:
    ```
    compute pace all pace potential.yace ... gamma_mode=1
    variable max_gamma equal max(c_pace)
    fix watchdog all halt 10 v_max_gamma > 5.0 error hard
    ```

### 3.2. Dynamics Engine (`lammps_engine.py`)
Wraps the LAMMPS binary execution.
*   **Execution**: subprocess call to `lmp_serial` or `lmp_mpi`.
*   **Output Handling**: Parses `log.lammps` to determine if the run finished normally ("Loop time of...") or was halted ("Halted by fix...").
*   **Halt Extraction**: If halted, it reads the dump file, finds the last frame, and returns it as a `Structure` object marked with `metadata={'cause': 'high_uncertainty'}`.

## 4. Implementation Approach

1.  **Dependencies**: No new python dependencies (we invoke binary), but we assume `lammps` is installed in the environment.
2.  **Input Generator**: Implement the string template logic for `in.lammps`. Test it by generating a file and manually checking the `pair_style` lines.
3.  **Engine Logic**: Implement `run_exploration`.
    *   **Crucial**: Implement a `MockLAMMPS` mode for CI that simulates a halt by writing a dummy log file containing the "Halted" message after a random delay.
4.  **Orchestrator Update**: Update the main loop to handle the `ExplorationResult`. If `status == HALTED`, it must trigger the `Oracle` -> `Trainer` path.

## 5. Test Strategy

### 5.1. Unit Testing Approach
*   **Input Generation**: Verify that `generate_input(hybrid=True)` produces the correct `pair_style hybrid/overlay` command. Verify that element mappings (Si -> 1, O -> 2) are correct.
*   **Log Parsing**: Create sample log files (one success, one halted, one error). Verify that `parse_log(file)` correctly classifies the outcome and extracts the final timestep.

### 5.2. Integration Testing Approach
*   **Mock Active Loop**: Setup the Orchestrator with `MockLAMMPS` configured to "Halt" once. Verify:
    1.  Orchestrator calls Dynamics.
    2.  Dynamics returns "Halt".
    3.  Orchestrator calls Oracle with the "Halted" structure.
    4.  Orchestrator calls Trainer.
    5.  Orchestrator calls Dynamics again.
    6.  Dynamics returns "Success".
    This proves the closed-loop logic works.
*   **Real LAMMPS (Local)**: Run a tiny system (2 atoms). Verify that `compute pace` runs without syntax error.
