# Cycle 08: Integration & Dashboard

## 1. Summary
This is the final cycle. We connect all the independently developed modules into a cohesive, self-driving system. We implement the **Core Active Learning Loop** within the Orchestrator, enabling fully autonomous operation. Additionally, we build the **Dashboard**, a web-based reporting tool that provides users with real-time insights into the learning progress, including parity plots, error convergence curves, and validation metrics.

## 2. System Architecture

We finalize the `orchestration` module and add the `dashboard` component.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── orchestration/
│   ├── **coordinator.py**       # Final implementation of the Main Loop
│   └── **dashboard/**
│       ├── **__init__.py**
│       ├── **generator.py**     # HTML Report Generator
│       └── **templates/**       # Jinja2 templates for the dashboard
└── tests/
    └── **test_integration.py**  # End-to-End System Tests
```

## 3. Design Architecture

### The Orchestrator Loop (`coordinator.py`)
The logic flows as follows:
```python
while cycle < max_cycles:
    # 1. Policy
    strategy = policy.decide(state)

    # 2. Exploration
    halted_runs = dynamics.run(strategy, current_potential)

    # 3. Selection
    candidates = selection.process(halted_runs)

    # 4. Oracle
    labels = oracle.compute(candidates)

    # 5. Training
    dataset = trainer.update_dataset(labels)
    new_potential = trainer.train(dataset)

    # 6. Validation
    report = validator.validate(new_potential)

    # 7. Reporting
    dashboard.update(state, report)

    # 8. Deploy or Loop
    if report.passed:
        deploy(new_potential)
        break
```

### Dashboard (`generator.py`)
*   Uses **Jinja2** to render static HTML.
*   Uses **Plotly** or **Matplotlib** (saved as base64 images) for graphs.
*   **Content**:
    *   RMSE Evolution (Energy/Force vs Cycle).
    *   Data Accumulation (Number of structures).
    *   Validation Status (Phonon Bands, EOS curves).

## 4. Implementation Approach

1.  **Orchestrator Finalization**: Replace the Cycle 01 stub with the actual loop logic invoking all sub-modules.
2.  **State Persistence**: Ensure `workflow_state.json` is saved after every step to allow crash recovery.
3.  **Dashboard**: Create a simple but informative HTML template. Implement the generator to read the state and produce `report.html`.
4.  **End-to-End Test**: Run the full "Simulation Mode" test where DFT and Training are mocked but the file flow is real.

## 5. Test Strategy

### Integration Testing
*   **`test_integration.py`**: A "Dry Run" test.
    *   Initialize system with `config.yaml`.
    *   Mock external calls (return immediate success).
    *   Run for 2 cycles.
    *   Verify `report.html` is created.
    *   Verify `potentials/` directory has versioned files.

### User Acceptance Testing
*   **Full Run**: Execute the system on a small real example (e.g., LJ Argon) if possible, or a mock equivalent, and verify the user experience from `init` to `dashboard`.
