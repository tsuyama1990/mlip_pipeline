# Cycle 06 Specification: Advanced Dynamics & Final Integration

## 1. Summary
Cycle 06 represents the final evolution of PyAcemaker. It expands the exploration horizon beyond the nanosecond scale of MD by integrating **Adaptive Kinetic Monte Carlo (aKMC)** via the EON software. This allows the system to find rare events (diffusions, reactions) that MD misses. Additionally, we implement the full **Adaptive Exploration Policy**, which dynamically tunes sampling parameters (temperature, defect density) based on the material's feedback, completing the "Zero-Config" vision.

## 2. System Architecture

### 2.1. File Structure
```text
src/mlip_autopipec/
├── dynamics/
│   └── eon_wrapper.py              # [CREATE] EON kMC Interface
├── generator/
│   └── adaptive_policy.py          # [CREATE] Dynamic parameter tuning
└── orchestration/
    └── advanced_loop.py            # [MODIFY] Support for kMC branches
```

### 2.2. Component Interaction
- **`EONWrapper`**: Manages EON client/server. It provides a Python driver (`pace_driver.py`) to EON that calculates Energy/Forces using the ACE potential and checks $\gamma$. If $\gamma$ is high, it triggers a custom exit code to signal the Orchestrator.
- **`AdaptivePolicy`**:
    - Input: Current iteration metrics (RMSE, Halt rate, material type).
    - Output: Next iteration parameters (e.g., "Increase Max Temp to 2000K", "Switch to 50% MC moves").

## 3. Design Architecture

### 3.1. EON Integration
- **Challenge**: EON is an external C++ code.
- **Solution**: Use EON's "Potentials" interface.
    - Write a script `pace_driver.py` that EON calls.
    - This script loads `potential.yace`, calculates E/F, and prints to stdout.
    - **Crucial**: This script also checks $\gamma$. If high, it aborts the EON job locally.

### 3.2. Adaptive Policy Logic
- **Heuristics**:
    - If `Halt Rate` is low (< 1/ns) -> Aggressively increase Temperature.
    - If `Validation` fails on Elasticity -> Add more shear-strain samples.
    - If material is "Insulator" (from initial guess) -> Propose Charged Defects (if supported).

## 4. Implementation Approach

1.  **EON Driver**: Create the standalone `pace_driver.py`. Test it with manual input.
2.  **Wrapper**: Implement `run_kmc()` in `EONWrapper`. Should look similar to `LammpsRunner` but manages EON config files (`config.ini`).
3.  **Policy**: Implement a simple decision tree or Rule Engine in `AdaptivePolicy`.

## 5. Test Strategy

### 5.1. Unit Testing
- **Policy**: Feed mock history data. Verify the output parameters change logically (e.g., increasing temperature).

### 5.2. Integration Testing
- **kMC-Loop**: Run a kMC search on a vacancy system. Verify that it finds a saddle point.
- **Uncertainty Trigger in kMC**: Manually set a low gamma threshold. Verify EON stops and reports the "Transition State" candidate.

### 5.3. Final System UAT
- Run the complete "Zero-Config" workflow on a new material (e.g., TiO2) and verify it converges to a stable potential without manual intervention.
