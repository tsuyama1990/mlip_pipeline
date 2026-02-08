# Cycle 06: Validation & Full Orchestration

## 1. Summary

The final cycle integrates all previous components into a cohesive, autonomous system. The **Orchestrator** is upgraded from the skeleton in Cycle 01 to a fully-featured manager capable of handling the complex active learning loop (Generate -> DFT -> Train -> Dynamics -> Halt -> DFT...).

Additionally, we introduce the **Validator** component, which acts as the quality assurance gate. Before a potential is marked as "Production Ready," it must pass a battery of physical tests: Phonon stability, Elastic constants, and Equation of State (EOS) smoothness. This ensures that the MLIP not only fits the training data (low RMSE) but also respects fundamental physical laws.

Finally, we implement the **Reporting** module to generate a comprehensive HTML summary of the entire project, including learning curves, validation plots, and the final potential file.

## 2. System Architecture

Files in **bold** are new or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── **phonons.py**        # Phonon Stability Check
│   │   ├── **elastic.py**        # Elastic Constants Check
│   │   └── **eos.py**            # Equation of State Check
│   └── ...
├── core/
│   ├── **orchestrator.py**       # Final Logic
│   └── **report.py**             # HTML Report Generator
```

## 3. Design Architecture

### 3.1. Validator (`phonons.py`, `elastic.py`)
-   **Class `Validator`**:
    -   `validate(potential: Potential, structure: Structure) -> ValidationReport`
    -   **Phonons**: Use `phonopy` (if installed) or a simple finite-displacement method to compute the force constants matrix. Check for imaginary frequencies ($\omega^2 < 0$) at high-symmetry points.
    -   **Elasticity**: Deform the unit cell by $\pm 1\%$, compute stress, and fit the stiffness tensor $C_{ij}$. Check Born stability criteria.
    -   **EOS**: Fit Energy-Volume curve to Birch-Murnaghan equation.

### 3.2. Final Orchestrator Logic (`orchestrator.py`)
-   **State Machine**:
    -   **Phase 1: Cold Start**: Use `StructureGenerator` (M3GNet/Random) to create initial dataset.
    -   **Phase 2: Active Learning Loop**:
        1.  Train potential (Cycle 04).
        2.  Run Dynamics exploration (Cycle 05).
        3.  If Halt -> Embed & DFT (Cycle 03) -> Loop.
        4.  If Converged (no halts) -> Proceed to Validation.
    -   **Phase 3: Validation**: Run `Validator`.
        -   Pass -> Deploy.
        -   Fail -> Add failure configurations to dataset -> Loop.

### 3.3. Reporting (`report.py`)
-   **Class `ReportGenerator`**:
    -   Collects metrics from all cycles (RMSE, active set size, validation scores).
    -   Generates `report.html` with interactive plots (using `plotly` or static images).

## 4. Implementation Approach

1.  **Validator**: Implement `PhononCalc` and `ElasticCalc`.
    -   Use `ase.phonons` or internal finite displacement.
    -   Ensure robust error handling (e.g. if supercell is too small).
2.  **Orchestrator**: Wire everything together.
    -   Implement the "Halt & Diagnose" loop.
    -   Add checkpointing (save state after each cycle).
3.  **Reporting**: Create a simple Jinja2 template for the HTML report.
4.  **CLI**: Finalize `main.py` commands (`run`, `validate`, `report`).

## 5. Test Strategy

### 5.1. Unit Tests
-   **Validator**: Test `ElasticCalc` on a known Lennard-Jones crystal. The calculated bulk modulus should match analytical value.
-   **Orchestrator**: Verify state transitions (e.g. Halt -> Oracle).

### 5.2. Integration Tests
-   **Full Loop (Mock)**: Run the entire Fe/Pt on MgO scenario using Mocks for DFT/Train/MD but real logic for Orchestrator.
    -   Verify the loop runs for N cycles.
    -   Verify `report.html` is generated.
-   **Validation Trigger**: Manually supply a "bad" potential (random forces) to the Validator. Ensure it returns `passed=False`.

### 5.3. End-to-End System Test (The "Grand Challenge")
-   **Scenario**: Run the `tutorials/01_...ipynb` through `04_...ipynb` in sequence.
-   **Success**: The final potential can simulate the deposition process without crashing.
