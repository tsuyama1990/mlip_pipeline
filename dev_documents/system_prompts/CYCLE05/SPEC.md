# Cycle 05 Specification: Advanced Exploration & kMC (Adaptive Policy & EON)

## 1. Summary
Cycle 05 expands the system's capabilities beyond simple MD to include long-time-scale phenomena via Kinetic Monte Carlo (kMC) and intelligent structure generation. We introduce the `AdaptivePolicy` engine, which dynamically selects the best sampling strategy (e.g., High-T MD, Strain, Defects) based on the material type and current uncertainty. Additionally, we integrate with the EON software suite to perform saddle point searches, enabling the study of diffusion and reaction barriers.

## 2. System Architecture

### File Structure
Files to be created/modified are marked in **bold**.

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **config.py**       # Add GeneratorConfig (Policy params)
│       │   └── **policy.py**       # Policy Enum and Decision Logic
│       ├── infrastructure/
│       │   ├── dynamics/
│       │   │   ├── **eon_wrapper.py** # EON (kMC) implementation
│       │   │   └── **__init__.py**
│       │   ├── generator/
│       │   │   ├── **policy_engine.py** # Adaptive Decision Logic
│       │   │   ├── **defects.py**       # Vacancy/Interstitial generation
│       │   │   └── **strain.py**        # Elastic strain application
│       └── utils/
│           └── **eon_driver.py**   # Python script for EON to call Potentials
└── tests/
    └── integration/
        └── **test_policy_pipeline.py**
```

## 3. Design Architecture

### Domain Models (`domain_models/`)

-   **`GeneratorConfig`**:
    -   `policy_mode`: Literal["static", "adaptive"]
    -   `defect_density`: float (Default: 0.01)
    -   `strain_range`: float (Default: 0.10)
    -   `md_mc_ratio`: float (Default: 0.5)

-   **`ExplorationPolicy` (Enum)**:
    -   `HIGH_TEMPERATURE_MD`
    -   `STRAIN_SCAN`
    -   `DEFECT_SAMPLING`
    -   `KMC_SADDLE_SEARCH`

### Infrastructure (`infrastructure/`)

-   **`AdaptivePolicyEngine`**:
    -   `decide_strategy(current_uncertainty: float, material_properties: Dict) -> ExplorationPolicy`:
        -   If uncertainty is high -> return `CAUTIOUS_MD`.
        -   If material is metal -> return `HIGH_MC`.
        -   If material is stiff -> return `STRAIN_SCAN`.

-   **`EONWrapper` (implements `BaseDynamics`)**:
    -   `run_kmc(structure: Structure, potential: Path) -> ExplorationResult`:
        -   Sets up `config.ini` for EON.
        -   Executes `eonclient`.
        -   Parses `processes/` directory to find saddle points.

## 4. Implementation Approach

1.  **Policy Logic**: Implement a simple rule-based system in `policy_engine.py`. This avoids the complexity of a full RL agent for now but provides the necessary hooks.
2.  **Defect Generation**: Implement `defects.py` to randomly remove atoms (vacancies) or insert atoms (interstitials) using `ase.build`.
3.  **Strain Generation**: Implement `strain.py` to apply deformation gradient tensors to the simulation cell.
4.  **EON Integration**:
    -   Create a standalone script `utils/eon_driver.py` that EON can call. This script must load the `.yace` potential and output Energy/Forces in the format EON expects.
    -   Implement the `EONWrapper` to manage the EON client execution.

## 5. Test Strategy

### Unit Testing (`tests/unit/`)
-   **`test_policy_engine.py`**:
    -   Mock input features (e.g., "high bulk modulus").
    -   Assert that `decide_strategy` returns `STRAIN_SCAN`.
-   **`test_structure_modifiers.py`**:
    -   Test that `VacancyGenerator` actually removes an atom (count decreases by 1).
    -   Test that `StrainGenerator` changes the cell volume/shape.

### Integration Testing (`tests/integration/`)
-   **`test_kmc_pipeline.py`**:
    -   Mock `eonclient` execution (touch a file).
    -   Verify that `EONWrapper` creates the correct `config.ini`.
    -   Verify that `eon_driver.py` can be imported and runs against a dummy potential.
