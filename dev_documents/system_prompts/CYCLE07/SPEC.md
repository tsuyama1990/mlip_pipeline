# Cycle 07 Specification: Advanced Exploration (EON & Generator)

## 1. Summary

Cycle 07 expands the "Exploration" capabilities of the system. While standard MD (Cycle 04) is excellent for sampling thermal vibrations and liquids, it fails to capture "rare events" like diffusion in solids or phase transitions that occur on timescales beyond nanoseconds. To address this, we integrate **Adaptive Kinetic Monte Carlo (aKMC)** via the EON software.

Furthermore, we implement the "Structure Generator" module with advanced strategies. Instead of relying solely on dynamics to find new structures, this module proactively generates defects (vacancies, interstitials) and strained lattices. This "Rational Design" approach significantly accelerates the learning of defect energetics and elastic properties.

## 2. System Architecture

**Files to be created/modified in this cycle are marked in Bold.**

```ascii
src/mlip_autopipec/
├── inference/
│   ├── **eon/**
│   │   ├── **__init__.py**
│   │   ├── **wrapper.py**      # EON execution control
│   │   ├── **inputs.py**       # EON config generation
│   │   └── **drivers.py**      # Python driver for EON potentials
├── generator/
│   ├── **__init__.py**
│   ├── **defects.py**          # Defect generation logic
│   ├── **strain.py**           # Strain application logic
│   └── **policy.py**           # Decision logic (MD vs kMC vs Generator)
```

## 3. Design Architecture

### `EONWrapper` (in `inference/eon/wrapper.py`)
*   **Responsibility**: Manage EON client/server execution.
*   **Logic**:
    *   Generates `config.ini` for EON.
    *   Provides a `pace_driver.py` script that EON calls to calculate Energy/Force using our ACE potential.
    *   Monitors the EON process. If the driver detects high uncertainty, it returns a special exit code, causing EON to halt, similar to the LAMMPS watchdog.

### `DefectStrategy` (in `generator/defects.py`)
*   **Responsibility**: SYSTEMATICALLY generate defects.
*   **Methods**:
    *   `create_vacancies(atoms, concentration)`
    *   `create_interstitials(atoms, element)`
    *   `create_antisites(atoms)`

### `StrainStrategy` (in `generator/strain.py`)
*   **Responsibility**: Apply deformations to learn elasticity.
*   **Logic**:
    *   Apply random strain tensors $\epsilon_{ij}$ (e.g., up to 15%).
    *   Volume scaling (EOS).

### `ExplorationPolicy` (in `generator/policy.py`)
*   **Responsibility**: Decide *how* to explore in the current cycle.
*   **Logic**:
    *   If `cycle < 3`: Use `StrainStrategy` + `DefectStrategy` (Build basics).
    *   If `cycle >= 3` & `temperature < Tm`: Use `EONWrapper` (Learn diffusion).
    *   If `temperature >= Tm`: Use `LammpsRunner` (Learn liquid/melting).

## 4. Implementation Approach

1.  **Generators**: Implement `defects.py` and `strain.py` using ASE. These are deterministic and easy to test.
2.  **EON Driver**: This is the tricky part. We need a standalone script that loads the `.yace` potential and answers EON's requests via stdin/stdout. We must embed the uncertainty check here.
3.  **Policy**: Implement the simple heuristic logic.

## 5. Test Strategy

### Unit Testing
*   **Defects**: Create a 2x2x2 supercell. Request 1 vacancy. Verify atom count decreases by 1.
*   **Strain**: Apply 10% strain. Verify cell vectors change accordingly.

### Integration Testing
*   **EON Driver**:
    *   Run the `pace_driver.py` manually, piping in a structure.
    *   Verify it outputs Energy/Forces in the format EON expects.
    *   Verify it exits with code 100 (or similar) if we force high uncertainty.
