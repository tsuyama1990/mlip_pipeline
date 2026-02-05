# CYCLE 05 Specification: Adaptive Structure Generation

## 1. Summary

In this cycle, we replace the `MockExplorer` (and simple MD) with an **`AdaptiveStructureGenerator`**. This module implements the "Adaptive Exploration Policy" described in the requirements. Instead of relying solely on MD, it intelligently selects between MD, Monte Carlo (MC), and Defect Engineering based on the material properties and the current state of uncertainty.

## 2. System Architecture

Files to be modified/created:

```ascii
src/mlip_autopipec/
├── structure_generation/
│   ├── **__init__.py**
│   ├── **generator.py**        # Main Entry Point
│   └── **policies.py**         # Policy Logic (Metal vs Insulator)
└── orchestration/
    └── orchestrator.py         # Integrate Generator
```

## 3. Design Architecture

### 3.1. `Policy` Protocol
*   `decide_action(context: Context) -> ExplorationAction`
*   **Context**: Contains current composition, estimated bandgap, uncertainty stats.
*   **Action**: `MDAction` (temp, pressure), `MCAction` (swap ratio), `DefectAction` (vacancy count).

### 3.2. `AdaptiveStructureGenerator` Class
*   **Responsibilities**:
    1.  **Cold Start**: If no potential exists, use `RandomGen` or external universal potentials (M3GNet) to create initial structures.
    2.  **Policy Execution**: query the active policy to determine the next move.
    3.  **Action Execution**: Call `LammpsDynamics` (for MD) or internal scripts (for defects/MC).

### 3.3. Implemented Policies
1.  **High-MC Policy** (for Alloys/Metals): High swap probability to accelerate mixing.
2.  **Defect-Driven Policy** (for Insulators): Introduce vacancies/interstitials to sample off-stoichiometry.
3.  **Strain-Heavy Policy** (for Hard Materials): Apply shear strains.

## 4. Implementation Approach

1.  **Policy Engine**: Implement the logic to switch strategies.
    *   Simple heuristic: If `bandgap > 0.1 eV` -> Defect Policy. If `elements > 1` -> MC Policy.
2.  **Generators**:
    *   Implement `DefectGenerator` (using `pymatgen` or `ase`).
    *   Implement `StrainGenerator` (applying deformation gradient).
3.  **Orchestrator**: Update the `Exploration` phase to consult the Generator before running MD.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_policy_switching.py`**:
    *   Input a "Metal" context -> Assert output is `MCAction`.
    *   Input an "Insulator" context -> Assert output is `DefectAction`.

### 5.2. Integration Testing
*   **`test_generation_variety.py`**:
    *   Run the generator for 5 cycles.
    *   Verify that the output structures include both high-temperature MD snapshots and defected supercells.
