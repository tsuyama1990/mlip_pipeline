# Cycle 02 Specification: Structure Generator

## 1. Summary
This cycle focuses on the implementation of the `StructureGenerator` component, a critical part of the active learning loop. The Generator is responsible for proposing candidate atomic structures that will be evaluated by the Oracle (DFT). To maximise data efficiency, we move away from simple random sampling and implement an "Adaptive Exploration Policy". This policy intelligently selects between different generation strategies—such as random rattling, template-based substitution, or MD-driven snapshots—depending on the current state of the project (e.g., initial exploration vs. fine-tuning). We will also robustly implement the `Structure` domain model to ensure it can handle various crystallographic data formats.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the primary deliverables for this cycle.

```ascii
src/
└── mlip_autopipec/
    ├── domain_models/
    │   └── structure.py            # Enhanced Structure Model
    ├── implementations/
    │   └── **generator/**
    │       ├── **__init__.py**
    │       ├── **structure_generator.py**  # Main Class
    │       └── **policies.py**         # Adaptive Exploration Logic
    └── interfaces/
        └── generator.py            # BaseGenerator Interface
```

## 3. Design Architecture

### 3.1. Structure Generator
The `StructureGenerator` class implements the `BaseGenerator` interface. It is configured via `GeneratorConfig` which specifies the target material system (elements, composition range) and generation parameters.
-   **Capabilities**:
    -   **Random Rattling**: Takes a bulk unit cell, creates a supercell, and applies random displacements to atoms (Gaussian noise) to sample local potential energy surface curvature.
    -   **Template Mutation**: Loads an existing structure (e.g., from a file or previous iteration) and applies mutations like atom substitution (for alloys) or vacancy creation.
    -   **MD Snapshots**: (Future integration) Selects frames from MD trajectories.

### 3.2. Adaptive Exploration Policy
The `AdaptiveExplorationPolicy` class (or function) determines *which* strategy to use.
-   **Cold Start**: If no potential exists, use "Random Rattling" and "Template Mutation" to cover the basic phase space.
-   **Active Learning**: If a potential exists, use the policy to decide based on uncertainty metrics (though full integration with uncertainty comes in Cycle 05, the hooks are established here).

## 4. Implementation Approach

### Step 1: Enhanced Structure Model
Refine `src/mlip_autopipec/domain_models/structure.py` to ensure it can robustly handle:
-   Periodic Boundary Conditions (PBC).
-   Conversion to/from `ase.Atoms` objects (crucial for utilizing ASE's crystal generation tools).
-   Serialisation to JSON/YAML for logging.

### Step 2: Policy Implementation
Implement `policies.py`. Create a simple strategy:
-   `get_strategy(iteration: int) -> str`: Returns "random" for iter=0, "mutation" for iter>0, etc.

### Step 3: Generator Implementation
Implement `StructureGenerator` in `structure_generator.py`.
-   Use `ase.build` (bulk, surface) to generate initial templates.
-   Implement methods `_generate_rattled()` and `_generate_mutated()`.
-   Ensure all generated structures are wrapped in the `Structure` domain model.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Generation Validity**: Call `generate(n=10)`. Assert that 10 structures are returned. Check that they contain the correct elements and number of atoms.
-   **Sanity Checks**: Assert that atoms are not overlapping closer than a physical threshold (e.g., 0.5 Å) after rattling.
-   **Policy Logic**: Assert that `AdaptiveExplorationPolicy` returns the expected strategy for different inputs (e.g., iteration 0 vs 10).

### 5.2. Integration Testing
-   **ASE Compatibility**: Verify that generated `Structure` objects can be converted to `ase.Atoms` and back without loss of information (cell, positions, numbers).
