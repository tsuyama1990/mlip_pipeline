# CYCLE05: Data Modelling and Configuration (SPEC.md)

## 1. Summary

This document provides the detailed technical specification for Cycle 5 of the MLIP-AutoPipe project. The focus of this cycle is to establish the formal **data architecture** that governs the entire system. While previous cycles have developed the functional modules, this cycle defines the robust, user-friendly, and validated "control panel" through which a user interacts with the system, and the internal "single source of truth" that orchestrates the workflow. This involves creating the definitive Pydantic models for all configuration and formalising the schema for the central training database.

The first key deliverable is the design of the user-facing configuration file, `input.yaml`. The guiding principle here is **minimalism and clarity**. The user should only need to specify the "what" and "why" (e.g., "I want to study an Fe-Ni alloy for its melting properties"), not the "how." The system will be responsible for translating this high-level request into a detailed execution plan.

The second deliverable is the creation of a comprehensive, internal `SystemConfig` Pydantic model. This model will be the backbone of the application, aggregating all the detailed configurations from the previous cycles (`DFTConfig`, `ExplorerConfig`, `TrainingConfig`, `InferenceConfig`, etc.) into a single, validated object. A new `WorkflowManager` class will be introduced, whose primary initial role is to parse the simple `UserInputConfig` and expand it into the detailed `SystemConfig`, applying intelligent defaults and heuristics. This ensures that every component in the system operates from a single, consistent, and validated source of truth.

Finally, this cycle will formalise the schema of the extended ASE database. This includes specifying the exact metadata that will be stored alongside every DFT result, such as a unique ID, the configuration type (e.g., 'initial_sqs', 'active_learning_gen3'), and the force mask for embedded structures. This ensures data provenance, making the final dataset traceable, reproducible, and ready for sophisticated analysis. By the end of this cycle, the project will have a robust, schema-enforced data architecture, a clear user interface, and the foundational class for workflow orchestration.

## 2. System Architecture

The architecture for Cycle 5 is primarily focused on the `config` package and introducing the main orchestrator. It consolidates data definitions and establishes the class that will manage the entire workflow.

**File Structure for Cycle 5:**

This cycle involves consolidating models into `models.py`, creating the `WorkflowManager`, and adding tests for the new configuration logic. New or heavily modified files are in **bold**.

```
mlip-autopipe/
├── dev_documents/
│   └── system_prompts/
│       └── CYCLE05/
│           ├── **SPEC.md**
│           └── **UAT.md**
├── mlip_autopipec/
│   ├── __init__.py
│   ├── **workflow_manager.py** # Main orchestrator class
│   ├── config/
│   │   ├── __init__.py
│   │   └── **models.py**       # Consolidated Pydantic models
│   ├── modules/
│   │   └── ...
│   └── utils/
│       └── **ase_utils.py**    # Updated for new metadata schema
├── tests/
│   ├── __init__.py
│   ├── **test_config.py**      # Tests for all Pydantic models
│   └── modules/
│       └── ...
└── pyproject.toml
```

**Component Blueprint: `config/models.py`**

This file becomes the single source of truth for all data structures in the project. It will contain all the models previously defined, plus the new high-level configuration models.

-   **`UserInputConfig` (New):** The model for parsing the user's `input.yaml`.
-   **`SystemConfig` (New):** The comprehensive internal model. It will contain instances of all the module-specific configurations: `DFTConfig`, `ExplorerConfig`, `TrainingConfig`, `InferenceConfig`, etc.
-   All other models (`DFTConfig`, `UncertainStructure`, etc.) will be finalised and placed here.

**Component Blueprint: `workflow_manager.py`**

This new file introduces the central orchestrator class.

-   **`WorkflowManager` class:**
    -   **`__init__(self, user_config: UserInputConfig)`**: The constructor takes the parsed user configuration.
    -   **`self.system_config: SystemConfig`**: The manager immediately calls a private method to build the full `SystemConfig`, which is then stored as an instance attribute.
    -   **`_build_system_config(self) -> SystemConfig`**: This is the core logic of this cycle. It takes the high-level user input and populates the detailed configuration objects. For example, it will use the `elements` from the user input to populate the `species` list required by the `FingerprintConfig`. It applies sensible defaults for hundreds of potential low-level parameters.
    -   **`run(self)` (Skeleton):** A placeholder for the main workflow loop, which will be implemented in a future cycle.

**Component Blueprint: `utils/ase_utils.py`**

The database utility will be updated to handle the new, richer metadata.

-   **`save_dft_result(db_path: Path, atoms: ase.Atoms, result: DFTResult, metadata: Dict)`**: The function signature is updated to accept a metadata dictionary. This dictionary will contain keys like `config_type`, `uuid`, and optionally `force_mask`. The function will be responsible for attaching this information to the `atoms.info` or `atoms.arrays` properties before writing to the database.

## 3. Design Architecture

This cycle formalises the Schema-First design for the entire application.

**Pydantic Schema Definitions (in `config/models.py`):**

1.  **User-Facing Models:** These models define the simple, clean API for the user in `input.yaml`.
    -   **`TargetSystem(BaseModel)`**:
        -   `elements: List[str]`: List of element symbols.
        -   `composition: Dict[str, float]`: Maps elements to their fraction.
        -   `crystal_structure: str`: e.g., 'fcc', 'bcc', or a path to a CIF file.
        -   `@field_validator('elements')`: Ensures all elements are valid chemical symbols.
        -   `@model_validator(mode='after')`: A model-level validator to ensure that the keys in `composition` are consistent with `elements` and that the fractions sum to approximately 1.0.
    -   **`SimulationGoal(BaseModel)`**:
        -   `type: Literal['melt_quench', 'elastic', 'diffusion']`
        -   `temperature_range: Optional[Tuple[float, float]] = None`
    -   **`UserInputConfig(BaseModel)`**: The top-level model.
        -   `project_name: str`
        -   `target_system: TargetSystem`
        -   `simulation_goal: SimulationGoal`
        -   `model_config = ConfigDict(extra='forbid')`

2.  **Internal-Facing Models:**
    -   **`SystemConfig(BaseModel)`**: The comprehensive, internal state.
        -   `project_name: str`
        -   `run_uuid: UUID`
        -   `dft_config: DFTConfig`
        -   `explorer_config: ExplorerConfig`
        -   `training_config: TrainingConfig`
        -   `inference_config: InferenceConfig`
        -   `... and so on for all modules.`

3.  **Database Schema (Conceptual):** The `ase.db` is a key-value store, but we will enforce a strict schema on the keys we add.
    -   Standard ASE columns: `id`, `energy`, `forces`, `stress`.
    -   Custom metadata in `atoms.info` dictionary:
        -   `uuid: str`: A unique ID for this specific calculation.
        -   `config_type: str`: The provenance tag (e.g., 'sqs_rattled', 'active_learning_gen2').
    -   Custom array data in `atoms.arrays`:
        -   `force_mask: np.ndarray`: An array of 0s and 1s, present only for structures from the active learning loop.

**Data Flow:**
1.  A user creates a simple `input.yaml` file.
2.  The CLI (future cycle) will load this file and parse it into a `UserInputConfig` object.
3.  This object is passed to the `WorkflowManager`.
4.  The `WorkflowManager`'s `_build_system_config` method consumes the `UserInputConfig` and produces a single, massive `SystemConfig` object.
5.  This `SystemConfig` object is then used to instantiate all the other modules (`DFTFactory`, `SurrogateExplorer`, etc.). This ensures every component shares the exact same, validated configuration for the entire run.
6.  When the `DFTFactory` produces a result, the `WorkflowManager` will create a metadata dictionary (e.g., `{'config_type': 'dft_run', 'uuid': ...}`) and pass it along with the result to `save_dft_result`.

## 4. Implementation Approach

1.  **Consolidate Models:** Move all Pydantic models created in the `SPEC.md` files of Cycles 1-4 into the central `mlip_autopipec/config/models.py` file.
2.  **Implement User-Facing Models:** Add the `TargetSystem`, `SimulationGoal`, and `UserInputConfig` models to `models.py`. Implement their validators, especially the model-level validator for the composition check.
3.  **Implement `SystemConfig`:** Create the `SystemConfig` model that composes all the other configuration models.
4.  **Implement `WorkflowManager`:** Create the `workflow_manager.py` file and the `WorkflowManager` class. Implement the `__init__` and `_build_system_config` methods. The build method will be a large but straightforward function that creates instances of `DFTConfig`, `ExplorerConfig`, etc., filling their fields with a combination of defaults and values derived from the `UserInputConfig`.
5.  **Update Database Utilities:** Modify the `save_dft_result` function in `ase_utils.py` to accept the new `metadata` dictionary argument and correctly save the keys to the database via `atoms.info` and `atoms.arrays`.
6.  **Create Configuration Tests:** Create a new test file, `tests/test_config.py`. This file will be dedicated to testing the Pydantic models.

## 5. Test Strategy

The testing for this cycle is almost entirely focused on unit testing the validation logic of the data models.

**Unit Testing Approach (Min 300 words):**

The `tests/test_config.py` file will be crucial for ensuring the robustness of the user interface. It will contain numerous tests for the `UserInputConfig` and its sub-models.

-   **Happy Path Validation:** A test `test_valid_user_config_parses_correctly` will define a dictionary representing a perfect `input.yaml`. It will call `UserInputConfig.model_validate()` on it and assert that no exception is raised and that the resulting object has the correct values.
-   **Testing Validation Rules:** A series of tests will be written, each designed to fail a specific validation rule. `pytest.raises(ValidationError)` will be used to assert that the model correctly rejects bad input. Examples include:
    -   `test_composition_sum_not_one_raises_error`: Provide a composition `{'Fe': 0.7, 'Ni': 0.2}`.
    -   `test_composition_elements_mismatch_raises_error`: Provide `elements=['Fe', 'Ni']` but `composition={'Fe': 0.5, 'Cr': 0.5}`.
    -   `test_invalid_element_symbol_raises_error`: Provide `elements=['Fe', 'Xy']`.
    -   `test_extra_field_raises_error`: Provide a config with a typo, like `project_namee`, and assert that the `extra='forbid'` rule catches it.
-   **Testing Config Expansion:** A test for the `WorkflowManager._build_system_config` will be implemented. It will create a simple `UserInputConfig` and pass it to the `WorkflowManager`. It will then inspect the generated `manager.system_config` and assert that it has been correctly and intelligently populated. For example, it will assert that `manager.system_config.explorer_config.fingerprint.species` is equal to the `['Fe', 'Ni']` provided in the user input.

**Integration Testing Approach (Min 300 words):**

Integration testing for this cycle is lighter but still important for verifying file I/O and database interactions.

-   **File-to-Model Test:** An integration test will create a real `test_input.yaml` file on disk. The test will read the file content, parse it using a YAML library, and then validate it with the `UserInputConfig` model. This confirms that the model works with real file formats, not just Python dictionaries.
-   **Database Schema Test:** An important test, `test_save_and_load_metadata`, will validate the updated database logic.
    1.  It will create a mock `DFTResult` and a metadata dictionary containing a `uuid`, a `config_type`, and a NumPy `force_mask`.
    2.  It will call the updated `save_dft_result` to write this to a temporary database.
    3.  It will then use `ase.db.connect` to read the entry back.
    4.  The assertions will be critical: it will check that `row.get('uuid')` returns the correct string, that `row.get('config_type')` is correct, and that `row.get_atoms().arrays['force_mask']` exists and is numerically equal to the original force mask. This end-to-end check verifies that the custom metadata is being saved and retrieved without corruption.
