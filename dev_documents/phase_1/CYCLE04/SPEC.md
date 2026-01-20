# CYCLE04: On-The-Fly (OTF) Inference and Embedding (SPEC.md)

## 1. Summary

This document provides the detailed technical specification for Cycle 4 of the MLIP-AutoPipe project. This cycle represents a pivotal moment in the project's development, as it finally **closes the active learning loop**. While previous cycles have established the capabilities to generate data (Cycle 1 & 2) and train a model (Cycle 3), this cycle introduces the "production" component that uses the trained model to perform simulations, intelligently identifies its own weaknesses, and generates the precise data needed to remedy them. This is the mechanism that allows the system to autonomously improve.

The core deliverables for this cycle are twofold. First is the **On-The-Fly (OTF) Inference Engine**, which will be implemented as a `LammpsRunner` class. This class will be responsible for taking a trained MLIP (the `.yace` file from Cycle 3) and using it to run a full-scale molecular dynamics (MD) simulation using the LAMMPS engine. Its most critical function, however, is not just to run the simulation, but to monitor it in real-time. It will be configured to track the uncertainty of the MLIP's predictions for every single atom at every timestep, using the `extrapolation_grade` metric provided by the Pacemaker potential.

The second, and arguably most innovative, deliverable is the **Intelligent Data Extraction** module. When the `LammpsRunner` detects that an atom's uncertainty has breached a predefined threshold, it will pause the simulation and trigger this extraction logic. This is not a simple "save the whole frame" operation. Instead, it will implement the advanced **Periodic Embedding** and **Force Masking** strategies. It will extract a small, periodic sub-system centred on the uncertain atom, preserving the local bulk environment without introducing artificial surface effects. It will also generate a "force mask," a data array that distinguishes the core atoms (whose forces are reliable training targets) from the buffer atoms at the edge of the extracted box (whose forces might be affected by the artificial boundary). This `(embedded_structure, force_mask)` pair represents a highly informative and precisely curated piece of data to send back to the DFT Factory. By the end of this cycle, the system will have a complete, closed-loop mechanism for a model to explore, find its own deficiencies, and generate the exact data needed to learn and improve.

## 2. System Architecture

Cycle 4 introduces the final core module, `mlip_autopipec/modules/inference.py`, which is responsible for running production simulations and closing the active learning loop.

**File Structure for Cycle 4:**

The following ASCII tree highlights the new files for this cycle in **bold**.

```
mlip-autopipe/
├── dev_documents/
│   └── system_prompts/
│       ├── ...
│       └── CYCLE04/
│           ├── **SPEC.md**
│           └── **UAT.md**
├── mlip_autopipec/
│   ├── modules/
│   │   ├── ...
│   │   ├── training.py
│   │   └── **inference.py**    # Core LammpsRunner and embedding logic
│   └── utils/
│       └── ase_utils.py
├── tests/
│   └── modules/
│       ├── ...
│       ├── test_training.py
│       └── **test_inference.py** # Unit and integration tests for inference.py
└── pyproject.toml
```

**Component Blueprint: `modules/inference.py`**

This file will house the `LammpsRunner` class and its helper functions for embedding and masking.

-   **`LammpsRunner` class:**
    -   **`__init__(self, inference_config)`**: The constructor takes a comprehensive `InferenceConfig` Pydantic model. This config will contain everything needed for the run: the path to the LAMMPS executable, the path to the `.yace` potential, MD simulation parameters (temperature, timestep, etc.), and uncertainty detection settings.
    -   **`run(self, initial_structure: ase.Atoms) -> Optional[UncertainStructure]`**: The main public method. It takes a starting `ase.Atoms` object and runs an MD simulation. If the simulation completes without exceeding the uncertainty threshold, it returns `None`. If high uncertainty is detected, it stops and returns an `UncertainStructure` object, which contains the newly extracted, smaller atomic structure and its corresponding force mask.
    -   **`_prepare_lammps_input(self, working_dir: Path, structure_file: Path, potential_file: Path) -> Path`**: This private method generates the complex LAMMPS input script required for the OTF simulation. Critically, this script will not only define the MD run but will also include commands for the `pair_style pace` to compute the per-atom `extrapolation_grade` and dump it to an output file at regular intervals.
    -   **`_execute_lammps(self, working_dir: Path, input_script: Path) -> subprocess.CompletedProcess`**: Invokes the LAMMPS executable as a subprocess, running in the specified working directory.
    -   **`_find_first_uncertain_frame(self, working_dir: Path) -> Optional[Tuple[int, int]]`**: This method parses the output files generated by LAMMPS (the trajectory and the uncertainty data). It scans through the simulation, frame by frame, to find the *first* timestep where any atom's uncertainty value exceeds the configured threshold. It returns the timestep and the ID of the uncertain atom. If no uncertainty is found, it returns `None`.
-   **Helper Functions within `inference.py`:**
    -   **`extract_embedded_structure(large_cell: ase.Atoms, center_atom_index: int, config: UncertaintyConfig) -> UncertainStructure`**: This function contains the core scientific logic. It will be called by `LammpsRunner` when an uncertain frame is found.
        1.  It performs the **Periodic Embedding**: It calculates a new, smaller periodic cell centred on the `center_atom_index`. It then carves out all atoms within this new box, correctly wrapping their positions across the original periodic boundaries to create a new, small, and fully periodic `ase.Atoms` object.
        2.  It performs **Force Masking**: It then calculates a boolean/integer `force_mask` array for this new, smaller structure. Atoms within a "core" radius from the center are marked as `1` (to be trained on), while atoms in the outer "buffer" region are marked as `0` (to be ignored during training).
        3.  It returns a validated `UncertainStructure` Pydantic object containing both the new `ase.Atoms` object and the NumPy array for the force mask.

This design separates the concerns of process management (`LammpsRunner`) from the complex geometric/scientific logic (`extract_embedded_structure`).

## 3. Design Architecture

The design for Cycle 4 relies heavily on Pydantic to manage the complex configuration of an MD simulation and to define the structure of the data that closes the active learning loop.

**Pydantic Schema Definitions:**

The following models will be added to `mlip_autopipec/config/models.py`.

1.  **`MDConfig(BaseModel)`**: Defines the parameters for the molecular dynamics simulation.
    -   `ensemble: Literal['nvt', 'npt'] = 'nvt'`: The thermodynamic ensemble.
    -   `temperature: float = Field(300.0, gt=0)`: The simulation temperature in Kelvin.
    -   `timestep: float = Field(1.0, gt=0)`: The simulation timestep in femtoseconds.
    -   `run_duration: int = Field(1000, gt=0)`: The total number of steps in the simulation.

2.  **`UncertaintyConfig(BaseModel)`**: Defines the parameters for the active learning trigger.
    -   `threshold: float = Field(5.0, gt=0)`: The `extrapolation_grade` value that triggers an extraction.
    -   `embedding_cutoff: float = Field(8.0, gt=0)`: The radius for the periodic embedding box in Angstroms.
    -   `masking_cutoff: float = Field(5.0, gt=0)`: The inner radius for the force masking core region.
    -   `@field_validator('masking_cutoff')`: A validator to ensure `masking_cutoff` is always smaller than `embedding_cutoff`.

3.  **`InferenceConfig(BaseModel)`**: The top-level configuration for the `LammpsRunner`.
    -   `lammps_executable: FilePath`: Path to the LAMMPS binary.
    -   `potential_path: FilePath`: Path to the `.yace` potential file to be used.
    -   `md_params: MDConfig = Field(default_factory=MDConfig)`: Nested MD simulation settings.
    -   `uncertainty_params: UncertaintyConfig = Field(default_factory=UncertaintyConfig)`: Nested uncertainty and embedding settings.

4.  **`UncertainStructure(BaseModel)`**: The data transfer object that is the output of this cycle. This is what connects the inference engine back to the DFT factory.
    -   `atoms: Any`: The small, embedded `ase.Atoms` object. A validator will ensure it's a real `Atoms` object.
    -   `force_mask: Any`: The NumPy array for the force mask. A validator will ensure it's a NumPy array with the same length as the number of atoms.
    -   `metadata: Dict[str, Any] = {}`: A dictionary to store metadata, like the original timestep it was extracted from.

**Data Flow:**
-   **Input:** The `LammpsRunner` consumes a trained potential (`.yace` file) from Cycle 3 and an initial `ase.Atoms` structure.
-   **Process:** It runs an external LAMMPS simulation, which generates trajectory and uncertainty files.
-   **Output:** It parses these files and, if the uncertainty threshold is met, it produces a single `UncertainStructure` object.
-   **Next Stage:** This `UncertainStructure` object is the key. It is designed to be passed back to the start of the workflow. The `atoms` part will be sent to the `DFTFactory` (Cycle 1) for calculation. The `force_mask` will be stored alongside the DFT result in the database, to be used by the `PacemakerTrainer` (Cycle 3) to correctly weight the forces during the next training iteration. This completes the loop.

## 4. Implementation Approach

The implementation will focus on first building the core scientific logic (the embedding) and then wrapping it with the process management class.

1.  **Update Pydantic Models:** Add the `MDConfig`, `UncertaintyConfig`, `InferenceConfig`, and `UncertainStructure` models to `mlip_autopipec/config/models.py`.

2.  **Implement Periodic Embedding:** In `modules/inference.py`, implement the `extract_embedded_structure` function. This is the most complex part of the cycle.
    -   It will take the large cell, the index of the central atom, and the config.
    -   It will define a new, smaller cell matrix based on the `embedding_cutoff`.
    -   It will find all atoms from the large cell that fall within this new box, being careful to handle periodic boundary conditions correctly (e.g., using `ase.geometry.wrap_positions`).
    -   It will create a new `ase.Atoms` object with these selected atoms and the new cell.

3.  **Implement Force Masking:** The second part of `extract_embedded_structure` is to generate the mask.
    -   It will calculate the distances of all atoms in the *new, small cell* from the central point of that cell.
    -   It will create a NumPy array, setting the value to 1 for atoms whose distance is less than `masking_cutoff` and 0 otherwise.

4.  **Implement LAMMPS Input Generation:** In the `LammpsRunner` class, implement the `_prepare_lammps_input` method. This requires specific knowledge of LAMMPS syntax for the `pace` pair style. The generated script must contain:
    -   `pair_style pace`: To use the potential.
    -   `pair_coeff * * ...`: To specify the potential file.
    -   `compute uncert all pace/extrapol`: To calculate the per-atom uncertainty.
    -   `dump ...`: To write the trajectory and the per-atom uncertainty values to output files at each step.
    -   The MD run commands (`fix nvt`, `run ...`).

5.  **Implement LAMMPS Execution:** Implement `_execute_lammps` as a straightforward wrapper around `subprocess.run`.

6.  **Implement Uncertainty Parsing:** Implement the `_find_first_uncertain_frame` method. This will involve reading the LAMMPS dump file containing the uncertainty values. It will likely use a library like `pandas` or simple NumPy file I/O to load the data and efficiently find the first row where any value exceeds the threshold.

7.  **Assemble `LammpsRunner.run`:** The main `run` method will orchestrate the entire process:
    -   Create a temporary working directory.
    -   Prepare the LAMMPS input files and the initial structure file.
    -   Execute LAMMPS.
    -   Call `_find_first_uncertain_frame` to check for uncertainty.
    -   If a frame is found, read the corresponding structure from the trajectory file, and call `extract_embedded_structure` to produce and return the final `UncertainStructure` object.
    -   If no uncertainty is found, return `None`.

## 5. Test Strategy

Testing this cycle requires validating the complex geometric logic of the embedding and the interaction with the external LAMMPS process.

**Unit Testing Approach (Min 300 words):**

Unit tests in `tests/modules/test_inference.py` will meticulously validate the embedding and masking logic in isolation.

-   **Testing Periodic Embedding:** A key test, `test_embedding_wraps_atoms_correctly`, will be created. It will start with a large (e.g., 3x3x3) supercell of a crystal. The test will choose a central atom near a periodic boundary. It will then call `extract_embedded_structure`. The assertions will be critical:
    1.  The number of atoms in the returned small cell must be correct.
    2.  The test will identify a specific atom that it knows should have been "wrapped" from the other side of the large cell and assert that its position in the new cell is correct.
    3.  The new cell's lattice vectors must be correct.

-   **Testing Force Masking:** The test `test_force_mask_is_correct` will use a pre-defined `ase.Atoms` object and call the masking logic. It will then manually check the distances of a few atoms it knows are inside and outside the `masking_cutoff`. It will assert that the corresponding values in the returned `force_mask` array are 1 and 0, respectively. It will also assert that the total number of '1's in the mask is as expected.

-   **Testing LAMMPS Input Generation:** The test `test_lammps_script_generation` will not run LAMMPS. It will instantiate a `LammpsRunner` with a specific `InferenceConfig` and call `_prepare_lammps_input`. It will then read the generated script as a string and assert that it contains the correct values from the config (e.g., `fix myfix all nvt temp 300.0 300.0 0.1`, `pair_coeff * * potential.yace ...`). This confirms the configuration is being correctly translated into the LAMMPS language.

**Integration Testing Approach (Min 300 words):**

The integration test will confirm that the `LammpsRunner` can successfully launch and monitor a real LAMMPS simulation and correctly trigger the extraction process.

-   **End-to-End Uncertainty Detection and Extraction:** This will be the main integration test.
    1.  **Setup:** The test will require a simple pre-trained potential (like one from the Cycle 3 tests) and a starting structure. It will also require a working LAMMPS installation in the test environment.
    2.  **Configuration:** An `InferenceConfig` will be created. Crucially, the `uncertainty_threshold` will be set to an artificially *low* value (e.g., 0.1). This is a trick to ensure that even a stable simulation will trigger the uncertainty mechanism almost immediately, making the test fast and deterministic.
    3.  **Execution:** The test will instantiate `LammpsRunner` and call the `run()` method.
    4.  **Assertion:** This is the most important part. The test will assert that the return value is **not** `None`. It will assert that the returned value is an instance of the `UncertainStructure` Pydantic model. It will check that the `atoms` object inside the result is smaller than the initial simulation cell. Finally, it will assert that the `force_mask` array contains both 0s and 1s, proving that the core/buffer distinction was made. This single test validates the entire chain: simulation launch -> uncertainty computation -> output parsing -> triggering -> extraction -> masking -> packaging the result.
