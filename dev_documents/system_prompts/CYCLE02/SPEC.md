# MLIP-AutoPipe: Cycle 02 Specification

- **Cycle**: 02
- **Title**: The Seed - Physics-Informed Generator
- **Status**: Scoping

---

## 1. Summary

Cycle 02 addresses a fundamental challenge in MLIP creation: the "cold-start" problem. A robust potential requires a diverse training set, but without an existing potential, it is difficult to generate diverse structures. This cycle focuses on implementing **Module A: the Physics-Informed Generator**, a component designed to break this deadlock by creating a rich and varied initial dataset of atomic structures without relying on any prior DFT calculations or expensive AIMD simulations. The core philosophy of this module is to inject physical and chemical knowledge directly into the initial data generation process, ensuring the resulting structures cover a wide range of configurations relevant to the material's behaviour.

The implementation will focus on creating the `PhysicsInformedGenerator` class. This class will act as a factory for atomic structures, driven by the high-level goals specified in the `SystemConfig`. The generator will support multiple material types, with a primary focus in this cycle on alloys and crystalline solids. For alloys, the generator will implement a three-stage protocol: first, it will use the `icet` library to generate Special Quasirandom Structures (SQS), which are the best small-supercell approximations of a random alloy. Second, it will apply a series of volumetric and shear strains to these SQS cells to generate data that can inform the model about the material's elastic properties. Third, it will apply random atomic displacements ("rattling") of varying magnitudes to simulate thermal vibrations and explore regions of the potential energy surface near local minima.

For crystalline materials, the generator will leverage the `pymatgen` library to systematically introduce common point defects. The "One Defect" strategy will be implemented, where single vacancies, interstitials, and (for compounds) antisite defects are created within a pristine supercell. This is critical for accurately modelling defect formation energies and diffusion kinetics. By the end of this cycle, the system will be capable of taking a `UserConfig` specifying an alloy or crystal and autonomously generating a directory containing hundreds or thousands of physically meaningful structures in a standard format (like `.xyz`), ready to be consumed by the next stages of the pipeline.

---

## 2. System Architecture

The work in this cycle is highly focused, involving the creation of a single new module file that encapsulates all the generation logic. This module will depend on the configuration schemas developed in Cycle 01.

**File Structure for Cycle 02:**

The following files will be created or modified. New files are marked in **bold**.

```
.
└── src/
    └── mlip_autopipec/
        ├── config/
        │   └── system.py       # Modified to add GeneratorParams
        └── modules/
            ├── __init__.py
            ├── dft_factory.py
            └── **generator.py**    # Module A: PhysicsInformedGenerator class
```

**Component Breakdown:**

*   **`config/system.py`**: The `SystemConfig` Pydantic model will be extended with a new sub-model, `GeneratorParams`. This model will contain all the parameters necessary to control the generator's behaviour, such as the size of the SQS supercell, the range of strains to apply, the standard deviation for rattling, and the types of defects to create. This continues our schema-driven approach, ensuring that all generation parameters are validated and explicitly defined.

*   **`modules/generator.py`**: This new file will contain the `PhysicsInformedGenerator` class. This class will be the sole component responsible for all initial structure generation logic. It will be initialized with the `SystemConfig` object, from which it will read the relevant `GeneratorParams`. It will contain a main public method, `generate()`, which acts as a dispatcher, calling the appropriate private methods based on the material type specified in the configuration (e.g., `_generate_for_alloy()`, `_generate_for_crystal()`). This file will house the integration logic for the external libraries `icet` and `pymatgen`. The final output of the `generate()` method will be a list of ASE `Atoms` objects, representing the complete initial dataset.

This architecture cleanly isolates the complex logic of structure generation into a single, dedicated module, making it easy to maintain and test independently of the other system components.

---

## 3. Design Architecture

The design of the `PhysicsInformedGenerator` is that of a configurable factory. It takes a high-level specification and uses a set of internal "tools" (library integrations) to manufacture the desired output (a diverse set of atomic structures).

**Pydantic Schema Design (`system.py` extension):**

*   **`GeneratorParams`**: This new `BaseModel` will be added to the `SystemConfig`.
    *   **Sub-models**: To keep the design clean, it will contain nested models like `AlloyParams` and `CrystalParams`.
    *   **`AlloyParams`**: Will include fields like `sqs_supercell_size: List[int]`, `strain_magnitudes: List[float]`, and `rattle_std_devs: List[float]`. All fields will have sensible default values.
    *   **`CrystalParams`**: Will include a list of defect types to generate, e.g., `defect_types: List[Literal['vacancy', 'interstitial']]`.
    *   **Producers and Consumers**: The `HeuristicEngine` is the producer of these parameters. The `PhysicsInformedGenerator` is the sole consumer.

**`PhysicsInformedGenerator` Class Design (`generator.py`):**

*   **Interface**: The class will have a simple public interface: `__init__(self, config: SystemConfig)` and `generate() -> List[Atoms]`. The return of a list of `Atoms` objects makes the generator's output immediately usable by any other component in the ASE ecosystem.
*   **Internal Logic**:
    *   The `generate()` method will read the material type from the config and dispatch to the appropriate private method. This acts as a router.
    *   `_generate_for_alloy()`: This method will orchestrate the alloy generation protocol.
        1.  It will first call an internal `_create_sqs_structure()` method, which will wrap the `icet` library calls.
        2.  It will then iterate through the strain magnitudes defined in `config.generator.alloy.strain_magnitudes`. For each magnitude, it will call an `_apply_strains()` method, which creates copies of the SQS structure with various volumetric and shear strains applied.
        3.  Finally, for each strained structure, it will iterate through the rattle standard deviations and call an `_apply_rattling()` method. This chained, combinatorial approach ensures a comprehensive exploration of the local configuration space.
    *   `_generate_for_crystal()`: This method will handle defect generation. It will use `pymatgen` to create a pristine supercell, then iterate through the `defect_types` in the config, calling a dedicated method for each (e.g., `_create_vacancy()`).
*   **Immutability and statelessness**: The generator itself will be stateless. Each call to `generate()` should produce the exact same set of structures if given the same configuration. It does not modify its own state between calls. Internally, it will work with copies of the `Atoms` objects to avoid side effects. For example, `_apply_strains` will not modify the base SQS structure but will return a new list of strained structures.
*   **External Dependencies**: The class will encapsulate all interactions with `icet` and `pymatgen`. No other part of the MLIP-AutoPipe system will need to be aware of these libraries. This adheres to the principle of information hiding and makes the system more modular. If we ever wanted to replace `icet` with a different SQS generator, only this file would need to be modified.

This design results in a component that is robust, deterministic, and easy to extend with new generation methods or material types in the future.

---

## 4. Implementation Approach

The implementation will focus on building the `PhysicsInformedGenerator` class and its supporting configuration schemas methodically.

1.  **Extend Configuration (`system.py`):**
    *   First, define the `AlloyParams` and `CrystalParams` Pydantic models with their respective fields and default values.
    *   Define the main `GeneratorParams` model that includes the two sub-models.
    *   Add the `generator: GeneratorParams` attribute to the main `SystemConfig` model.

2.  **Scaffold the Generator Class (`generator.py`):**
    *   Create the new file `src/mlip_autopipec/modules/generator.py`.
    *   Define the `PhysicsInformedGenerator` class with its `__init__(self, config: SystemConfig)` method.
    *   Create the public `generate(self) -> List[Atoms]` method. Initially, it will contain placeholder logic for dispatching based on material type.
    *   Create empty private methods for the different generation steps (e.g., `_generate_for_alloy`, `_create_sqs_structure`, `_apply_strains`, `_apply_rattling`, `_generate_for_crystal`, `_create_vacancy`).

3.  **Implement Alloy Generation Logic:**
    *   Flesh out the `_create_sqs_structure` method. This will involve using `icet`'s API to build a cluster expansion and find the best SQS structure for the given composition and supercell size. The result will be converted into an ASE `Atoms` object.
    *   Implement the `_apply_strains` method. This method will take an `Atoms` object, create multiple copies, and use `atoms.set_cell` with a modified cell matrix to apply volumetric and shear deformations. It will ensure the `scale_atoms` flag is set to `True`.
    *   Implement the `_apply_rattling` method. This will take an `Atoms` object, create a copy, and add Gaussian noise to the atomic positions using `atoms.rattle()`.

4.  **Implement Crystal Defect Generation Logic:**
    *   Flesh out the `_generate_for_crystal` method. It will first create a pristine supercell using `pymatgen`.
    *   Implement the `_create_vacancy` method. It will take the supercell, remove one atom, and convert the resulting `pymatgen` structure back to an ASE `Atoms` object.
    *   Implement similar methods for other defect types as specified in the configuration.

5.  **Integrate and Finalise:**
    *   Implement the main `generate` dispatcher logic to call the correct generation workflow.
    *   The method will collect all the generated `Atoms` objects into a single list and return it.
    *   Add docstrings and type hints to all methods.
    *   Run `ruff` and `mypy` to ensure code quality and type safety.

6.  **Write Tests (`tests/modules/test_generator.py`):**
    *   Create a new test file for the generator.
    *   Write unit tests for the helper methods like `_apply_strains` and `_apply_rattling`, asserting that the cell and positions are modified as expected.
    *   Write integration tests for the main `generate` method, mocking the external libraries (`icet`, `pymatgen`) to ensure the orchestration logic is correct and that the final list of structures is assembled as expected.

---

## 5. Test Strategy

Testing the `PhysicsInformedGenerator` is crucial to ensure that the initial dataset is valid and diverse. The strategy will focus on verifying the structural properties of the output `Atoms` objects.

**Unit Testing Approach (Min 300 words):**

Unit tests will target the individual transformation methods within the generator class, ensuring each function performs its specific task correctly. These tests will be fast and will not require mocking complex external libraries.

*   **Strain Application (`tests/modules/test_generator.py`):**
    We will write a dedicated test function, `test_apply_strains`. This test will create a simple, known `Atoms` object (e.g., a 2-atom unit cell). We will then call the `_apply_strains` method with a specific strain magnitude, for example, 1.05 (a 5% volumetric expansion). The core of the test will be to assert that the cell volume of the *output* `Atoms` object is precisely `1.05**3` times the volume of the original cell. We will also test a shear strain, asserting that the cell vectors are modified correctly while the volume remains constant. This verifies that our matrix manipulations for applying strain are mathematically correct.

*   **Rattling Application (`tests/modules/test_generator.py`):**
    The `test_apply_rattling` function will verify the random displacement logic. We will start with a fixed `Atoms` object and apply rattling with a specific standard deviation and a fixed random seed to ensure reproducibility. The test will assert two things. First, it will assert that the positions of the atoms in the output object are *not* equal to the original positions, proving that some displacement has occurred. Second, it will calculate the mean displacement of all atoms and assert that it is statistically close to the target standard deviation. This confirms that the magnitude of the random noise is as expected.

*   **Configuration Logic:**
    We will also unit-test the dispatcher logic within the main `generate` method. Using a mock `SystemConfig`, we will test that if the config specifies an "alloy", the `_generate_for_alloy` method is called, and if it specifies a "crystal", `_generate_for_crystal` is called. This can be achieved with `mocker.patch.object`, asserting that the correct private methods were called once. This test doesn't check the output structures but verifies the correctness of the internal control flow.

**Integration Testing Approach (Min 300 words):**

Integration tests will verify the generator's interaction with external libraries and the correctness of the overall generated dataset. These tests may be slower and will involve mocking the expensive parts of the external library calls.

*   **SQS Generation Workflow (`tests/modules/test_generator.py`):**
    The main integration test will be `test_generate_alloy_workflow`. This test will verify the entire chain of operations for an alloy.
    1.  **Mocking `icet`**: The call to the core SQS generation function within `icet` will be mocked. This is essential because a real SQS calculation can be time-consuming. The mock will be configured to return a pre-defined, valid SQS `Atoms` object.
    2.  **Execution**: We will instantiate the `PhysicsInformedGenerator` with a `SystemConfig` that specifies a simple alloy generation task (e.g., one strain level, one rattle level). We will then call the public `generate()` method.
    3.  **Assertions**: The test will assert several properties of the returned list of `Atoms` objects. First, it will check the total number of structures generated. If we requested one strain and one rattle level, we should get `1 (original) + 1 (strained) + 1 (rattled original) + 1 (rattled strained) = 4` structures (or as per the implemented logic). Second, it will inspect one of the final structures and verify that its chemical composition matches the composition specified in the config. Finally, it will check that the `pbc` (periodic boundary conditions) flags are all set to `True` on the generated structures.

*   **Defect Generation Workflow (`tests/modules/test_generator.py`):**
    A similar test, `test_generate_crystal_workflow`, will be created for defect generation. It will mock the `pymatgen` defect creation methods. The test will provide a `SystemConfig` requesting one vacancy. After calling `generate()`, it will assert that the returned list contains one `Atoms` object and that the number of atoms in this object is exactly one less than in the pristine supercell. This confirms the defect generation and conversion back to ASE is working as intended.
