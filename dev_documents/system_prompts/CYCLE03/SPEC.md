# MLIP-AutoPipe: Cycle 03 Specification

- **Cycle**: 03
- **Title**: The Filter - Surrogate Explorer and Selector
- **Status**: Scoping

---

## 1. Summary

Cycle 03 introduces the first layer of intelligence and cost-saving to the MLIP-AutoPipe workflow. The previous cycle, the Physics-Informed Generator, is designed to produce a vast number of candidate structures—potentially tens of thousands. Performing DFT calculations on all of them would be computationally prohibitive and inefficient, as many structures may be redundant or physically unimportant. This cycle focuses on implementing **Module B: the Surrogate Explorer and Selector**, a critical component designed to intelligently down-select this large pool of candidates to a small, information-rich subset that is truly worthy of DFT resources. This module embodies the "Surrogate-First" strategy, a core tenet of the system's design.

The implementation will be encapsulated in a new `SurrogateExplorer` class. This class will orchestrate a two-stage filtering process. First, it will perform a rapid pre-screening using a general-purpose, pre-trained MLIP, such as MACE-MP-0. This surrogate model will be used to quickly calculate the energy and forces of every candidate structure generated in Cycle 02. Any structure that is deemed unphysical—for example, by having an exceptionally high energy, indicating overlapping atoms—will be immediately discarded. This step acts as a low-cost sanity check.

The second stage is a sophisticated diversity selection process. For the structures that pass the initial screening, the module will compute a structural fingerprint or "descriptor" for each one. We will use the Smooth Overlap of Atomic Positions (SOAP) descriptor, a powerful and widely-used representation of local atomic environments. With each structure now represented as a vector in a high-dimensional descriptor space, the module will employ the **Farthest Point Sampling (FPS)** algorithm. FPS is an iterative selection process that, starting from a random point, greedily selects the structure that is farthest away in descriptor space from all previously selected structures. This process ensures that the final selection of a few hundred structures is maximally diverse, providing the most "bang for the buck" for the subsequent DFT calculations. By the end of this cycle, we will have a complete "cold-start" data preparation pipeline: Module A generates a massive candidate pool, and Module B intelligently filters it down to a small, high-value dataset ready for the DFT Factory.

---

## 2. System Architecture

This cycle's architecture introduces a new module for the exploration logic and requires extending the system configuration to manage its parameters. It directly consumes the output of Module A.

**File Structure for Cycle 03:**

The following files will be created or modified. New files are marked in **bold**.

```
.
└── src/
    └── mlip_autopipec/
        ├── config/
        │   └── system.py       # Modified to add ExplorerParams
        └── modules/
            ├── generator.py
            └── **explorer.py**     # Module B: SurrogateExplorer class
```

**Component Breakdown:**

*   **`config/system.py`**: The `SystemConfig` Pydantic model will be extended to include a new `ExplorerParams` sub-model. This configuration schema will define the parameters for Module B, such as the path to the pre-trained surrogate model file (e.g., `MACE-MP-0.model`), the energy threshold for the initial screening, the number of structures to select via FPS, and the parameters for the SOAP descriptor calculation (e.g., cutoff radius, atomic sigma).

*   **`modules/explorer.py`**: This new file will contain the `SurrogateExplorer` class. It will be initialized with the `SystemConfig`. Its main public method, `select()`, will take the list of `Atoms` objects produced by the `PhysicsInformedGenerator` as input. The class will encapsulate all the logic for the two-stage filtering process:
    1.  **Surrogate Screening:** It will use the `mace-torch` library to load the surrogate model and run predictions.
    2.  **Descriptor Calculation:** It will integrate with the `dscribe` library to compute the SOAP descriptors.
    3.  **FPS Implementation:** It will contain a direct implementation of the Farthest Point Sampling algorithm.
    The `select()` method will return a much smaller list of `Atoms` objects, representing the final, diverse selection.

This design neatly separates the concerns of generation (Module A) and selection (Module B), making the data preparation pipeline modular and clear.

---

## 3. Design Architecture

The `SurrogateExplorer` is designed as a data processing pipeline, taking a large list of data points and applying a sequence of filtering and selection transformations to it.

**Pydantic Schema Design (`system.py` extension):**

*   **`ExplorerParams`**: This new `BaseModel` within `SystemConfig` will be the single source of truth for all selection parameters.
    *   **Nested Structure**: To maintain clarity, it will contain sub-models like `SurrogateModelParams` and `FPSParams`.
    *   **`SurrogateModelParams`**: Will include `model_path: str` and `energy_threshold_ev: float`. The threshold will be used to discard any structure whose predicted energy per atom is above this value.
    *   **`FPSParams`**: Will define `n_select: int` (the final number of structures to choose) and `soap_params: SOAPParams`, a further nested model defining the hyperparameters for the SOAP descriptor (e.g., `n_max`, `l_max`, `r_cut`).
    *   **Producers and Consumers**: The `HeuristicEngine` produces these parameters, and the `SurrogateExplorer` is the sole consumer.

**`SurrogateExplorer` Class Design (`explorer.py`):**

*   **Interface**: The public API will be minimal and clear: `__init__(self, config: SystemConfig)` and `select(self, candidates: List[Atoms]) -> List[Atoms]`. This functional design (list in, list out) makes the component easy to test and reason about.
*   **Internal Logic**: The `select` method will orchestrate the filtering pipeline:
    1.  **`_screen_with_surrogate`**: A private method that takes the full list of candidates. It will initialize the MACE model (from the path in the config) and iterate through the structures, calculating the potential energy for each. It will return a new list containing only those structures whose energy per atom is below the configured threshold.
    2.  **`_calculate_descriptors`**: This method will take the screened list of structures. It will configure the SOAP descriptor generator from `dscribe` using the `soap_params` from the config. It will then compute the average SOAP descriptor for each structure, returning a 2D NumPy array where each row is the descriptor vector for a structure.
    3.  **`_farthest_point_sampling`**: This method will contain the core FPS logic. It will take the descriptor matrix and the `n_select` parameter. Its implementation will be as follows:
        *   Initialize an empty list for the selected indices and choose a random starting index.
        *   Maintain an array of the minimum distance from each point to any *already selected* point.
        *   In a loop that runs `n_select` times:
            *   Find the point with the maximum value in the minimum-distance array. This is the point "farthest" from the current selection.
            *   Add its index to the list of selected indices.
            *   Update the minimum-distance array by comparing each point's distance to the newly selected point.
        *   This method will return the list of indices of the selected structures.
    4.  The main `select` method will then use these indices to build the final list of `Atoms` objects to be returned.
*   **Efficiency**: For performance, descriptor calculation will be batched if possible. The distance calculations in FPS will leverage NumPy's vectorized operations to be efficient.

This design ensures the selection process is deterministic (with a fixed random seed for the initial point), configurable, and modular. Replacing the selection algorithm or the descriptor type would only require changing one or two private methods, without altering the class's public interface.

---

## 4. Implementation Approach

The implementation will focus on integrating the `mace-torch` and `dscribe` libraries and then implementing the FPS algorithm.

1.  **Extend Configuration (`system.py`):**
    *   Add the `mace-torch` and `dscribe` libraries to the dependencies in `pyproject.toml`.
    *   Define the `SOAPParams`, `SurrogateModelParams`, and `FPSParams` Pydantic models.
    *   Compose them into the main `ExplorerParams` model.
    *   Add the `explorer: ExplorerParams` attribute to the `SystemConfig`.

2.  **Scaffold the Explorer Class (`explorer.py`):**
    *   Create the new file `src/mlip_autopipec/modules/explorer.py`.
    *   Define the `SurrogateExplorer` class with its `__init__` and `select` methods.
    *   Create empty private methods for each stage of the pipeline: `_screen_with_surrogate`, `_calculate_descriptors`, and `_farthest_point_sampling`.

3.  **Implement Surrogate Screening:**
    *   In `_screen_with_surrogate`, write the code to load a MACE model using `mace.models.MACE.load()`.
    *   Use the ASE `mace_mp_calculator` to attach the calculator to each `Atoms` object and get the potential energy.
    *   Implement the filtering logic based on the `energy_threshold_ev` from the config.

4.  **Implement Descriptor Calculation:**
    *   In `_calculate_descriptors`, instantiate `dscribe.descriptors.SOAP` with the parameters from `config.explorer.fps.soap_params`.
    *   Use the `soap.create()` method to generate the descriptors for the list of structures. Crucially, SOAP returns a 3D array (`n_atoms`, `n_features`) for each structure, so we must average this over the atoms (axis=1) to get a single descriptor vector per structure.

5.  **Implement Farthest Point Sampling:**
    *   In `_farthest_point_sampling`, implement the algorithm as described in the Design Architecture section. Use `scipy.spatial.distance.cdist` for efficient distance matrix calculations between descriptor vectors. Ensure the initial point selection is seeded for reproducibility.

6.  **Integrate and Finalise:**
    *   Assemble the logic in the main `select` method to call the private methods in the correct sequence.
    *   Add logging statements to provide visibility into the filtering process (e.g., "Generated 10,000 candidates. After surrogate screening, 8,500 remain. Selecting 200 via FPS.").
    *   Ensure all necessary imports are added and run linters and type checkers.

7.  **Write Tests (`tests/modules/test_explorer.py`):**
    *   Create a new test file.
    *   Unit test the FPS algorithm with a simple, 2D dataset where the correct selection sequence is known by inspection.
    *   Write an integration test for the `select` method. This test will use a fixture providing a list of `Atoms` objects. It will mock the `mace` model's predictions and the `dscribe` descriptor calculation to return predefined values. The test will then assert that the final selection contains the expected number of structures and that the selection logic was applied correctly.

---

## 5. Test Strategy

Testing for Cycle 03 must verify both the filtering logic and the core selection algorithm's correctness.

**Unit Testing Approach (Min 300 words):**

The primary focus of unit testing will be the `_farthest_point_sampling` method, as it is a pure, self-contained algorithm.

*   **FPS Algorithm Correctness (`tests/modules/test_explorer.py`):**
    We will design a test case, `test_fps_selection_logic`, that is simple and visually verifiable. We will create a 2D NumPy array representing the descriptor vectors of a few points arranged in a specific geometry, for example, several points clustered together and two or three points far away from the cluster and each other. We will then call the `_farthest_point_sampling` method, asking it to select, say, 3 points. The test will fix the random seed so the starting point is always the same. By simple geometric intuition, the algorithm should select the three widely separated points. The test will assert that the indices returned by the FPS function are exactly the indices of these known "far" points. We can have another test where the points are on a simple line (e.g., at x=0, 1, 5, 10). If we ask for 3 points and start at 0, it should select 10, then 5. This test directly validates the core logic of the diversity selection mechanism without any dependencies on surrogate models or descriptors.

**Integration Testing Approach (Min 300 words):**

The integration tests will verify the orchestration of the entire selection pipeline within the `SurrogateExplorer` class, using mocks for the external libraries.

*   **Full Selection Pipeline (`tests/modules/test_explorer.py`):**
    The main integration test, `test_select_pipeline`, will verify the sequence of operations.
    1.  **Test Fixture**: We will create a `pytest` fixture that provides a list of ten simple ASE `Atoms` objects.
    2.  **Mocking**: We will use `mocker` to patch the `mace` model's `get_potential_energy` method and the `dscribe` `SOAP.create` method.
    3.  **Configuration**: We will create a `SystemConfig` for the test. The `SurrogateModelParams` will have an `energy_threshold_ev` set to a specific value. The `FPSParams` will request `n_select = 3`.
    4.  **Mock Behaviour**: We will configure the mock for the MACE model to return high energies for three of the ten atoms, ensuring they should be filtered out. For the remaining seven structures, the mock for `dscribe` will be configured to return a pre-defined set of descriptor vectors, where we know which three are the most diverse (as in the unit test).
    5.  **Execution**: We will instantiate the `SurrogateExplorer` and call its public `select` method with the list of ten `Atoms` objects.
    6.  **Assertions**: The test will make several assertions. First, it will assert that the returned list contains exactly 3 `Atoms` objects, matching `n_select`. Second, it will assert that none of the three structures that were assigned high energies are present in the final list. Finally, by checking the identity of the atoms in the final list, it will assert that the three structures selected are indeed the ones corresponding to the most diverse descriptors we pre-defined. This test comprehensively validates that both the energy screening and FPS selection stages are working together as intended.
