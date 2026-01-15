# MLIP-AutoPipe: Cycle 05 Specification

- **Cycle**: 05
- **Title**: The Intelligence - Full Active Learning Loop
- **Status**: Scoping

---

## 1. Summary

Cycle 05 is the "Aha!" moment of the MLIP-AutoPipe project. It takes all the individual, powerful components built in the preceding cycles and wires them together to create a fully functional, end-to-end active learning engine. The primary objective is to implement the intelligent feedback loop that allows the system to autonomously improve its own MLIP. This cycle will deliver a working prototype of the "Zero-Human" protocol, where the system can start with a rudimentary potential, use it to explore the configuration space, identify its own weaknesses, and trigger the generation of new DFT data to remedy those weaknesses, all without human intervention.

The implementation will focus on two key areas. First is the creation of **Module E: the Inference Engine**. This will be a new class, likely named `LammpsRunner`, responsible for executing large-scale Molecular Dynamics (MD) simulations using the latest MLIP trained in Cycle 04. The crucial feature of this module is its ability to perform on-the-fly uncertainty quantification. During the MD simulation, for every step, it will use Pacemaker's built-in "extrapolation grade" metric to assess how confident the model is in its own predictions.

Second, and most importantly, this cycle will implement the main application logic, likely in `app.py`, that orchestrates the entire active learning loop. This central controller will initiate an MD run using the `LammpsRunner`. When the runner detects that the extrapolation grade has exceeded a pre-defined threshold—a sign that the simulation has entered an unknown region of the potential energy surface—it will pause the simulation. The main application will then take the high-uncertainty structure, add it to the DFT processing queue, and trigger the DFT Factory (Cycle 01) and Pacemaker Trainer (Cycle 04) to refine the potential. Once the new, improved potential is ready, the application will seamlessly resume the MD simulation. By the end of this cycle, we will have a demonstrable, self-improving system that truly learns on the fly.

---

## 2. System Architecture

This cycle integrates all previously built modules into a cohesive workflow, orchestrated by a new main application entry point.

**File Structure for Cycle 05:**

The following files will be created or modified. New files are marked in **bold**.

```
.
└── src/
    └── mlip_autopipec/
        ├── **app.py**              # Main CLI entry point using Typer
        ├── config/
        │   └── system.py       # Modified to add InferenceParams
        └── modules/
            ├── dft_factory.py
            ├── trainer.py
            └── **inference.py**    # Module E: LammpsRunner and uncertainty logic
```

**Component Breakdown:**

*   **`config/system.py`**: The `SystemConfig` will be extended with `InferenceParams`. This sub-model will contain parameters for the MD simulation, such as the simulation timestep, temperature, pressure, and the total number of steps. Crucially, it will also define `uncertainty_threshold`, the extrapolation grade value that triggers the active learning feedback loop.

*   **`modules/inference.py`**: This new file will contain the `LammpsRunner` class. It will be initialized with the `SystemConfig`. Its primary method, `run()`, will be designed as a Python generator (`yield`). It will execute the LAMMPS simulation step-by-step. In each step, it will calculate the extrapolation grade. If the grade is below the threshold, it will `yield` a status update (e.g., `'stable'`). If the grade exceeds the threshold, it will `yield` the high-uncertainty `Atoms` object and pause its execution until the generator is iterated again. This generator-based design provides a clean and powerful way to pause and resume the simulation.

*   **`app.py`**: This new file will be the main entry point for the entire application, built using the `typer` CLI library. It will contain the main orchestration logic. The `run` command will:
    1.  Load the `UserConfig` from a YAML file and expand it into a `SystemConfig`.
    2.  Instantiate all the necessary modules (`DatabaseManager`, `QEProcessRunner`, `PacemakerTrainer`, `LammpsRunner`).
    3.  Enter the main active learning loop:
        *   Call `trainer.train()` to get the latest potential.
        *   Instantiate `LammpsRunner` and start iterating through its `run()` generator.
        *   If the runner yields a high-uncertainty structure, the loop will:
            *   Send the structure to a (conceptual) DFT queue.
            *   Call `dft_runner.run()` on it.
            *   Write the result to the database.
            *   Break the inner MD loop and restart the main loop, which will trigger a new training run.
        *   If the MD simulation completes without exceeding the uncertainty threshold, the application finishes.

This architecture creates a clear separation between the simulation engine (`LammpsRunner`) and the high-level orchestration logic (`app.py`).

---

## 3. Design Architecture

The design of Cycle 05 focuses on managing the state and flow of the active learning loop. The `LammpsRunner` is designed as a pausable state machine, and `app.py` is the state machine's controller.

**Pydantic Schema Design (`system.py` extension):**

*   **`InferenceParams`**: This new `BaseModel` will govern the simulation.
    *   **`MDEnsemble`**: A nested model to define the thermostat and barostat settings (e.g., `ensemble_type: Literal['nvt', 'npt']`, `target_temperature_k: float`).
    *   **`uncertainty_threshold: float`**: A critical parameter, with a sensible default like `4.0`, which balances exploration against stability.
    *   **Producers and Consumers**: Produced by the `HeuristicEngine`, consumed by the `LammpsRunner` and `app.py`.

**`LammpsRunner` Class Design (`inference.py`):**

*   **Interface**: `__init__(self, config: SystemConfig, potential_path: str)` and `run() -> Generator[Union[str, Atoms], None, None]`. The use of a Python generator is the key design choice.
*   **Internal Logic**: The `run` method will be implemented as a loop over the number of MD steps.
    1.  **Setup**: The method will first programmatically generate a LAMMPS input script based on the parameters in `self.config.inference`. This script will load the potential specified by `potential_path`.
    2.  **Execution**: It will use the `lammps` Python library to run the simulation one step at a time (`run 1`).
    3.  **Uncertainty Check**: After each step, it will use the `pacemaker` library's Python API to calculate the `extrapolation_grade` for the current atomic configuration.
    4.  **Yielding**:
        *   If `extrapolation_grade < threshold`, it will `yield "stable"`.
        *   If `extrapolation_grade >= threshold`, it will `yield current_atoms_object` and then effectively pause. The state (current positions, velocities, etc.) is maintained within the running LAMMPS instance. When the `for` loop in `app.py` calls `next()` on the generator again, it will resume from this point.
*   **State Management**: The state of the simulation is managed internally by the LAMMPS engine, which is controlled by the `LammpsRunner`. The runner itself is stateless between calls to `run`.

**`app.py` Orchestration Logic:**

*   **CLI Interface**: A `typer` application will be created. A main command, `run`, will take the path to the user's `input.yaml` file as an argument.
*   **Main Loop**: The core of the application will be a `while` loop that continues as long as the simulation has not completed its total number of steps.
    *   **Train**: At the top of the loop, `trainer.train()` is called. This ensures the simulation always uses the very latest potential.
    *   **Simulate**: A `for` loop iterates over the `lammps_runner.run()` generator.
    *   **Handle Yield**:
        *   If the yielded value is the string `"stable"`, the loop continues to the next MD step.
        *   If the yielded value is an `Atoms` object, it means uncertainty is high. The application will:
            1.  Log this event clearly.
            2.  Call the `dft_runner.run()` with the `Atoms` object.
            3.  Call the `db_manager.write_calculation()` to save the result.
            4.  Use `break` to exit the inner `for` loop. This will cause the outer `while` loop to start its next iteration, which immediately calls `trainer.train()`, thus incorporating the new data.

This design creates a robust, self-correcting loop. The system is designed to be interrupted (by high uncertainty) and to resume gracefully with a better model.

---

## 4. Implementation Approach

The implementation will focus on integrating the LAMMPS and Pacemaker Python APIs and building the main application logic.

1.  **Dependencies and Configuration:**
    *   Add `lammps` and `typer` to the project dependencies in `pyproject.toml`.
    *   Implement the `InferenceParams` and its nested models in `src/mlip_autopipec/config/system.py`.

2.  **Implement the `LammpsRunner` (`inference.py`):**
    *   Create the class structure as designed.
    *   Implement the LAMMPS input script generation. This will involve creating a string with commands like `pair_style pace`, `pair_coeff * * ...`, `fix nvt`, etc., populated with parameters from the config.
    *   In the `run` method's loop, use the `lammps` library's Python interface to execute single MD steps.
    *   After each step, get the current atomic positions and use the `pacemaker` library to calculate the extrapolation grade.
    *   Implement the `yield` logic based on the comparison with the threshold.

3.  **Implement the CLI Application (`app.py`):**
    *   Create a new `typer.Typer()` application object.
    *   Define the `run(config_path: str)` command.
    *   Inside `run`, write the logic for loading and expanding the configuration.
    *   Instantiate all the module classes (`DatabaseManager`, `QEProcessRunner`, `PacemakerTrainer`).
    *   Implement the main `while` loop as described in the Design Architecture.
    *   Within the loop, instantiate the `LammpsRunner` with the latest potential file path.
    *   Implement the inner `for` loop to iterate over the runner and handle the yielded values.
    *   Add copious logging to provide a clear, step-by-step narrative of the active learning process in the console output.

4.  **Testing (`tests/modules/test_inference.py`, `tests/test_app.py`):**
    *   Write unit tests for the `LammpsRunner`. The `run` method is a generator, so the test will call `next()` on it multiple times. We will mock the `pacemaker` uncertainty calculation. The test will configure the mock to return a low uncertainty for the first few calls, and then a high uncertainty. We will assert that the `yield` values are correct (strings first, then an `Atoms` object).
    *   Write an integration test for the `app.py` CLI. This will be a high-level test that mocks all the modules (`trainer`, `runner`, `dft_factory`). The test will use `typer.testing.CliRunner` to invoke the `run` command. The test will assert that the modules are called in the correct sequence when a high-uncertainty structure is "found". For example, it should assert that `trainer.train` is called, then `runner.run`, then `dft_runner.run`, then `db_manager.write`, and then `trainer.train` is called again.

---

## 5. Test Strategy

Testing for Cycle 05 is complex because it involves the interaction of all system components. The strategy will focus on testing the orchestration logic and the behaviour of the `LammpsRunner` generator.

**Unit Testing Approach (Min 300 words):**

The unit tests will focus on the `LammpsRunner`'s generator interface, ensuring it behaves correctly as a pausable state machine.

*   **`LammpsRunner` Generator Behavior (`tests/modules/test_inference.py`):**
    The test function `test_runner_yields_on_uncertainty` will verify the core behaviour of the inference module.
    1.  **Mocks**: We will heavily mock the internals of the `LammpsRunner`. The `lammps` library itself will be mocked to avoid any real simulation. Most importantly, the call to the `pacemaker` library for calculating the extrapolation grade will be mocked.
    2.  **Mock Configuration**: We will configure the uncertainty mock to return a sequence of values when called repeatedly. For example, `[1.0, 1.5, 5.0, 1.2]`. This simulates a simulation that is stable for two steps, becomes unstable on the third, and would become stable again.
    3.  **Test Logic**: The test will instantiate the `LammpsRunner` and get the generator object by calling `runner.run()`. It will then call `next()` on the generator three times.
    4.  **Assertions**: We will assert that the first call to `next()` returns the string `'stable'`. We will assert that the second call also returns `'stable'`. We will assert that the third call returns an ASE `Atoms` object, because the mock's return value (`5.0`) exceeded the default threshold. This test isolates and confirms the correctness of the crucial yield-on-uncertainty mechanism, which is the foundation of the entire active learning loop.

**Integration Testing Approach (Min 300 words):**

The integration tests will target the main application in `app.py`, verifying that it correctly orchestrates the interactions between all the (mocked) modules.

*   **Full Active Learning Loop Orchestration (`tests/test_app.py`):**
    The test `test_active_learning_loop_logic` will be the most comprehensive test in the project so far. It will use `typer.testing.CliRunner` to invoke the application from a test function.
    1.  **Mock All Modules**: We will mock the classes for every module: `DatabaseManager`, `QEProcessRunner`, `PacemakerTrainer`, and `LammpsRunner`. Each mock will be configured to track whether its methods have been called.
    2.  **Mock `LammpsRunner`**: The mock for `LammpsRunner` is the most important. Its `run` method will be configured to return a mock generator. This generator will be programmed to yield `'stable'` once, and then on the second call to `next()`, yield a dummy `Atoms` object.
    3.  **Mock `PacemakerTrainer`**: The `train` method of the mocked trainer will be configured to return a dummy potential file path, e.g., `'model_v1.yace'`.
    4.  **Execution**: The `CliRunner` will be used to invoke the `run` command on a dummy YAML file.
    5.  **Assertions**: This test will assert the sequence of events that defines one active learning cycle. We will assert that `mock_trainer.train.call_count == 2` (once at the start, and once again after the uncertain structure was processed). We will assert that `mock_dft_runner.run.call_count == 1`. We will assert that `mock_db_manager.write_calculation.call_count == 1`. Finally, we will check the call order to ensure the sequence was: `train`, `run` (on LAMMPS), `run` (on QE), `write`, `train`. This comprehensively verifies that the central nervous system of our application is correctly wiring all the components together to achieve the intelligent feedback loop.
