# MLIP-AutoPipe: Cycle 05 User Acceptance Testing

- **Cycle**: 05
- **Title**: The Intelligence - Full Active Learning Loop
- **Status**: Design

---

## 1. Test Scenarios

User Acceptance Testing for Cycle 05 is the grand reveal. It's where the "magic" of the autonomous, self-improving system is made manifest. The UAT is designed to be a compelling narrative, demonstrating a system that can identify its own knowledge gaps and actively work to fill them. The user will be amazed as they watch the system's performance metrics (like the error rate of the potential) tangibly improve with each iteration of the active learning loop. The UAT will take the form of a Jupyter Notebook (`05_active_learning_in_action.ipynb`) that simulates the entire end-to-end workflow, using extensive mocks and visualizations to tell a clear and powerful story of a system that learns.

---

### **Scenario ID: UAT-C05-01**
- **Title**: Observing the Self-Improving MLIP in a Live Simulation
- **Priority**: Critical

**Description:**
This scenario provides a simulated but complete end-to-end demonstration of the active learning loop. The user will start with a "bad" version-1 potential trained on minimal data. They will launch a simulated MD run which will quickly encounter a situation where the potential is uncertain. The user will then watch as the system automatically pauses the simulation, triggers a new DFT calculation (which will be mocked), retrains the potential to create a "better" version-2 model, and then resumes the simulation. A simple plot of the potential's error over time will show a distinct drop after the retraining event, providing clear, quantitative proof of the system's learning capability.

**UAT Steps via Jupyter Notebook (`05_active_learning_in_action.ipynb`):**

**Part 1: The Initial State**
*   The notebook will import all the necessary application components and plotting libraries.
*   **Step 1.1:** It will start with a pre-prepared database containing a small initial dataset, and a pre-trained "v1" potential file based on this data.
*   **Step 1.2:** To establish a baseline, the notebook will load the "v1" potential and evaluate its Root Mean Square Error (RMSE) on a small, known test set of structures. The RMSE will be intentionally high. A plot will be created showing the current RMSE value.

**Part 2: The Active Learning Loop in Action**
*   **Step 2.1:** The user will execute a cell that kicks off the main application logic from `app.py`. The notebook will explain that the MD simulation (`LammpsRunner`) and DFT calculations (`QEProcessRunner`) are heavily mocked to make the process visual and instantaneous.
*   **Step 2.2 (The Trigger):** The mocked `LammpsRunner` is programmed to run for a few steps and then report a high uncertainty. The notebook cell's output will print log messages in real-time, showing:
    *   "Starting MD simulation with potential v1..."
    *   "MD step 1... stable."
    *   "MD step 2... stable."
    *   "MD step 3... UNCERTAINTY DETECTED! Pausing simulation."
    *   "Submitting high-uncertainty structure to DFT Factory..."
    *   (Mocked) "DFT calculation complete."
    *   "Adding new data to database..."
*   This clear, narrative output lets the user follow the system's decision-making process.

**Part 3: The "Aha!" Moment - Verifiable Improvement**
*   **Step 3.1:** The log output will continue, showing the system's response:
    *   "New data detected. Retraining potential..."
    *   (Mocked) "Training complete. New potential 'v2' created."
*   **Step 3.2 (The Payoff):** The user will execute a new cell that loads the newly generated "v2" potential. It will then re-calculate the RMSE on the *exact same test set* as before.
*   **Step 3.3:** The notebook will update the plot of RMSE vs. Training Cycle. The user will see a clear, significant drop in the error bar from cycle 1 to cycle 2. This is the ultimate proof: the system identified a weakness, acquired new knowledge, and demonstrably improved its own performance. The user is amazed because they have just witnessed a fully autonomous learning cycle. The notebook will conclude by explaining that the real system would now resume the MD simulation with this more accurate potential, continuing the cycle until the entire simulation is stable.

---

## 2. Behavior Definitions

These Gherkin-style definitions specify the expected high-level behavior of the fully integrated active learning system.

**Feature: Autonomous Potential Refinement**
As a materials scientist, I want the system to automatically detect when its MLIP is uncertain and then trigger a retraining loop with new DFT data, so that the potential's accuracy and reliability improve over the course of the simulation.

---

**Scenario: Successful Active Learning Cycle**

*   **GIVEN** a simulation is running with the current best MLIP, "potential_v1".
*   **AND** the `LammpsRunner` is executing an MD simulation.
*   **AND** at timestep 50, the `extrapolation_grade` calculated for the current structure exceeds the `uncertainty_threshold`.
*   **WHEN** this uncertainty is detected.
*   **THEN** the `LammpsRunner` must pause and yield the high-uncertainty `Atoms` object.
*   **AND** the main application logic must send this `Atoms` object to the `QEProcessRunner` to be calculated.
*   **AND** the result of the DFT calculation must be written to the database.
*   **AND** the `PacemakerTrainer` must then be invoked to train a new potential, "potential_v2", which now includes the new data point.
*   **AND** the simulation must then be prepared to resume using the improved "potential_v2".

---

**Scenario: Simulation Completes Without Uncertainty**

*   **GIVEN** a simulation is running with a highly accurate MLIP.
*   **AND** the simulation is configured to run for 1000 steps.
*   **WHEN** the `LammpsRunner` executes all 1000 steps of the MD simulation.
*   **AND** the `extrapolation_grade` never exceeds the `uncertainty_threshold` during the entire run.
*   **THEN** the DFT Factory and the Pacemaker Trainer must not be called at all.
*   **AND** the simulation should run to completion without interruption.
*   **AND** the application should exit gracefully, reporting a successful run.

---

**Scenario: Resuming a Paused Simulation**

*   **GIVEN** an active learning cycle has just completed, producing a new "potential_v2".
*   **AND** the MD simulation was paused at timestep 50.
*   **WHEN** the main application loop continues.
*   **THEN** a new `LammpsRunner` must be instantiated, loaded with the new "potential_v2".
*   **AND** the simulation it starts must resume from the state at timestep 50 (or a closely related starting point).
*   **AND** it should not restart the simulation from scratch at timestep 0.
