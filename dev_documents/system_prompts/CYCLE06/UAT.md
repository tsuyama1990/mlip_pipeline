# MLIP-AutoPipe: Cycle 06 User Acceptance Testing

- **Cycle**: 06
- **Title**: Production Readiness - Advanced Embedding and Scalability
- **Status**: Design

---

## 1. Test Scenarios

User Acceptance Testing for Cycle 06 is designed to showcase the system's most scientifically advanced and operationally robust features. This UAT provides the user with a powerful demonstration of why MLIP-AutoPipe is not just a prototype, but a production-grade tool capable of generating high-quality, physically accurate potentials at scale. The key "wow" factor will be a clear, visual demonstration of the **Periodic Embedding and Force Masking** strategy, showing how this advanced technique leads to better, more transferable models. The UAT will be presented as a Jupyter Notebook (`06_advanced_techniques_and_scaling.ipynb`), which will use visualizations and simulated performance metrics to make the benefits of these advanced features clear and compelling.

---

### **Scenario ID: UAT-C06-01**
- **Title**: Visualising the Benefit of Force Masking
- **Priority**: Critical

**Description:**
This scenario provides a powerful, side-by-side comparison to demonstrate the scientific value of the force masking technique. The user will train two simple potentials. The first will be trained the "naive" way, using a small, periodically embedded cell but without applying a force mask. The second will be trained on the exact same structure but *with* force masking applied. The notebook will then evaluate a key physical property (e.g., the energy of a point defect) with both potentials. The user will be amazed to see that the potential trained with force masking gives a physically accurate result, while the naive potential gives a completely wrong answer. This provides a direct, quantitative demonstration of how the advanced scientific features in this cycle lead to superior outcomes.

**UAT Steps via Jupyter Notebook (`06_advanced_techniques_and_scaling.ipynb`):**

**Part 1: The Problem - Boundary Artefacts**
*   The notebook will begin by importing the necessary components and creating a large, pristine crystal structure.
*   **Step 1.1:** It will then programmatically create a vacancy in the center.
*   **Step 1.2:** The notebook will perform the periodic embedding extraction around the vacancy, creating a small sub-cell.
*   **Step 1.3:** A (mocked) DFT calculation is run on this small cell to get the "ground truth" forces.
*   **Step 1.4:** The notebook will use arrows to visualise the forces on the atoms in this sub-cell. The user will clearly see that the forces on the atoms at the very edge of the box are distorted and unphysical due to the artificial periodic boundary. This visually establishes the problem that force masking solves.

**Part 2: Training Two Potentials**
*   **Step 2.1 (Naive):** The notebook will use the `PacemakerTrainer` to train a simple potential (`potential_naive.yace`) using the data from the sub-cell, but with force masking turned OFF.
*   **Step 2.2 (Masked):** It will then train a second potential (`potential_masked.yace`) on the *exact same data*, but this time with force masking turned ON. The notebook will show the `force_mask` array of `1`s and `0`s, making it clear which atomic forces are being ignored.

**Part 3: The Payoff - Verifiable Accuracy**
*   **Step 3.1:** The user will now test the quality of both potentials. A key physical property that depends on accurate forces, like the vacancy formation energy, will be the test case.
*   **Step 3.2:** The notebook will calculate this property using the `potential_naive.yace`. The result will be significantly different from the expected DFT value.
*   **Step 3.3:** It will then calculate the same property using `potential_masked.yace`. The user will see that this result is in excellent agreement with the DFT value.
*   **Step 3.4:** A simple bar chart will compare the results: "DFT Ground Truth", "Prediction with Force Masking", and "Prediction without Force Masking". The visual makes the conclusion inescapable: the advanced force masking technique is essential for scientific accuracy.

---

### **Scenario ID: UAT-C06-02**
- **Title**: Demonstrating DFT Auto-Recovery
- **Priority**: High

**Description:**
This scenario provides the user with a tangible sense of the system's operational robustness. It simulates a common failure mode in a DFT calculation and shows how the `QEProcessRunner` can autonomously recover from it. This builds confidence that the system can be trusted to run unattended for long periods.

**UAT Steps via Jupyter Notebook:**

*   **Step 1:** The user will instantiate the `QEProcessRunner`.
*   **Step 2:** The notebook will explain that the underlying `subprocess.run` call is mocked. This mock is specially configured to fail the *first* time it is called (simulating a convergence failure) but to succeed on the second call.
*   **Step 3:** The user will execute `runner.run(atoms)`.
*   **Step 4:** The cell output will show a real-time log of the runner's internal logic:
    *   `INFO: Starting DFT calculation (Attempt 1/3)...`
    *   `WARNING: DFT calculation failed. Retrying with relaxed parameters (mixing_beta=0.3)...`
    *   `INFO: Starting DFT calculation (Attempt 2/3)...`
    *   `INFO: DFT calculation successful.`
*   The user is amazed because the system handled a potentially critical failure gracefully and automatically, without crashing or requiring any human input.

---

## 2. Behavior Definitions

**Feature: High-Fidelity Training with Force Masking**
As a scientist, I want the system to use periodic embedding with force masking when learning from local atomic events, so that the resulting potential is free from artificial boundary effects and yields physically accurate properties.

**Scenario: Training with Force Masking Enabled**

*   **GIVEN** an active learning loop has detected an uncertain atom and extracted a periodic sub-cell.
*   **AND** a `force_mask` array has been generated, with `0.0` for buffer atoms and `1.0` for core atoms.
*   **AND** this data has been saved to the database.
*   **WHEN** the `PacemakerTrainer` is invoked.
*   **THEN** it must query the database and retrieve the `force_mask` array along with the structure.
*   **AND** it must configure the Pacemaker training job to use these per-atom force weights.
*   **AND** the resulting MLIP's predictions for physical properties must be more accurate than a potential trained without the mask.

---

**Feature: Resilient DFT Execution**
As an HPC operator, I want the system to be resilient to common, transient DFT calculation failures, so that long-running workflows can complete successfully without unnecessary manual intervention.

**Scenario: Automatic Recovery from Convergence Failure**

*   **GIVEN** the `QEProcessRunner` is asked to calculate a difficult-to-converge structure.
*   **AND** the initial DFT run with default parameters fails to converge.
*   **WHEN** the `QEProcessRunner` detects this failure.
*   **THEN** it must not immediately crash or report failure.
*   **AND** it must automatically generate a new QE input file with more robust, relaxed convergence parameters (e.g., a lower `mixing_beta`).
*   **AND** it must re-submit the calculation with these new parameters.
*   **AND** if the second attempt succeeds, it should return the successful result as normal.

---

**Feature: Scalable Parallel Execution**
As a researcher, I want the system to leverage a task queue to execute multiple DFT calculations in parallel, so that the overall time-to-solution for generating a potential is dramatically reduced.

**Scenario: Submitting Multiple DFT Jobs to a Queue**

*   **GIVEN** the main application is connected to a Dask task queue.
*   **AND** the MD simulation identifies 5 high-uncertainty structures in rapid succession.
*   **WHEN** these 5 structures are detected.
*   **THEN** the main application must not block and calculate them one by one.
*   **AND** it must submit 5 independent DFT calculation tasks to the Dask queue in a non-blocking manner.
*   **AND** it should then proceed to monitor the status of the 5 corresponding "futures".
*   **AND** as each future is completed, the result should be added to the database.
