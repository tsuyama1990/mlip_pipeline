# CYCLE03 User Acceptance Testing: The Intelligence

This document outlines the User Acceptance Testing (UAT) plan for CYCLE03. This is the most exciting UAT yet, as it allows the user to witness the MLIP-AutoPipe system operating as a truly autonomous learning agent for the first time. The goal is to validate the entire closed-loop active learning pipeline. The user will be guided through a scenario where the system starts with a deliberately flawed potential, encounters a situation it doesn't understand during a simulation, and then automatically performs all the steps necessary to improve itself and fix the flaw. This UAT is designed to be a compelling demonstration of the system's "intelligence" and its ability to learn from experience without human guidance.

## 1. Test Scenarios

The UAT for CYCLE03 is a single, narrative-driven scenario that showcases the entire On-the-Fly (OTF) learning loop. This provides a much more powerful and intuitive experience than testing the individual components separately.

| Scenario ID | Description | Priority |
| :--- | :--- | :--- |
| **UAT-C3-001** | **Observe the Autonomous "Healing" of a Flawed Potential** | **High** |

**Scenario UAT-C3-001 Details:**

*   **Objective**: To allow a user to observe the full, end-to-end active learning process in action. The user will see the system detect a high-uncertainty event during a molecular dynamics (MD) simulation, automatically extract the relevant atomic configuration, send it for a new "DFT" calculation, and retrain the potential to create a more robust and accurate version.
*   **User Story**: "As a materials scientist, I've been told the system can learn on its own, but I want to see it happen. I want to start a simulation with a 'bad' potential that I know will fail under certain conditions. I expect to see the system recognize its own failure, and I want to watch it automatically gather the data it needs to fix the problem and produce a better potential. I need to be able to compare the behavior of the 'bad' potential and the new, 'healed' potential to be convinced that the system has actually learned something useful."
*   **Methodology**: This UAT will use a Jupyter Notebook (`UAT_CYCLE03.ipynb`) and a simplified, 2D Lennard-Jones (LJ) system. The 2D system allows for easy visualization of the atomic simulation. The "DFT" calculations will be replaced with a mock calculator that returns the exact analytical LJ forces and energy, making the test fast and deterministic.
    1.  **Setup - The "Flawed" Potential**: The notebook will begin by providing the user with a pre-trained MLIP for the 2D LJ system. This potential will have been trained on data from a liquid state, but with a deliberate hole in the training data for very short inter-atomic distances. It's a "brittle" potential that knows how atoms interact normally, but not during high-energy collisions.
    2.  **Act 1 - The Failure**: The user will launch an MD simulation of a few LJ particles with high initial velocities, using this flawed potential. The notebook will generate a real-time animation of the particle trajectories. The user will see the particles move correctly until two get very close, at which point the simulation will become unstable or crash because the potential is extrapolating into unknown territory. The notebook will simultaneously display a plot of the potential's `extrapolation_grade` (uncertainty), showing a dramatic spike at the moment of near-collision. The system will report that it has detected an uncertainty event and triggered the learning pipeline.
    3.  **Act 2 - The Learning**: The notebook will show the user updates from the background learning process. It will report that a new structure has been sent to the "DFT" queue, that the "DFT" calculation (the mock LJ calculator) is complete, and that this new data has triggered a retraining of the potential.
    4.  **Act 3 - The Redemption**: Once the new, retrained potential (`.yace` file) is ready, the user will execute the final cells in the notebook. These cells will re-run the exact same high-velocity simulation, but this time using the new, "healed" potential. The notebook will again show a real-time animation of the particles. This time, the user will see the particles collide and scatter off each other realistically and smoothly. The plot of the `extrapolation_grade` will remain low and stable throughout, proving that the potential is now confident in this region of the configuration space.
*   **Why this UAT is amazing for the user**: This UAT tells a story. It creates a moment of dramatic failure and then shows the system intelligently recovering and learning from that failure. The side-by-side comparison of the "before" and "after" simulations provides an undeniable, visual proof of the system's autonomous learning capability. It's the most direct and compelling way to demonstrate the core value proposition of the entire MLIP-AutoPipe project.

## 2. Behavior Definitions

The following Gherkin-style definitions describe the expected behavior of the system for the UAT scenario.

### Scenario: Observe the Autonomous "Healing" of a Flawed Potential

*   **GIVEN** a user who has the `UAT_CYCLE03.ipynb` notebook.
*   **AND** the system is configured with a "flawed" Lennard-Jones potential that lacks data for short-range atomic interactions.
*   **AND** the system's OTF uncertainty threshold is set to a reasonable value (e.g., 5.0).

*   **WHEN** the user launches the MD simulation of high-velocity particles using the "flawed" potential.
*   **THEN** the simulation should run until two or more particles approach each other at a very short distance.
*   **AND** at that moment, the system's uncertainty metric (`extrapolation_grade`) must exceed the defined threshold.
*   **AND** the simulation must pause automatically.
*   **AND** the system must log a message indicating that an uncertainty event was detected and that a structure has been extracted for learning.

*   **GIVEN** an uncertainty event has been triggered.

*   **WHEN** the background learning process is active.
*   **THEN** the extracted atomic configuration should be processed by the mock DFT calculator.
*   **AND** the new, correct data point must be added to the training database.
*   **AND** the Trainer module must automatically detect the new data and begin a retraining job.
*   **AND** a new, versioned potential file (e.g., `lj_potential_v2.yace`) must be created upon completion.

*   **GIVEN** a new, retrained potential has been generated.

*   **WHEN** the user launches the MD simulation again, this time using the new "healed" potential.
*   **THEN** the simulation should run stably for its entire duration, including during high-energy collisions.
*   **AND** the particles should scatter realistically after colliding.
*   **AND** the uncertainty metric (`extrapolation_grade`) must remain below the threshold for the entire duration of the simulation.
*   **AND** the user should see a final confirmation message in the notebook declaring the UAT successful.
