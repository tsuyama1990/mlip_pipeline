# Cycle 08: User Acceptance Testing (UAT)

## 1. Test Scenarios

Cycle 08 is about the User Experience.

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-08-01** | High | CLI Usability | Verify that a new user can initialize and run a project using only the command line. |
| **UAT-08-02** | Medium | Dashboard Visualization | Verify that the dashboard displays the learning progress and atomic structures correctly. |
| **UAT-08-03** | High | Full Pipeline "Zero-Human" | Verify that the entire process runs from start to finish without any manual intervention. |

### Recommended Notebooks
*   `notebooks/UAT_08_EndToEnd.ipynb`:
    *   This notebook will act as the "User Manual".
    *   It will execute CLI commands via `!mlip-auto ...` cells.
    *   It will embed the Dashboard (via IFrame or screenshots) to show the result.

## 2. Behavior Definitions

### UAT-08-01: CLI

**Narrative**:
A new grad student downloads the package. They want to start working immediately. They type `mlip-auto init`. A file `input.yaml` appears. They edit it to say "Cu". They type `mlip-auto run`. The system starts printing "Iteration 1: Generating structures...". They press Ctrl+C to go to lunch. The system saves `state.json`. They come back and type `mlip-auto run` again. It resumes.

```gherkin
Feature: Command Line Interface

  Scenario: Initializing a Project
    GIVEN a fresh directory
    WHEN the user types "mlip-auto init"
    THEN a template "input.yaml" should appear
    AND the template should contain helpful comments

  Scenario: Running a Project with Interruption
    GIVEN a valid input.yaml
    WHEN the user types "mlip-auto run input.yaml"
    AND interrupts the process with Ctrl+C after 10 seconds
    THEN a "state.json" file should be created
    AND when "mlip-auto run input.yaml" is run again
    THEN it should print "Resuming from iteration..."
```

### UAT-08-03: Zero-Human Protocol

**Narrative**:
The ultimate test. We set up a clean AWS instance. We install the package. We define a task: "Find the melting point of Al". We start the run and disconnect SSH. We come back 2 days later. We expect to see a `report.pdf` or `results` folder containing a graph of Density vs Temperature showing a sharp drop at 933K, and a converged `.yace` potential.

```gherkin
Feature: End-to-End Automation

  Scenario: Alloy Melt-Quench Campaign
    GIVEN a configured "Alloy Melt-Quench" task for Aluminum
    WHEN the pipeline is launched in background mode
    THEN it should:
      1. Generate initial SQS/Strain structures
      2. Run DFT (mocked for speed in test environment)
      3. Train potential to RMSE < 5 meV/A
      4. Run MD melting simulation
      5. Automatically sample the liquid phase when uncertainty spikes
      6. Retrain
    AND finally produce a "results/" directory containing the potential and property report
    WITHOUT requiring the user to press any keys or fix any errors manually
```
