# CYCLE08: Monitoring and Usability (UAT.md)

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 8 is designed to showcase the system's transparency and user-friendliness. The primary user experience is one of **insight and control**. After setting up a long-running, autonomous job, the user needs a simple way to check its pulse and understand its progress. The `mlip-auto status` command and the resulting dashboard provide exactly that. This UAT will demonstrate how easy it is to get a comprehensive overview of the workflow's state and performance.

| Scenario ID   | Test Scenario                                             | Priority |
|---------------|-----------------------------------------------------------|----------|
| UAT-C8-001    | **Generating and Viewing the Status Dashboard**           | **High**     |
| UAT-C8-002    | **Interpreting Dashboard for Workflow Insights**          | **High**     |

---

### **Scenario UAT-C8-001: Generating and Viewing the Status Dashboard**

**(Min 300 words)**

**Description:**
This is the "happy path" UAT for the monitoring feature. It demonstrates the primary user journey: running a single command to generate a comprehensive status report. The user will simulate being in their project directory, execute `mlip-auto status`, and immediately see a professional, easy-to-read HTML dashboard open in their browser. This scenario verifies that the command is correctly wired, that the data is gathered without errors, and that the final report is successfully generated and presented to the user.

**UAT Steps in Jupyter Notebook:**
1.  **Setup: Simulate a Running Project:** The notebook will first create a realistic-looking project directory. This involves:
    -   Programmatically creating a `checkpoint.json` file with plausible data (e.g., `current_generation: 3`, a list of pending job IDs, and a populated `training_history`).
    -   Programmatically creating an ASE database (`project.db`) and populating it with a mix of structures tagged with different `config_type` metadata (e.g., 100 'sqs_initial', 30 'active_learning_gen1', 25 'active_learning_gen2').
    -   The notebook will display the key data it has just created, setting the stage for what the user should expect to see in the dashboard.
2.  **Execution:** A `%%bash` cell will execute the new CLI command.
    ```bash
    %%bash
    # Run the status command in the simulated project directory
    mlip-auto status .
    ```
3.  **Verify Output:** The notebook will capture and display the output of the command. The user will see:
    ```
    ✅ Gathering data from project directory...
    📊 Generating plots and rendering HTML...
    🎉 Dashboard generated at: /path/to/project/dir/dashboard.html
    Opening dashboard in your web browser...
    ```
4.  **Display the Result:** While the command would open a real browser tab, the notebook will simulate this by loading and displaying the contents of the generated `dashboard.html` file directly within an `IFrame`. The user will see the fully rendered dashboard inside the notebook itself.
5.  **Explanation:** A markdown cell will conclude: "With a single command, `mlip-auto status`, the system interrogated the project files and produced a comprehensive, interactive dashboard. This provides a simple and powerful way to check on your project's status at any time."

---

### **Scenario UAT-C8-002: Interpreting Dashboard for Workflow Insights**

**(Min 300 words)**

**Description:**
This scenario focuses on the *value* of the information presented in the dashboard. It goes beyond simply generating the report and teaches the user how to interpret its contents to gain meaningful insights into their workflow's performance and progress. The user will be guided through the key sections of the dashboard generated in the previous scenario, explaining what each plot and statistic means. This helps the user appreciate that the dashboard isn't just a status report; it's a scientific tool for understanding the active learning process itself.

**UAT Steps in Jupyter Notebook:**
1.  **Prerequisite:** This scenario uses the dashboard displayed in the IFrame from UAT-C8-001.
2.  **Guidance Section 1: Overall Progress:** The notebook will display a screenshot of the top section of the dashboard (or refer to the IFrame) which shows "Key Metrics". A markdown cell will explain:
    > "This section gives you a high-level overview. You can see you are in **Generation 3** of the active learning loop. The system has completed **155 DFT calculations** so far and has **15 more pending** in the queue. This tells you that the system is actively working and making progress."
3.  **Guidance Section 2: Dataset Growth:** The notebook will highlight the "Dataset Composition" pie chart. The explanation will read:
    > "This chart shows the provenance of your training data. You can see that your dataset started with **100 initial structures** and has been augmented by **55 structures** discovered through the active learning process. This confirms that the OTF inference engine is successfully finding new, informative configurations."
4.  **Guidance Section 3: Model Performance:** The notebook will focus on the "Force RMSE vs. Generation" line plot. The explanation will be:
    > "This is the most important plot for judging the model's improvement. You can see the Force RMSE (a measure of error) started at a certain level in Generation 1 and has **decreased with each new generation** of active learning. This provides clear, quantitative evidence that the 'Zero-Human' protocol is working: the model is genuinely getting more accurate as it gathers more data."
5.  **Conclusion:** A final markdown cell will summarize the user's journey: "By using the dashboard, you can move beyond just running the workflow to *understanding* it. You can track progress, verify that the active learning is effective, and make informed decisions about your research, all from a simple, on-demand report."

---

## 2. Behavior Definitions

### **UAT-C8-001: Generating and Viewing the Status Dashboard**

```gherkin
Feature: On-Demand Workflow Monitoring
  As a user running a long-duration project,
  I want to be able to generate a status report on demand,
  So that I can check the progress of my workflow.

  Scenario: User runs the 'status' command in a valid project directory
    Given a project directory containing a valid `checkpoint.json` file and an `project.db` database
    When I execute the command `mlip-auto status .` in that directory
    Then the command should complete successfully with an exit code of 0
    And it should create a file named `dashboard.html` in the same directory
    And the command's output should state that the dashboard has been generated.
```

### **UAT-C8-002: Interpreting Dashboard for Workflow Insights**

```gherkin
Feature: Understanding Workflow Performance
  As a researcher,
  I want to view key performance metrics of my active learning run,
  So that I can determine if the model is improving and the workflow is efficient.

  Scenario: The generated dashboard correctly reflects the project state
    Given a project that is in active learning generation 3 and has a history of 3 training runs
    When the `dashboard.html` file is generated for this project
    Then the dashboard should display "Current Generation: 3"
    And the dashboard should contain a plot titled "Force RMSE vs. Generation"
    And that plot should contain exactly 3 data points, corresponding to the training history.

  Scenario: The dashboard correctly shows the composition of the dataset
    Given a project database containing 100 structures of type 'initial' and 50 of type 'active_learning'
    When the `dashboard.html` file is generated for this project
    Then the dashboard should contain a chart or table
    And that chart should show the count for 'initial' as 100 and the count for 'active_learning' as 50.
```
