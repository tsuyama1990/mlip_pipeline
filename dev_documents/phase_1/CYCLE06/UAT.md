# CYCLE06: Resilience and Scalability (UAT.md)

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 6 is designed to build the user's trust in the system's ability to handle long, complex, and potentially failure-prone workflows. The key user experiences are **security** and **speed**. The user should be amazed by the system's ability to recover from a simulated catastrophic failure without losing any progress, and they should clearly see the performance benefits of parallel execution. These scenarios are best demonstrated in a Jupyter Notebook, which allows for process control and clear timing metrics.

| Scenario ID   | Test Scenario                                             | Priority |
|---------------|-----------------------------------------------------------|----------|
| UAT-C6-001    | **"Pull the Plug" - Catastrophic Failure and Recovery**   | **High**     |
| UAT-C6-002    | **High-Throughput Parallel Execution with Dask**          | **High**     |
| UAT-C6-003    | **Demonstration of Automatic DFT Retry Decorator**        | **Medium**   |

---

### **Scenario UAT-C6-001: "Pull the Plug" - Catastrophic Failure and Recovery**

**(Min 300 words)**

**Description:**
This is the most critical and impressive UAT for this cycle. It directly addresses a major source of anxiety for anyone running computations that last for days: what happens if the power goes out or the system crashes? This scenario will simulate this exact event. The user will watch a workflow start, see that it has work in progress, and then see the process get forcefully terminated. They will then witness a new process start up, read the checkpoint file, and seamlessly resume the work exactly where the previous one left off. This provides a powerful demonstration of the system's resilience and gives the user peace of mind.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will set up a temporary working directory. It will also define a simple "mock" DFT function that takes 5 seconds to run (`time.sleep(5)`) and returns a dummy result. This allows the test to be run quickly and predictably.
2.  **Part 1: The Initial Run and the "Crash":**
    -   A `WorkflowManager` will be configured to run 10 of these mock DFT jobs.
    -   The `WorkflowManager` will be launched in a separate Python process using `multiprocessing.Process`. This is crucial as it allows the notebook to remain in control.
    -   The notebook will let the process run for just 7 seconds—enough time to submit the jobs, but only enough for one to complete.
    -   The notebook will then forcefully terminate the process using `process.kill()`. A dramatic message will be printed: "🔴 **Workflow process terminated unexpectedly!**"
3.  **Inspect the Damage (and the Checkpoint):**
    -   The notebook will then inspect the working directory. It will show that a `checkpoint.json` file exists.
    -   It will load and display the contents of this file, pointing out the key fields: `pending_job_ids` (which should list 9 jobs) and `completed_job_ids` (which should list 1 job). This makes the state of the crashed run completely transparent.
4.  **Part 2: The Recovery:**
    -   A *new* `WorkflowManager` instance will be created in a new process, pointed at the *same* working directory.
    -   The notebook will launch this new process. The user will see logs indicating that a checkpoint has been found and that the manager is re-submitting the 9 pending jobs.
    -   The notebook will let this new process run to completion.
5.  **Final Verification:**
    -   Finally, the notebook will connect to the ASE database in the working directory and count the number of entries. It will print the triumphant result: "✅ **Recovery successful!** The database contains all 10 results, proving that no work was lost."

---

### **Scenario UAT-C6-002: High-Throughput Parallel Execution with Dask**

**(Min 300 words)**

**Description:**
This scenario demonstrates the immense performance gain from using a parallel task scheduler like Dask. The user will see a direct, timed comparison between running a batch of jobs sequentially versus running them in parallel across multiple workers. The visual and quantitative difference will be striking, showing how the system is designed to effectively leverage modern multi-core processors or HPC clusters to accelerate research.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will import Dask and start a `dask.distributed.LocalCluster` with a specific number of workers (e.g., 4). It will also use the same 2-second mock DFT function (`time.sleep(2)`) from the previous scenario.
2.  **The Sequential Baseline:**
    -   The notebook will first run a simple `for` loop to execute 8 of the mock DFT jobs one after the other.
    -   It will use the `%%time` magic command to measure the total execution time.
    -   The result will be printed: "Total time for 8 jobs sequentially: **~16 seconds**."
3.  **The Parallel Workflow:**
    -   Next, the notebook will configure a `WorkflowManager` to use the local Dask cluster.
    -   It will then launch the workflow to process the same 8 mock DFT jobs.
    -   The notebook will provide a link to the Dask dashboard, allowing the user to open it in another tab and watch the tasks being distributed across the 4 workers in real-time. This is a very compelling visualisation.
    -   Again, the `%%time` magic command will measure the total time.
4.  **The Comparison:**
    -   The result will be printed directly below the sequential one: "Total time for 8 jobs in parallel with 4 workers: **~4 seconds**."
5.  **Explanation:** A markdown cell will summarize the outcome with a clear, powerful statement: "By distributing the work across 4 parallel workers, the system achieved a **4x speedup**, completing the same amount of work in a fraction of the time. This scalability is what allows the MLIP-AutoPipe to tackle large-scale problems efficiently."

---

## 2. Behavior Definitions

### **UAT-C6-001: "Pull the Plug" - Catastrophic Failure and Recovery**

```gherkin
Feature: Workflow Resilience and Recovery
  As a researcher running a multi-day simulation,
  I want the system to save its state continuously,
  So that I can resume the workflow from the last checkpoint if the system crashes.

  Scenario: A workflow is terminated mid-run and then resumed
    Given a workflow is running a batch of 20 DFT jobs
    And the workflow is forcefully terminated after only 5 jobs have completed
    When a new workflow is started in the same directory
    Then the new workflow should detect and load the checkpoint file
    And it should re-submit the 15 jobs that were pending
    And it should not re-submit the 5 jobs that were already complete
    And the new workflow should eventually complete, having processed all 20 unique jobs.
```

### **UAT-C6-002: High-Throughput Parallel Execution with Dask**

```gherkin
Feature: Scalable Parallel Execution
  As a user with access to a multi-core machine or HPC cluster,
  I want the system to execute independent DFT calculations in parallel,
  So that I can complete my data generation campaign faster.

  Scenario: Running a batch of jobs on a multi-worker Dask cluster
    Given a batch of 12 independent jobs, each of which takes 3 seconds
    And a Dask cluster with 6 available workers
    When the workflow is run to process the batch
    Then the total execution time should be significantly less than 36 seconds (12 * 3s)
    And the total execution time should be close to 6 seconds (12 jobs / 6 workers * 3s/job), plus system overhead.
```

### **UAT-C6-003: Demonstration of Automatic DFT Retry Decorator**

```gherkin
Feature: Automated Retry for Failed Calculations
  As a user running thousands of calculations,
  I want the system to automatically retry a calculation with different parameters if it fails,
  So that transient errors do not halt my entire workflow.

  Scenario: A DFT calculation fails, but succeeds on the second try
    Given a DFT calculation that is configured to fail on its first run but succeed on its second
    And the system is configured with a retry-decorator of 3 attempts
    When the workflow attempts to run this calculation
    Then the system logs should show that the first attempt failed
    And the system logs should show that it is "Retrying calculation (attempt 2 of 3)..."
    And the second attempt should succeed
    And the workflow should continue without crashing.
```
