# Cycle 02 UAT: Automated DFT Factory

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-02-01** | High | **Standard SCF Calculation** | Verify that the system can successfully generate an input file, "run" it (mocked), and parse the resulting Energy, Forces, and Stress tensors into a typed object. The results must match the mock data exactly. |
| **UAT-02-02** | High | **Convergence Auto-Recovery** | Verify that when the DFT code reports a convergence failure, the system automatically detects it, adjusts the mixing parameters, and retries the calculation without user intervention. The logs should show the retry attempts. |
| **UAT-02-03** | Medium | **Magnetic System Handling** | Verify that for systems containing magnetic elements (Fe, Co, Ni), the input generator automatically enables spin-polarization (`nspin=2`) and assigns initial magnetic moments. |
| **UAT-02-04** | Medium | **Resource Timeout Handling** | Verify that if a calculation exceeds the wall-time limit, it is terminated gracefully and marked as failed (or retried if applicable). |
| **UAT-02-05** | Low | **Output Validation** | Verify that the system rejects "Unphysical" results (e.g., Energy > 0, Forces > 1000 eV/A) even if the code exited with 0. |

### Recommended Demo
Create `demo_02_dft_factory.ipynb`.
Since we likely do not have `pw.x` installed in the demo environment, this notebook will use the **Mock Mode** of the `QERunner`.
1.  **Block 1**: Setup a `DFTConfig` enabling "mock_mode".
2.  **Block 2**: Create an `ase.Atoms` object (e.g., Aluminum FCC).
3.  **Block 3**: Run `runner.run(atoms)`. Show the input file that *would* have been generated (printing `runner.last_input_file`).
4.  **Block 4**: Print the resulting Energy and Forces (which comes from the mock).
5.  **Block 5**: Create a "Hard to Converge" mock scenario. Run it and show the logs where it says "Retrying with mixing_beta=0.3".

## 2. Behavior Definitions

### Scenario: Successful Calculation
**GIVEN** an `ase.Atoms` object representing an Aluminum crystal.
**AND** a `QERunner` configured with a valid pseudopotential path.
**WHEN** `runner.run(atoms)` is executed.
**THEN** the system should generate a `pw.in` file containing `ATOMIC_SPECIES Al Al.pbe-n-kjpaw_psl.1.0.0.UPF`.
**AND** it should execute the command defined in config.
**AND** it should return a `DFTResult` object where `succeeded` is `True`.
**AND** `result.forces` should be a Numpy array of shape (N, 3).
**AND** the `tstress` flag should be true in the input.

### Scenario: Auto-Recovery
**GIVEN** a structure that causes Quantum Espresso to print "convergence NOT achieved" (Mocked).
**WHEN** `runner.run(atoms)` is executed.
**THEN** the system should NOT raise an exception immediately.
**AND** the logs should show: "Detected error: Convergence Fail. Strategy: Reduce Mixing Beta".
**AND** the system should launch a second calculation.
**AND** the input file for the second calculation should contain `mixing_beta = 0.3` (assuming default was 0.7).
**AND** if the second run succeeds, the final result should be successful.
**AND** `result.final_mixing_beta` should be 0.3.

### Scenario: Magnetic Initialization
**GIVEN** an `Atoms` object containing Iron (Fe).
**WHEN** the input file is generated.
**THEN** the `SYSTEM` card should contain `nspin = 2`.
**AND** the `ATOMIC_SPECIES` card should likely define two types of Iron if trying to do AFM, or just one for FM.
**AND** `starting_magnetization(1)` should be set to a non-zero value (e.g., 0.5).

### Scenario: Fatal Failure
**GIVEN** a structure that causes a segmentation fault immediately.
**WHEN** `runner.run(atoms)` is executed.
**THEN** the system should retry N times (in case it was a glitch).
**AND** after N times, it should raise `DFTFatalError`.
**AND** the error should contain the last few lines of the stderr.
