# MLIP-AutoPipe

MLIP-AutoPipe (Machine Learning Interatomic Potential - Automated Pipeline) is a project to create a fully autonomous system for generating and validating machine learning interatomic potentials.

## CYCLE01

This cycle implemented the foundational components of the pipeline:

*   **Module A: Physics-Informed Generator**: Capable of generating structures for alloys (using SQS) and for equation of state calculations.
*   **Module C: Automated DFT Factory**: A robust wrapper around Quantum Espresso that can perform DFT calculations with automated error recovery.

The system is driven by a command-line interface and uses Pydantic for robust configuration management.

## CYCLE02

This cycle introduced the first layer of intelligence and efficiency into the pipeline:

*   **Module B: Surrogate Explorer**: Uses a pre-trained MACE model to filter and select the most diverse and informative structures for DFT calculations.
*   **Module D: Pacemaker Trainer**: A manually-triggered module to train a new MLIP using the Pacemaker framework.
