"""
MLIP-AutoPipeC: Autonomous Pipeline for MLIP Generation.

This package implements the "Zero-Human" protocol for generating Machine Learning
Interatomic Potentials. It is structured into distinct modules for configuration,
structure generation, surrogate pre-screening, DFT calculation, training, and inference.

## Package Architecture

*   **`config`**: Defines the data contracts (Pydantic schemas) for system configuration.
    These schemas (`SystemConfig`, `InferenceConfig`) act as the single source of truth.
*   **`data_models`**: Defines the domain objects exchanged between pipeline stages
    (e.g., `DFTResult`, `TrainingData`, `ExtractedStructure`).
*   **`inference`**: Contains the inference engine logic, including MD simulation (`LammpsRunner`),
    uncertainty quantification (`UncertaintyChecker`), and active learning extraction
    (`EmbeddingExtractor`, `ForceMasker`).
*   **`core`**: Provides low-level utilities like `DatabaseManager` and logging.
*   **`dft`**: Manages First-Principles calculations (Quantum Espresso).
*   **`training`**: Orchestrates ML potential training (Pacemaker).
*   **`generator`**: Creates initial atomic structures (SQS, NMS).
*   **`surrogate`**: Filters and selects structures using foundation models (MACE).

For detailed architecture design, see `dev_documents/system_prompts/SYSTEM_ARCHITECTURE.md`.
"""

__version__ = "0.1.0"
