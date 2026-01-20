"""
MLIP-AutoPipe: Machine Learning Interatomic Potential Automated Pipeline.

Architecture Overview:
----------------------
The system is designed as a modular, event-driven pipeline for active learning of MLIPs.

Core Modules:
- **Config**: Pydantic schemas for strict validation (`mlip_autopipec.config`).
- **Data**: Database abstraction layer using ASE DB (`mlip_autopipec.core.database`).
- **Generator**: Structure generation (SQS, NMS, Defects) (`mlip_autopipec.generator`).
- **Surrogate**: Pre-screening using MACE and FPS (`mlip_autopipec.surrogate`).
- **DFT**: Quantum Espresso execution and recovery (`mlip_autopipec.dft`).
- **Training**: ACE potential training via Pacemaker (`mlip_autopipec.training`).
- **Inference**: MD simulations (LAMMPS) and uncertainty quantification (`mlip_autopipec.inference`).
- **Orchestration**: Workflow management and distributed execution (`mlip_autopipec.orchestration`).

Data Flow:
----------
1. **Exploration**: Generator -> Candidates -> Surrogate -> Selected Structures -> DB.
2. **Labeling**: DB (pending) -> Orchestrator -> TaskQueue -> DFT -> DB (labeled).
3. **Training**: DB (labeled) -> DatasetBuilder -> Pacemaker -> Potential (.yace).
4. **Inference**: Potential -> LAMMPS -> Uncertainty Check -> DB (new candidates).

Dependencies:
-------------
- ASE (Atomic Simulation Environment)
- Pydantic (Data Validation)
- Dask (Distributed Computing)
- Matplotlib (Visualization)
"""
