from pydantic import BaseModel, ConfigDict, Field


class ExplorationConfig(BaseModel):
    """
    Configuration for the Structure Generator (Explorer).
    """
    model_config = ConfigDict(extra='forbid')

    strategy: str = Field("random", description="Exploration strategy (e.g., 'random', 'md', 'mutation').")
    max_structures: int = Field(10, description="Number of structures to generate per cycle.")
    temperature: float = Field(300.0, description="Temperature for MD-based exploration (K).")

class DFTConfig(BaseModel):
    """
    Configuration for the Oracle (DFT Calculator).
    """
    model_config = ConfigDict(extra='forbid')

    calculator: str = Field("espresso", description="DFT code to use (e.g., 'espresso', 'vasp').")
    command: str = Field("pw.x", description="Command to run the DFT code.")
    ecutwfc: float = Field(40.0, description="Wavefunction cutoff energy (Ry for QE).")
    kpoints: list[int] = Field(default_factory=lambda: [1, 1, 1], description="K-point grid (e.g., [2, 2, 2]).")

class TrainingConfig(BaseModel):
    """
    Configuration for the Trainer (Pacemaker).
    """
    model_config = ConfigDict(extra='forbid')

    potential_type: str = Field("ace", description="Type of potential to train.")
    cutoff: float = Field(5.0, description="Radial cutoff (Angstrom).")
    max_degree: int = Field(10, description="Maximum polynomial degree.")
    ladder_step: list[int] = Field(default_factory=lambda: [1, 2], description="Body order ladder steps.")

class GlobalConfig(BaseModel):
    """
    Root configuration for the MLIP Pipeline.
    """
    model_config = ConfigDict(extra='forbid')

    execution_mode: str = Field("mock", description="Execution mode: 'mock' or 'real'.")
    max_cycles: int = Field(5, description="Maximum number of active learning cycles.")
    project_name: str = Field("mlip_project", description="Name of the project.")

    exploration: ExplorationConfig = Field(default_factory=lambda: ExplorationConfig(), description="Exploration settings.")
    dft: DFTConfig = Field(default_factory=lambda: DFTConfig(), description="DFT settings.")
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig(), description="Training settings.")
