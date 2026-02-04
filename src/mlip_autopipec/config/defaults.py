from .config_model import DFTConfig, ExplorationConfig, GlobalConfig, TrainingConfig


def get_default_config() -> GlobalConfig:
    """Returns a default configuration for Cycle 01 (Mock mode)."""
    return GlobalConfig(
        project_name="mlip_project",
        execution_mode="mock",
        cycles=3,
        dft=DFTConfig(calculator="lj", kpoints_density=0.04, encut=500.0),
        training=TrainingConfig(potential_type="ace", cutoff=5.0, max_degree=1),
        exploration=ExplorationConfig(strategy="random", num_candidates=10, supercell_size=2),
    )
