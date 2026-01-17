from pathlib import Path
from uuid import uuid4
import yaml

from mlip_autopipec.config.models import (
    SystemConfig,
    MinimalConfig,
    UserInputConfig, # For backward compat
    # Import other configs for heuristics if needed
    DFTConfig,
    DFTInputParameters,
    ExplorerConfig,
    FingerprintConfig,
    InferenceConfig,
    MagnetismConfig,
    MDConfig,
    Pseudopotentials,
    SmearingConfig,
    StartingMagnetization,
    TrainingConfig,
    UncertaintyConfig,
    CutoffConfig
)

class ConfigFactory:
    """A factory for creating application configurations."""

    @staticmethod
    def from_yaml(config_path: Path) -> SystemConfig:
        """
        Loads a YAML file and creates a SystemConfig.
        This is the main entry point for the Cycle 1 CLI.
        """
        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Validate MinimalConfig
        minimal = MinimalConfig.model_validate(data)

        return ConfigFactory.from_minimal(minimal, base_dir=config_path.parent)

    @staticmethod
    def from_minimal(minimal: MinimalConfig, base_dir: Path | None = None) -> SystemConfig:
        """
        Expands MinimalConfig into SystemConfig.
        Creates necessary directory structures.
        """
        # Determine working directory
        # If base_dir is None, use cwd
        if base_dir is None:
            base_dir = Path.cwd()

        # Project directory
        project_dir = base_dir / minimal.project_name
        # Note: We resolve it to absolute path
        project_dir = project_dir.resolve()

        # Create directory
        project_dir.mkdir(parents=True, exist_ok=True)

        db_path = project_dir / "project.db"
        log_path = project_dir / "system.log"

        run_uuid = uuid4()

        # Apply Heuristics (Cycle 2+ features, kept for compatibility/completeness)
        # We can populate the optional fields of SystemConfig here.

        elements = minimal.target_system.elements

        # DFT Heuristics
        pseudos = {el: f"{el}_pbe_v1.uspp.F.UPF" for el in elements}
        cutoffs = CutoffConfig(wavefunction=60.0, density=240.0)
        magnetism = None
        if "Fe" in elements or "Ni" in elements or "Co" in elements:
             magnetism = MagnetismConfig(
                nspin=2,
                starting_magnetization=StartingMagnetization(dict.fromkeys(elements, 1.0)),
            )

        dft_input_params = DFTInputParameters(
            pseudopotentials=Pseudopotentials(pseudos),
            cutoffs=cutoffs,
            k_points=(3, 3, 3),
            smearing=SmearingConfig(),
            magnetism=magnetism,
        )
        dft_config = DFTConfig(dft_input_params=dft_input_params)

        # Explorer Config Heuristics
        fingerprint_config = FingerprintConfig(species=elements)
        explorer_config = ExplorerConfig(
            surrogate_model_path="path/to/mace.model",
            max_force_threshold=15.0,
            fingerprint=fingerprint_config,
        )

        # Training Config
        training_config = TrainingConfig(
            data_source_db=str(db_path.name), # relative name often used in pacemaker
        )

        # Inference Config
        # minimal no longer has simulation_goal required in Cycle 1 UAT input,
        # but if we want to support it if present (it's not in MinimalConfig definition anymore)
        # We'll use defaults.
        md_config = MDConfig(temperature=300.0)
        inference_config = InferenceConfig(
            md_params=md_config,
            uncertainty_params=UncertaintyConfig(),
        )

        return SystemConfig(
            minimal=minimal,
            working_dir=project_dir,
            db_path=db_path,
            log_path=log_path,
            project_name=minimal.project_name, # Optional field
            run_uuid=run_uuid, # Optional field
            dft_config=dft_config,
            explorer_config=explorer_config,
            training_config=training_config,
            inference_config=inference_config
        )

    @staticmethod
    def from_user_input(user_config: UserInputConfig) -> SystemConfig:
        """Alias for from_minimal for backward compatibility."""
        return ConfigFactory.from_minimal(user_config)
