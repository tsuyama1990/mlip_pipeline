"""
This module provides a factory class for creating the comprehensive
SystemConfig from a high-level UserInputConfig.
"""

from uuid import uuid4
from pathlib import Path
import tempfile

from mlip_autopipec.config.models import (
    CutoffConfig,
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
    SystemConfig,
    TrainingConfig,
    UncertaintyConfig,
    UserInputConfig,
)


class ConfigFactory:
    """A factory for creating application configurations."""

    @staticmethod
    def from_user_input(user_config: UserInputConfig) -> SystemConfig:
        """
        Constructs the comprehensive SystemConfig from the UserInputConfig.

        This method applies heuristics and sensible defaults to expand the
        user's high-level request into a detailed, low-level execution plan.
        """
        project_name = user_config.project_name
        run_uuid = uuid4()
        elements = user_config.target_system.elements

        # DFT Config Heuristics
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
            k_points=(3, 3, 3),  # Placeholder
            smearing=SmearingConfig(),
            magnetism=magnetism,
        )
        dft_config = DFTConfig(dft_input_params=dft_input_params)

        # Explorer Config Heuristics
        fingerprint_config = FingerprintConfig(species=elements)
        explorer_config = ExplorerConfig(
            surrogate_model_path="path/to/mace.model",  # Placeholder
            max_force_threshold=15.0,
            fingerprint=fingerprint_config,
        )

        # Training Config Heuristics
        training_config = TrainingConfig(
            data_source_db=f"{project_name}.db",
        )

        # Inference Config Heuristics
        temp_range = user_config.simulation_goal.temperature_range
        md_config = MDConfig(
            temperature=temp_range[1] if temp_range else 300.0,
        )
        inference_config = InferenceConfig(
            md_params=md_config,
            uncertainty_params=UncertaintyConfig(),
        )

        return SystemConfig(
            project_name=project_name,
            run_uuid=run_uuid,
            dft_config=dft_config,
            explorer_config=explorer_config,
            training_config=training_config,
            inference_config=inference_config,
        )
