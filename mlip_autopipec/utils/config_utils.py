"""Utilities for handling configuration objects."""

from mlip_autopipec.config_schemas import (
    DFTConfig,
    DFTExecutable,
    DFTInput,
    InferenceParams,
    MDEnsemble,
    SystemConfig,
    UserConfig,
)


def generate_system_config_from_user_config(
    user_config: UserConfig,
) -> SystemConfig:
    """Generate the internal SystemConfig from the user-facing UserConfig.

    This function translates the user's high-level requests into a detailed,
    self-contained configuration object for the entire workflow. It defines
    default DFT parameters, MD settings, and other internal details.

    Args:
        user_config: The user-provided configuration.

    Returns:
        A fully populated SystemConfig object.

    """
    dft_input = DFTInput(
        pseudopotentials={el: f"{el}.UPF" for el in user_config.target_system.elements}
    )
    dft_config = DFTConfig(executable=DFTExecutable(), input=dft_input)

    md_ensemble = MDEnsemble(target_temperature_k=350.0)
    inference_params = InferenceParams(md_ensemble=md_ensemble)

    system_config = SystemConfig(
        dft=dft_config,
        inference=inference_params,
        target_system=user_config.target_system.model_copy(),
    )
    return system_config
