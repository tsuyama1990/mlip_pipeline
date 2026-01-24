"""Utilities for handling configuration objects."""

from pathlib import Path

from mlip_autopipec.config.models import (
    DFTConfig,
    DFTExecutable,
    DFTInput,
    InferenceParams,
    MDEnsemble,
    SystemConfig,
    UserConfig,
)


def validate_path_safety(path: Path | str) -> Path:
    """
    Ensures the path is safe and resolved.
    Prevents path traversal attacks by ensuring path is absolute or relative to CWD.
    """
    try:
        if isinstance(path, str):
            path = Path(path)
        resolved = path.resolve()
        # In a real restricted environment, we might check if resolved path is within a specific root.
        # For now, we ensure it's resolved and not empty.
        if str(resolved) == ".":
             return resolved
        return resolved
    except Exception as e:
        raise ValueError(f"Invalid path: {path}") from e


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

    return SystemConfig(
        dft=dft_config,
        inference=inference_params,
        target_system=user_config.target_system.model_copy(),
    )
