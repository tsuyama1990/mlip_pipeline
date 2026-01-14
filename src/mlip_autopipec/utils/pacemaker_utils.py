from mlip_autopipec.schemas.user_config import TrainerConfig


def generate_pacemaker_input(config: TrainerConfig, dataset_path: str) -> str:
    """
    Generates the content for the pacemaker.in file.

    Args:
        config: The trainer configuration.
        dataset_path: The path to the training dataset.

    Returns:
        The content of the pacemaker.in file as a string.
    """
    return f"""[calculator]
calculator = "ace"
potential_file = "al_mg_si_ace.json"
param_file = "al_mg_si_ace.json"

[fit]
dataset_path = "{dataset_path}"
l_max = 2
max_body_order = {config.max_body_order}
radial_basis = "{config.radial_basis}"
loss_weights = {config.loss_weights}
delta_learning = True
# Other fitting parameters...
"""
