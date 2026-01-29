from pathlib import Path
from typing import Any

import yaml

from mlip_autopipec.domain_models.workflow import WorkflowState


def load_yaml(path: Path | str) -> dict[str, Any]:
    """
    Safely load a YAML file.

    WARNING: This function loads the entire file into memory.
    It is intended for configuration files, not large datasets.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is invalid.
        TypeError: If the YAML content is not a dictionary.
    """
    path = Path(path)
    if not path.exists():
        msg = f"YAML file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        msg = f"YAML file {path} must contain a dictionary, got {type(data)}"
        raise TypeError(msg)

    return data


def dump_yaml(data: Any, path: Path | str) -> None:
    """
    Dump data to a YAML file using safe dumper.

    Args:
        data: Data to dump.
        path: Path to the output file.
    """
    path = Path(path)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_state(state: WorkflowState, path: Path | str) -> None:
    """
    Persist the workflow state to a JSON file.

    Args:
        state: The WorkflowState object.
        path: Path to the output JSON file.
    """
    path = Path(path)
    with path.open("w") as f:
        f.write(state.model_dump_json(indent=2))


def load_state(path: Path | str) -> WorkflowState:
    """
    Load the workflow state from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The loaded WorkflowState object.
    """
    path = Path(path)
    if not path.exists():
        msg = f"State file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        json_str = f.read()
    return WorkflowState.model_validate_json(json_str)
