from typing import Any

from mlip_autopipec.infrastructure import MockDynamics, MockGenerator, MockOracle, MockTrainer
from mlip_autopipec.interfaces import BaseDynamics, BaseGenerator, BaseOracle, BaseTrainer


def create_generator(config: dict[str, Any]) -> BaseGenerator:
    typ = config.get("type")
    if typ == "mock":
        return MockGenerator(**config)
    msg = f"Unknown generator type: {typ}"
    raise ValueError(msg)

def create_oracle(config: dict[str, Any]) -> BaseOracle:
    typ = config.get("type")
    if typ == "mock":
        return MockOracle(**config)
    msg = f"Unknown oracle type: {typ}"
    raise ValueError(msg)

def create_trainer(config: dict[str, Any]) -> BaseTrainer:
    typ = config.get("type")
    if typ == "mock":
        return MockTrainer(**config)
    msg = f"Unknown trainer type: {typ}"
    raise ValueError(msg)

def create_dynamics(config: dict[str, Any]) -> BaseDynamics:
    typ = config.get("type")
    if typ == "mock":
        return MockDynamics(**config)
    msg = f"Unknown dynamics type: {typ}"
    raise ValueError(msg)
