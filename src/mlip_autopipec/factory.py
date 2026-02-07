from typing import Any, Dict
from mlip_autopipec.interfaces import BaseGenerator, BaseOracle, BaseTrainer, BaseDynamics
from mlip_autopipec.infrastructure import MockGenerator, MockOracle, MockTrainer, MockDynamics

def create_generator(config: Dict[str, Any]) -> BaseGenerator:
    typ = config.get("type")
    if typ == "mock":
        return MockGenerator(**config)
    msg = f"Unknown generator type: {typ}"
    raise ValueError(msg)

def create_oracle(config: Dict[str, Any]) -> BaseOracle:
    typ = config.get("type")
    if typ == "mock":
        return MockOracle(**config)
    msg = f"Unknown oracle type: {typ}"
    raise ValueError(msg)

def create_trainer(config: Dict[str, Any]) -> BaseTrainer:
    typ = config.get("type")
    if typ == "mock":
        return MockTrainer(**config)
    msg = f"Unknown trainer type: {typ}"
    raise ValueError(msg)

def create_dynamics(config: Dict[str, Any]) -> BaseDynamics:
    typ = config.get("type")
    if typ == "mock":
        return MockDynamics(**config)
    msg = f"Unknown dynamics type: {typ}"
    raise ValueError(msg)
