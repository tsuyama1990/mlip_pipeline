from mlip_autopipec.components.base_component import BaseComponent


class MockGenerator(BaseComponent):
    """
    Mock implementation of a Structure Generator.
    """


class MockOracle(BaseComponent):
    """
    Mock implementation of an Oracle (DFT/Property Calculator).
    """


class MockTrainer(BaseComponent):
    """
    Mock implementation of a Trainer (Machine Learning Potential).
    """


class MockDynamics(BaseComponent):
    """
    Mock implementation of a Dynamics Engine (MD/MC).
    """
