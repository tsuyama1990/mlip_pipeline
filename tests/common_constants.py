import numpy as np

# Constants for test configuration
MAX_CYCLES = 2
N_STRUCTURES = 5
CELL_SIZE = 10.0
N_ATOMS = 2
ATOMIC_NUMBERS = [1, 1]
SELECTION_RATE = 1.0
UNCERTAINTY_THRESHOLD = 5.0
CYCLE_01_DIR = "cycle_01"
CYCLE_02_DIR = "cycle_02"
CYCLE_03_DIR = "cycle_03"
POTENTIAL_FILE = "potential.yace"
DATASET_FILE = "dataset.jsonl"
STATE_FILE = "workflow_state.json"
STOPPED_STATUS = "STOPPED"
ERROR_STATUS = "ERROR"

# Constants for Structure tests
DUMMY_POSITIONS = np.zeros((2, 3))
DUMMY_ATOMIC_NUMBERS = np.array([1, 1])
DUMMY_CELL = np.eye(3)
DUMMY_PBC = np.array([True, True, True])
VOIGT_STRESS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
EXPECTED_TENSOR_STRESS = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])

# Constants for Oracle tests
ORACLE_TOLERANCE = 1e-10
