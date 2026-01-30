from pathlib import Path

DEFAULT_PROJECT_NAME = "MyMLIPProject"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILENAME = Path("mlip_pipeline.log")
DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_ELEMENTS = ["Si"]
DEFAULT_CUTOFF = 5.0
DEFAULT_SEED = 42

# Atomic masses for input generation
# Simplistic mapping or use ase.data.atomic_masses dynamically?
# Auditor wants decoupling from ASE in input_gen core logic if possible.
# But keeping a huge table here is redundant.
# We will keep a small set or load it.
# Actually, using a constant dict here is better than importing ase inside the function.
ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "As": 74.922,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.798,
    # Add more as needed
}
