import numpy as np
from ase import Atoms


class MaceWrapper:
    """
    Wrapper for MACE model to compute energy, forces, and descriptors.
    """

    def __init__(self, model_type: str = "mace_mp"):
        self.model_type = model_type
        self.model = None
        self.device = "cpu"

    def load_model(self, model_path: str, device: str) -> None:
        """
        Loads the MACE model.
        """
        self.device = device
        if self.model_type == "mock":
            return

        if self.model_type == "mace_mp":
            try:
                from mace.calculators import MACECalculator

                # MACECalculator expects model_paths (plural)
                # It handles "medium", "large" etc. if they are mapped, or paths.
                # We assume model_path is passed correctly.
                # We need to map 'cuda' to 'cuda' or 'cpu'.
                self.model = MACECalculator(
                    model_paths=model_path, device=device, default_dtype="float32"
                )
            except ImportError:
                raise ImportError("mace-torch is not installed.")
            except Exception as e:
                raise RuntimeError(f"Failed to load MACE model: {e}")

    def compute_energy_forces(self, atoms_list: list[Atoms]) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Computes energy and forces.
        """
        if self.model_type == "mock":
            N = len(atoms_list)
            energies = np.random.uniform(-5.0, -2.0, size=N)
            forces_list = [np.random.uniform(-0.1, 0.1, size=(len(at), 3)) for at in atoms_list]
            return energies, forces_list

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        energies = []
        forces_list = []

        # TODO: Implement batching for MACE
        for atoms in atoms_list:
            # Copy atoms to avoid side effects and attaching calculator permanently to original objects
            # type: ignore
            at = atoms.copy()
            at.calc = self.model
            energies.append(at.get_potential_energy())
            forces_list.append(at.get_forces())

        return np.array(energies), forces_list

    def compute_descriptors(self, atoms_list: list[Atoms]) -> np.ndarray:
        """
        Computes descriptors using dscribe (SOAP) as a robust proxy for structural similarity.
        """
        if self.model_type == "mock":
            # Return random descriptors
            return np.random.rand(len(atoms_list), 10)

        try:
            from dscribe.descriptors import SOAP  # type: ignore
        except ImportError:
            # Fallback if dscribe missing (should not happen per lock file)
            # Use simple features: Volume + flattened positions (bad but structurally relevant)
            # Or just fail.
            raise ImportError("dscribe is required for descriptors.")

        # Determine species
        species = set()
        for at in atoms_list:
            species.update(at.get_chemical_symbols())
        species = sorted(list(species))

        if not species:
            return np.zeros((len(atoms_list), 1))

        # SOAP Parameters
        soap = SOAP(
            species=species,
            periodic=True,
            rcut=5.0,
            nmax=4,
            lmax=4,
            average="inner",  # Global descriptor (average over sites)
        )

        # dscribe supports batching
        descriptors = soap.create(atoms_list, n_jobs=1)  # type: ignore

        # Ensure it's 2D array
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)

        return descriptors
