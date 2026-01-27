
class CandidateManager:
    """
    Manages candidate data persistence logic.
    Separates business logic (defaults, validation) from raw DB access.
    """
    def __init__(self, db_manager):
        self.db = db_manager

    def create_candidate(self, atoms, metadata: dict = None):
        from ase import Atoms
        if not isinstance(atoms, Atoms):
            msg = f"Expected ase.Atoms object, got {type(atoms)}"
            raise TypeError(msg)

        if metadata is None:
            metadata = {}
        # Apply defaults
        meta = metadata.copy()
        if "status" not in meta:
            meta["status"] = "pending"
        if "generation" not in meta:
            meta["generation"] = 0
        if "config_type" not in meta:
            meta["config_type"] = "candidate"

        self.db.add_structure(atoms, meta)
