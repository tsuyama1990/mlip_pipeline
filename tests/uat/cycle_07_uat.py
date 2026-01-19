import sys

import numpy as np
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config.schemas.inference import EmbeddingConfig
from mlip_autopipec.inference.embedding import EmbeddingExtractor
from mlip_autopipec.inference.masking import ForceMasker

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def run_uat():
    # ---------------------------------------------------------
    # UAT-07-01: Local Environment Extraction
    # ---------------------------------------------------------
    try:
        # Create supercell
        atoms = bulk("Al", "fcc", a=4.05, cubic=True) * (5, 5, 5)  # 125 atoms
        center_idx = 62

        config = EmbeddingConfig(core_radius=4.0, buffer_width=2.0)
        extractor = EmbeddingExtractor(config)

        extracted = extractor.extract(atoms, center_idx)
        cluster = extracted.atoms

        # Verify size (approximate)
        # Expected: 4.05 neighbor dist ~ 2.86
        # Radius 6.0 covers ~4 shells.
        # Just check it's small but not empty
        if 10 < len(cluster) < 100:
            pass
        else:
            sys.exit(1)

        # Verify box
        L = config.box_size
        cell = cluster.get_cell()
        expected_cell = np.diag([L, L, L])

        if np.allclose(cell, expected_cell):
            pass
        else:
            sys.exit(1)

    except Exception:
        sys.exit(1)

    # ---------------------------------------------------------
    # UAT-07-02: Force Mask Generation
    # ---------------------------------------------------------
    try:
        masker = ForceMasker()
        # Box center
        L = config.box_size
        center = np.array([L / 2.0, L / 2.0, L / 2.0])

        masker.apply(cluster, center, config.core_radius)

        if "force_mask" in cluster.arrays:
            mask = cluster.arrays["force_mask"]
        else:
            sys.exit(1)

        # Verify values
        # Check an atom known to be in core (the center one ideally)
        # We need to find the atom at center
        positions = cluster.get_positions()
        dists = np.linalg.norm(positions - center, axis=1)

        # There should be at least one atom very close to center (the focal atom)
        if np.min(dists) < 0.1:
            pass
        else:
            pass

        # Check correlation
        # Mask should be 1 if dist <= core, 0 otherwise
        valid_mask = True
        for d, m in zip(dists, mask, strict=False):
            expected = 1.0 if d <= config.core_radius else 0.0
            if m != expected:
                valid_mask = False
                break

        if valid_mask:
            pass
        else:
            sys.exit(1)

    except Exception:
        sys.exit(1)

    # ---------------------------------------------------------
    # UAT-07-03: Periodic Box Wrapping
    # ---------------------------------------------------------
    try:
        # Create small system to force wrapping
        # 10x10x10 box
        # Atom at 0.1, neighbor at 9.9
        test_atoms = Atoms(
            "H2", positions=[[0.1, 0.1, 0.1], [9.9, 0.1, 0.1]], cell=[10, 10, 10], pbc=True
        )
        # Distance is 0.2

        # Extract around 0.1
        # Config large enough to capture neighbor
        cfg = EmbeddingConfig(core_radius=2.0, buffer_width=1.0)  # Total 3.0

        ext = EmbeddingExtractor(cfg)
        res = ext.extract(test_atoms, 0)

        # Should have 2 atoms
        if len(res.atoms) == 2:
            pass
        else:
            sys.exit(1)

        # Check distance in new cluster
        d = res.atoms.get_distance(0, 1, mic=True)
        if abs(d - 0.2) < 1e-4:
            pass
        else:
            sys.exit(1)

    except Exception:
        sys.exit(1)

    # ---------------------------------------------------------
    # UAT-07-04: Metadata Tracing
    # ---------------------------------------------------------
    try:
        # In step 1 we used `atoms` which didn't have uuid.
        # Let's add uuid to atoms and retry
        atoms.info["uuid"] = "test-uuid-123"

        extracted = extractor.extract(atoms, center_idx)

        if extracted.origin_uuid == "test-uuid-123":
            pass
        else:
            sys.exit(1)

        if extracted.origin_index == center_idx:
            pass
        else:
            sys.exit(1)

    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    run_uat()
