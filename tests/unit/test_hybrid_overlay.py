
from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.enums import DynamicsType, HybridPotentialType
from mlip_autopipec.dynamics.hybrid_overlay import HybridOverlay


def test_hybrid_overlay_none() -> None:
    config = DynamicsConfig(
        type=DynamicsType.LAMMPS,
        hybrid_potential=HybridPotentialType.NONE,
    )
    overlay = HybridOverlay(config)

    # When None, we expect standard pace pair style
    style = overlay.get_pair_style()
    assert "hybrid/overlay" not in style
    assert "pace" in style

    coeff = overlay.get_pair_coeff(["Fe", "Pt"], "potential.yace")
    assert "pair_coeff * * pace potential.yace Fe Pt" in coeff
    assert "zbl" not in coeff


def test_hybrid_overlay_zbl() -> None:
    config = DynamicsConfig(
        type=DynamicsType.LAMMPS,
        hybrid_potential=HybridPotentialType.ZBL,
        zbl_cut_inner=0.5,
        zbl_cut_outer=1.2,
    )
    overlay = HybridOverlay(config)

    style = overlay.get_pair_style()
    assert "pair_style hybrid/overlay" in style
    assert "pace" in style
    assert "zbl 0.5 1.2" in style

    coeff = overlay.get_pair_coeff(["Fe", "Pt"], "potential.yace")
    # Must contain both pace and zbl lines
    assert "pair_coeff * * pace potential.yace Fe Pt" in coeff
    # Check ZBL specific coefficients using atomic numbers (Fe=26, Pt=78)
    # The implementation should handle looking up atomic numbers
    assert "pair_coeff 1 1 zbl 26 26" in coeff
    assert "pair_coeff 1 2 zbl 26 78" in coeff
    assert "pair_coeff 2 2 zbl 78 78" in coeff


def test_hybrid_overlay_lj() -> None:
    config = DynamicsConfig(
        type=DynamicsType.LAMMPS,
        hybrid_potential=HybridPotentialType.LJ,
        lj_epsilon=1.0,
        lj_sigma=2.0,
        lj_cutoff=3.0,
    )
    overlay = HybridOverlay(config)

    style = overlay.get_pair_style()
    assert "pair_style hybrid/overlay" in style
    assert "pace" in style
    assert "lj/cut 3.0" in style

    coeff = overlay.get_pair_coeff(["Fe", "Pt"], "potential.yace")
    assert "pair_coeff * * pace potential.yace Fe Pt" in coeff
    assert "pair_coeff * * lj/cut 1.0 2.0" in coeff
