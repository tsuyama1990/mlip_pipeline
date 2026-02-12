import sys
import shutil
from pathlib import Path
from ase import Atoms

# Add src to path
sys.path.append("src")

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig

def verify_random_generation(tmp_path: Path):
    print("--- Verifying Random Generation ---")
    tmp_path.mkdir(parents=True, exist_ok=True)
    seed_path = tmp_path / "seed_random.xyz"
    Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True).write(seed_path)

    config_dict = dict(
        orchestrator=dict(work_dir=tmp_path, max_cycles=1, execution_mode="mock"),
        generator=dict(
            type="random",
            seed_structure_path=str(seed_path),
            mock_count=5,
            policy=dict(
                strategy="random",
                strain_range=0.05
            )
        ),
        oracle=dict(type="mock"),
        trainer=dict(type="mock"),
        dynamics=dict(type="mock"),
        validator=dict(type="mock")
    )
    # Validate config
    config = GlobalConfig.model_validate(config_dict)

    orch = Orchestrator(config)
    # Access generator directly or simulate exploration
    candidates = list(orch.generator.explore({"count": 5}))

    assert len(candidates) == 5
    assert candidates[0].provenance == "random"
    print("✓ Random Generation Produced 5 candidates with correct provenance.")

def verify_adaptive_schedule(tmp_path: Path):
    print("--- Verifying Adaptive Schedule ---")
    tmp_path.mkdir(parents=True, exist_ok=True)
    seed_path = tmp_path / "seed_adaptive.xyz"
    Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True).write(seed_path)

    policy = dict(
        strategy="adaptive",
        temperature_schedule=[100.0, 500.0, 1000.0]
    )

    config_dict = dict(
        orchestrator=dict(work_dir=tmp_path, max_cycles=3, execution_mode="mock"),
        generator=dict(
            type="adaptive",
            seed_structure_path=str(seed_path),
            policy=policy,
            mock_count=1
        ),
        oracle=dict(type="mock"),
        trainer=dict(type="mock"),
        dynamics=dict(type="mock"),
        validator=dict(type="mock")
    )
    config = GlobalConfig.model_validate(config_dict)

    orch = Orchestrator(config)

    # Cycle 0
    c0 = list(orch.generator.explore({"cycle": 0, "count": 1}))
    print(f"Cycle 0 Provenance: {c0[0].provenance}")
    assert "md_100.0K" in c0[0].provenance

    # Cycle 1
    c1 = list(orch.generator.explore({"cycle": 1, "count": 1}))
    print(f"Cycle 1 Provenance: {c1[0].provenance}")
    assert "md_500.0K" in c1[0].provenance

    # Cycle 2
    c2 = list(orch.generator.explore({"cycle": 2, "count": 1}))
    print(f"Cycle 2 Provenance: {c2[0].provenance}")
    assert "md_1000.0K" in c2[0].provenance

    print("✓ Adaptive Schedule verified.")

if __name__ == "__main__":
    base_tmp = Path("tests/uat/tmp_cycle02")
    if base_tmp.exists():
        shutil.rmtree(base_tmp)
    base_tmp.mkdir(parents=True, exist_ok=True)

    try:
        verify_random_generation(base_tmp / "random")
        verify_adaptive_schedule(base_tmp / "adaptive")
        print("\nAll Cycle 02 Verification Scenarios Passed!")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
