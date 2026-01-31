from pathlib import Path

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.workflow import (
    CandidateStatus,
    CandidateStructure,
    WorkflowPhase,
    WorkflowState,
)
import numpy as np


def test_candidate_structure_valid():
    s = Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )
    c = CandidateStructure(
        structure=s,
        origin="test",
        uncertainty_score=0.5,
        status=CandidateStatus.PENDING
    )
    assert c.status == CandidateStatus.PENDING
    assert c.uncertainty_score == 0.5


def test_workflow_state_valid(tmp_path):
    dataset = tmp_path / "data.pckl"
    s = Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )
    cand = CandidateStructure(
        structure=s,
        origin="test",
        uncertainty_score=0.1
    )

    ws = WorkflowState(
        project_name="TestProject",
        generation=1,
        current_phase=WorkflowPhase.SELECTION,
        dataset_path=dataset,
        candidates=[cand]
    )

    assert ws.generation == 1
    assert ws.current_phase == WorkflowPhase.SELECTION
    assert len(ws.candidates) == 1


def test_workflow_state_serialization(tmp_path):
    dataset = Path("data.pckl")
    ws = WorkflowState(
        project_name="TestProject",
        dataset_path=dataset,
        current_phase=WorkflowPhase.EXPLORATION
    )

    json_str = ws.model_dump_json()
    loaded = WorkflowState.model_validate_json(json_str)

    assert loaded.project_name == ws.project_name
    assert loaded.current_phase == ws.current_phase
    assert loaded.dataset_path == dataset
