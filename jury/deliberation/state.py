"""
Graph state.

The case TEXT is deliberately NOT in here. LangGraph serialises state at every
checkpoint, and a full ILDC case averages ~3.2k tokens; carrying it through every
superstep would bloat checkpoints for no benefit. The state carries `case_id` and the
CaseStore resolves it — see docs/SYSTEM_DESIGN.md §4.3.

`positions` and `usages` use additive reducers because the five jurors write to them
concurrently from a Send() fan-out; without a reducer, parallel branches conflict.
"""

import operator
from typing import Annotated, TypedDict

from ..domain.models import (
    JuryVerdict,
    PersuaderAssignment,
    Position,
    PrivateAssessment,
    Usage,
)


class JuryState(TypedDict, total=False):
    # identity
    case_id: str
    condition: str                 # "control" | "treatment" | "sway"

    # round 0
    assessments: Annotated[list[tuple[str, PrivateAssessment]], operator.add]
    agenda: list[str]

    # deliberation
    positions: Annotated[list[Position], operator.add]
    round: int
    persuader: PersuaderAssignment | None

    # termination
    done: bool
    stopped_because: str

    # output
    verdict: JuryVerdict | None

    # accounting
    usages: Annotated[list[Usage], operator.add]


def empty_state(case_id: str, condition: str) -> JuryState:
    return JuryState(
        case_id=case_id,
        condition=condition,
        assessments=[],
        agenda=[],
        positions=[],
        round=0,
        persuader=None,
        done=False,
        stopped_because="",
        verdict=None,
        usages=[],
    )
