"""
The deliberation graph.

    load → [Send x5] private_review → clerk_agenda → assign_persuader? →
    [Send x5] deliberation_round ⟲ tally/stopping → clerk_verdict → end

Two structural choices worth defending:

SIMULTANEOUS ROUNDS. All five jurors respond to the same snapshot in parallel rather
than speaking in turn. This removes speaking-order effects — a confound in the closest
prior work, which used an LLM-driven speaker selector — and cuts wall-clock ~5x. It is a
deviation from real juries, documented in SYSTEM_DESIGN §4.2.

ROUND 0 IS SHARED. Private review is condition-independent, so the runner computes it
once and both conditions resume from it. That halves the paired design's cost and makes
the pairing exact rather than approximate.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..config import ExperimentConfig
from ..domain.models import CaseFile, JuryVerdict, Position, Usage
from ..domain.verdicts import majority_verdict
from ..jurors.clerk import Clerk
from ..jurors.juror import Juror, position_from_assessment, position_from_statement
from ..jurors.persuader import assign_persuader, build_overlay
from .digest import build_digest, namespaced_claim_id, vote_tally
from .rules import final_votes, should_stop
from .state import JuryState


class DeliberationEngine:
    """Owns the jurors, the clerk, and the compiled graph for one experiment."""

    def __init__(
        self,
        jurors: list[Juror],
        clerk: Clerk,
        cfg: ExperimentConfig,
        case_store: dict[str, CaseFile],
    ):
        self.jurors = {j.juror_id: j for j in jurors}
        self.clerk = clerk
        self.cfg = cfg
        self.case_store = case_store
        self.short_ids = {j.juror_id: j.short_id for j in jurors}
        self.graph = self._build()

    # -- helpers ---------------------------------------------------------- #

    def _case(self, state: JuryState) -> CaseFile:
        return self.case_store[state["case_id"]]

    def _namespace_claims(self, position: Position) -> Position:
        """Rewrite bare claim ids ('c1') into namespaced ones ('JTEX-R1-c1')."""
        short = self.short_ids.get(position.juror_id, "J??")
        for claim in position.claims:
            if not claim.claim_id.startswith(short):
                claim.claim_id = namespaced_claim_id(short, position.round, claim.claim_id)
        return position

    # -- nodes ------------------------------------------------------------ #

    def _entry(self, state: JuryState) -> list[Send] | str:
        """
        Entry routing — this is what makes round-0 sharing real.

        When the runner has replayed a cached round 0 into the state, the panel must NOT
        review the case again: the two conditions have to start from identical juror
        positions or the pairing is approximate rather than exact (and we pay twice).
        So a state that already carries round-0 positions skips straight to the point
        where the conditions diverge.
        """
        has_round0 = any(p.round == 0 for p in state.get("positions", []))
        if not has_round0:
            return [Send("private_review", {**state, "_juror_id": jid}) for jid in self.jurors]
        if not state.get("agenda"):
            return "clerk_agenda"
        return "assign_persuader" if state["condition"] == "treatment" else "round_start"

    def private_review(self, state: dict) -> dict:
        juror = self.jurors[state["_juror_id"]]
        case = self.case_store[state["case_id"]]
        assessment, usage = juror.review_case(case)
        position = self._namespace_claims(
            position_from_assessment(juror.profile, assessment, usage)
        )
        return {
            "assessments": [(juror.juror_id, assessment)],
            "positions": [position],
            "usages": [usage],
        }

    def clerk_agenda(self, state: JuryState) -> dict:
        case = self._case(state)
        issues, usage = self.clerk.build_agenda(
            case_prefix=f"CASE MATERIALS — {case.case_id}\n\n{case.render()}",
            assessments=state["assessments"],
        )
        return {"agenda": issues, "usages": [usage], "round": 0}

    def assign_persuader_node(self, state: JuryState) -> dict:
        """Treatment only. Runs after round 0 so the target can oppose the actual majority."""
        round0 = [p for p in state["positions"] if p.round == 0]
        assignment = assign_persuader(round0, seed=self.cfg.seed, case_id=state["case_id"])
        return {"persuader": assignment}

    def _fan_out_round(self, state: JuryState) -> list[Send]:
        return [Send("deliberation_round", {**state, "_juror_id": jid}) for jid in self.jurors]

    def deliberation_round(self, state: dict) -> dict:
        juror = self.jurors[state["_juror_id"]]
        case = self.case_store[state["case_id"]]
        round_num = state["round"] + 1
        positions = state["positions"]

        own = max(
            (p for p in positions if p.juror_id == juror.juror_id),
            key=lambda p: p.round,
        )
        digest = build_digest(positions, self.short_ids, juror.juror_id, self.cfg.deliberation)

        persuader = state.get("persuader")
        is_persuader = bool(persuader and persuader.juror_id == juror.juror_id)
        overlay = build_overlay(persuader.target_verdict) if is_persuader else None

        statement, usage = juror.deliberate(
            case=case,
            round_num=round_num,
            own_position=own,
            digest=digest,
            agenda=state.get("agenda", []),
            tally=vote_tally(positions),
            overlay=overlay,
        )
        position = self._namespace_claims(
            position_from_statement(juror.profile, statement, round_num, usage, is_persuader)
        )
        return {"positions": [position], "usages": [usage]}

    def tally(self, state: JuryState) -> dict:
        round_num = state["round"] + 1
        stop, reason = should_stop(round_num, state["positions"], self.cfg.deliberation)
        return {"round": round_num, "done": stop, "stopped_because": reason}

    def clerk_verdict(self, state: JuryState) -> dict:
        case = self._case(state)
        votes = final_votes(state["positions"])
        verdict, unanimous = majority_verdict(list(votes.values()))

        persuader = state.get("persuader")
        non_persuader_verdict = None
        if persuader:
            others = [v for jid, v in votes.items() if jid != persuader.juror_id]
            non_persuader_verdict, _ = majority_verdict(others)

        latest = {p.juror_id: p for p in sorted(state["positions"], key=lambda p: p.round)}
        written, usage = self.clerk.write_verdict(
            case_prefix=f"CASE MATERIALS — {case.case_id}\n\n{case.render()}",
            verdict=verdict,
            vote_breakdown=votes,
            final_positions=list(latest.values()),
        )
        return {
            "verdict": JuryVerdict(
                verdict=verdict,
                unanimous=unanimous,
                vote_breakdown=votes,
                rationale=written.rationale,
                dissent_summary=written.dissent_summary,
                non_persuader_verdict=non_persuader_verdict,
            ),
            "usages": [usage],
        }

    # -- wiring ----------------------------------------------------------- #

    def _route_after_agenda(self, state: JuryState) -> str:
        return "assign_persuader" if state["condition"] == "treatment" else "deliberate"

    def _route_after_tally(self, state: JuryState) -> str:
        return "clerk_verdict" if state["done"] else "deliberate"

    def _build(self):
        g = StateGraph(JuryState)

        g.add_node("private_review", self.private_review)
        g.add_node("clerk_agenda", self.clerk_agenda)
        g.add_node("assign_persuader", self.assign_persuader_node)
        # No-op join point. Every path into a deliberation round goes through here, so
        # the five-way fan-out is defined once and the loop-back reuses it.
        g.add_node("round_start", lambda state: {})
        g.add_node("deliberation_round", self.deliberation_round)
        g.add_node("tally", self.tally)
        g.add_node("clerk_verdict", self.clerk_verdict)

        g.add_conditional_edges(
            START,
            self._entry,
            ["private_review", "clerk_agenda", "assign_persuader", "round_start"],
        )
        g.add_edge("private_review", "clerk_agenda")
        g.add_conditional_edges(
            "clerk_agenda",
            self._route_after_agenda,
            {"assign_persuader": "assign_persuader", "deliberate": "round_start"},
        )
        g.add_edge("assign_persuader", "round_start")
        g.add_conditional_edges("round_start", self._fan_out_round, ["deliberation_round"])
        # Fan-in: tally runs once, after all five jurors have spoken.
        g.add_edge("deliberation_round", "tally")
        g.add_conditional_edges(
            "tally",
            self._route_after_tally,
            {"clerk_verdict": "clerk_verdict", "deliberate": "round_start"},
        )
        g.add_edge("clerk_verdict", END)

        return g.compile()

    def run(self, state: JuryState) -> JuryState:
        # recursion_limit must cover: fan-outs + rounds + clerk calls.
        limit = 10 + self.cfg.deliberation.max_rounds * (len(self.jurors) + 3)
        return self.graph.invoke(state, config={"recursion_limit": limit})


def total_usage(usages: list[Usage]) -> Usage:
    if not usages:
        return Usage()
    return Usage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cached_tokens=sum(u.cached_tokens for u in usages),
        model=usages[0].model,
    )
