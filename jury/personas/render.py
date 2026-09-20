"""
Rendering a persona into the prompt.

The persona card is the VOLATILE-adjacent part of the prompt, not the shared prefix:
prompt layout is [court instructions][case file][persona card][task] so that the
large prefix stays byte-identical across all five jurors and Gemini's implicit cache
hits. See docs/SYSTEM_DESIGN.md §4.3.
"""

from .registry import JurorProfile

_DIMENSION_LABELS = {
    "orientation": "Legal reasoning",
    "deliberation_style": "Deliberation style",
    "attitudinal_lean": "Disposition",
    "need_for_cognition": "Engagement with argument",
    "conformity_susceptibility": "Susceptibility to group pressure",
    "confidence_calibration": "Confidence",
}


def render_persona_card(profile: JurorProfile) -> str:
    """The persona block a juror sees. Mirrors memorizz's own phrasing for continuity."""
    lines = [
        f"YOU ARE: {profile.name}. You are a {profile.role} on this panel.",
        "",
        f"Your goals: {profile.goals}",
        "",
        f"Your background: {profile.background}",
    ]
    if profile.conduct:
        lines += ["", f"How you conduct yourself: {profile.conduct}"]

    dims = [
        f"  - {_DIMENSION_LABELS.get(k, k)}: {v}"
        for k, v in profile.dimensions.items()
        if k in _DIMENSION_LABELS
    ]
    if dims:
        lines += ["", "Your disposition as a juror:", *dims]

    lines += [
        "",
        "Stay recognisably yourself. Where you agree with another juror's conclusion, reach it "
        "through your own reasoning rather than restating theirs.",
    ]
    return "\n".join(lines)


def render_from_memorizz(profile: JurorProfile) -> str:
    """
    Alternative renderer that delegates the identity sentence to memorizz's own
    generate_system_prompt_input(). Kept so the memorizz persona record stays the
    single source of identity if we later enable its history features.
    """
    from memorizz.long_term.semantic.persona import Persona

    persona = Persona.from_dict(
        {
            "persona_id": profile.id,
            "name": profile.name,
            "role": profile.role,
            "goals": profile.goals,
            "background": profile.background,
            "version": profile.version,
            "evolution_history": [],
            "embedding": None,
        }
    )
    base = persona.generate_system_prompt_input(include_history=False)
    extra = render_persona_card(profile).split("Your background:", 1)[-1]
    return f"{base}\n\nYour background:{extra}"
