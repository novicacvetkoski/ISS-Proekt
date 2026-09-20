"""
Persona registry — YAML profiles are the source of truth, memorizz provides
storage, versioning and a stable persona_id for provenance.

Two constraints are enforced here rather than trusted, both from reading memorizz's
source (see docs/HANDOFF.md §2.4):

1. Personas are built with Persona.from_dict(), never Persona(...). The constructor
   APPENDS your text onto generic RoleType defaults — for an unrecognised role like
   "Juror" that injects "Provide versatile support across various domains." into a
   juror's goals — and calls an embedding provider at construction time. from_dict
   bypasses both ("would corrupt a round-trip", per memorizz's own docstring).

2. Persona EVOLUTION IS DISABLED. memorizz supports self-updating personas with a
   versioned evolution_history. A persona that mutated between cases would leak
   information across the 100 trials and destroy their independence, so version must
   stay 1 and evolution_history must stay empty. Asserted at load.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).parent / "profiles"
MEMORY_ROOT = Path(__file__).resolve().parents[2] / ".memorizz"


@dataclass(frozen=True)
class JurorProfile:
    """A persona as the rest of the system sees it. Immutable by construction."""
    id: str
    name: str
    role: str
    version: int
    dimensions: dict
    goals: str
    background: str
    conduct: str
    persona_id: str | None = None   # memorizz id, when mirrored

    @property
    def short_id(self) -> str:
        """Stable, compact juror tag used in claim ids: 'textualist' -> 'JTEX'."""
        return "J" + "".join(w[0] for w in self.id.split("_")).upper()[:3]


def load_profiles(ids: list[str] | None = None) -> list[JurorProfile]:
    """Load persona YAML from disk. Order follows `ids` when given."""
    paths = sorted(PROFILE_DIR.glob("*.yaml"))
    by_id: dict[str, JurorProfile] = {}
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile = JurorProfile(
            id=data["id"],
            name=data["name"],
            role=data.get("role", "Juror"),
            version=int(data.get("version", 1)),
            dimensions=data.get("dimensions", {}),
            goals=data["goals"].strip(),
            background=data["background"].strip(),
            conduct=data.get("conduct", "").strip(),
        )
        if profile.version != 1:
            raise ValueError(
                f"Persona {profile.id!r} is at version {profile.version}. Personas must stay "
                "frozen at version 1 for the duration of an experiment — a mutating persona "
                "leaks information across cases. Start a new experiment instead."
            )
        by_id[profile.id] = profile

    if ids is None:
        return [by_id[k] for k in sorted(by_id)]

    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"Unknown persona id(s): {missing}. Available: {sorted(by_id)}")
    return [by_id[i] for i in ids]


def mirror_to_memorizz(profiles: list[JurorProfile], root: Path | None = None) -> list[JurorProfile]:
    """
    Store profiles in memorizz and return copies carrying their persona_id.

    Optional: the deliberation pipeline runs from JurorProfile alone. This exists so
    persona identity is versioned and auditable outside the repo, and so each run
    artifact can cite the persona_id it actually used.
    """
    from memorizz.long_term.semantic.persona import Persona
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    root = root or MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    provider = FileSystemProvider(FileSystemConfig(root_path=root, lazy_vector_indexes=True))

    stored: list[JurorProfile] = []
    for profile in profiles:
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
        if persona.version != 1 or persona.evolution_history:
            raise RuntimeError(f"Persona {profile.id!r} came back evolved; refusing to run.")
        storage_id = persona.store_persona(provider)
        stored.append(
            JurorProfile(
                id=profile.id,
                name=profile.name,
                role=profile.role,
                version=profile.version,
                dimensions=profile.dimensions,
                goals=profile.goals,
                background=profile.background,
                conduct=profile.conduct,
                persona_id=str(storage_id),
            )
        )
    return stored
