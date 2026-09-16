"""Domain packs for the live copilot: everything that is specific to one kind of conversation lives here.

A pack is: the system prompt and JSON view schema the writer maintains every tick; the hold/tier guard applied in code;
which context documents the user can supply before the session (job description, CV, ...); whether reference lookups
and lexicon-based mishearing flags apply; and the panel layout the page renders from (so the UI needs no per-pack code).

  PACKS["clinical"]   the in-visit GP decision-support view from demo/livesynth.py (NHS/patient.info lookups, medical lexicon)

Other conversation types plug in as further Pack entries: a system prompt, a panel layout and optional hold rules.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from demo import livesynth as LS

Panel = dict  # {"key","title","type": "text"|"list"|"objlist","label","sub","badge","zone": "top"|"left"|"right"|"wide"}


@dataclass
class Pack:
    name: str
    title: str
    lede: str
    system: str
    panels: list[Panel]
    context_fields: list[dict] = field(default_factory=list)   # [{"key","label","placeholder"}]
    hold_s: float = 90.0
    hold: Callable[[dict, float], dict] | None = None
    lookups: bool = False            # clinical NHS/patient.info lookups
    mishearings: bool = False        # medical-lexicon mishearing flags
    activity_lookup_label: str = ""
    max_tokens: int = 1300

    def synthesize(self, text: str, prev: dict | None, upto_t: float, model: str, refs: dict | None, context: dict | None, think: bool = False) -> dict:
        if self.name == "clinical":
            return LS.synthesize_once(text, prev, upto_t, model, refs, think=think)
        ctx = ""
        for f in self.context_fields:
            v = (context or {}).get(f["key"]) or ""
            if v.strip():
                ctx += f"\n\n{f['label'].upper()}:\n{v.strip()[:4000]}"
        # stable prefix first (context docs, then the append-only transcript) so the server's prefix cache covers most of the prompt; the previous view last
        user = (ctx.lstrip() + f"\n\nTRANSCRIPT SO FAR ({upto_t:.0f} s into the conversation; no speaker labels, the recogniser may mishear names and product terms):\n{text}"
                + f"\n\nPREVIOUS VIEW (update it; replace next_probe if the transcript now answers it):\n{json.dumps(prev) if prev else 'none yet'}"
                + "\n\nUpdate the view now. Output one JSON object only.")
        raw = LS._llm(self.system, user, max_tokens=3000 if think else self.max_tokens, model=model, think=think)
        js = LS.parse_json(raw)
        if js is None:  # truncated or malformed JSON: keep the previous view but say so, never pretend it was updated
            js = dict(prev or {}); js["_stale"] = f"model output not parseable ({len(raw)} chars); previous view kept"
        else:
            js.pop("_stale", None)
        return self.hold(js, upto_t) if self.hold else js


# ---------------------------------------------------------------------------------------------------------------------
# Clinical pack: wraps the existing view. Panels mirror demo/live.html.
CLINICAL_PANELS = [
    {"key": "complaint", "title": "", "type": "headline", "zone": "top"},
    {"key": "differential", "title": "Differential", "type": "cards", "label": "dx", "sub": "likelihood", "flag": "time_critical", "flag_text": "time-critical · exclude", "lists": ["evidence"], "note": "missing", "note_prefix": "to confirm/exclude: ", "zone": "wide"},
    {"key": "next_question", "title": "Next question", "type": "callout", "zone": "wide"},
    {"key": "red_flags", "title": "Red flags", "type": "objlist", "label": "feature", "badge": "tier", "badge_hot": "act now", "why": "why", "arrow": "action", "zone": "wide"},
    {"key": "plan_stated", "title": "Plan stated", "type": "list", "zone": "left"},
    {"key": "plan_suggested", "title": "Plan suggested", "type": "objlist", "label": "item", "sub": "basis", "zone": "right"},
    {"key": "safety_netting_stated", "title": "Safety-netting given", "type": "list", "zone": "left"},
    {"key": "safety_netting_suggested", "title": "Safety-netting to add", "type": "list", "zone": "right"},
    {"key": "history", "title": "History", "type": "list", "zone": "left"},
    {"key": "findings", "title": "Findings", "type": "list", "zone": "right"},
    {"key": "assumptions", "title": "Assumptions under test", "type": "objlist", "label": "assumption", "sub": "would_change_if", "sub_prefix": "changes if: ", "zone": "left"},
    {"key": "gaps", "title": "Gaps", "type": "list", "zone": "right"},
    {"key": "terms", "title": "Terms heard", "type": "list", "zone": "wide"},
]

def _as_objlist(v, label: str) -> list:
    """Models sometimes emit {name: {...}} instead of [{name: ..., ...}]; normalise to the list form the UI expects."""
    if isinstance(v, dict):
        return [{label: k, **(x if isinstance(x, dict) else {"value": x})} for k, x in v.items()]
    return [x for x in (v or []) if x is not None] if isinstance(v, list) else []


PACKS: dict[str, Pack] = {
    "clinical": Pack(
        name="clinical", title="Live decision support", lede="In-visit clinical view: differential, next question, tiered red flags, plan stated vs suggested, safety-netting; guideline lookups attached to the next tick.",
        system=LS.SYNTH_SYSTEM, panels=CLINICAL_PANELS, hold=LS.apply_hold, lookups=True, mishearings=True, activity_lookup_label="NHS + patient.info"),
}


def pack_public(p: Pack) -> dict:
    return {"name": p.name, "title": p.title, "lede": p.lede, "panels": p.panels, "context_fields": p.context_fields, "lookups": p.lookups, "mishearings": p.mishearings}


_NONWORD = re.compile(r"[^a-z0-9'-]")
