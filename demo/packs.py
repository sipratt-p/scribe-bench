"""Domain packs for the live copilot: everything that is specific to one kind of conversation lives here.

A pack is: the system prompt and JSON view schema the writer maintains every tick; the hold/tier guard applied in code;
which context documents the user can supply before the session (job description, CV, ...); whether reference lookups
and lexicon-based mishearing flags apply; and the panel layout the page renders from (so the UI needs no per-pack code).

  PACKS["clinical"]   the in-visit GP decision-support view from demo/livesynth.py (NHS/patient.info lookups, medical lexicon)
  PACKS["interview"]  interviewer-side interview assistant: competencies covered, claims to verify, next probe, bias guard
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

# ---------------------------------------------------------------------------------------------------------------------
# Interview pack (interviewer side, disclosed). The view is for the person asking the questions.
INTERVIEW_SYSTEM = """You are a live assistant for the INTERVIEWER during a job interview. The transcript is a live stream from a speech recogniser: no speaker labels, sentences may be cut mid-way, names and product terms may be misheard. Your job is to help the interviewer run a fair, specific, well-covered interview, not to judge the candidate for them. Maintain a running view and output ONE JSON object only, with these keys:

"role": the role being interviewed for, as stated or as given in the job description ("" if unknown);
"stage": one of "intro", "background", "technical", "behavioural", "role and team", "candidate questions", "wrap-up", "unclear";
"summary": 1-2 sentences of where the interview is right now;
"claims": list of up to 8 objects {"claim": what the candidate asserted about their experience, skills or results, "quote": at most 12 words of near-verbatim evidence, "specificity": "specific" (numbers, names, concrete actions) | "vague" | "unverified"};
"competencies": list of objects {"name": competency, "status": "covered" | "partial" | "not yet", "evidence": [at most 2 short paraphrases from the transcript]}. Use the job description's requirements as the competency list when one is given; otherwise use: technical depth, problem solving, ownership and delivery, communication, collaboration, leadership or influence, learning and adaptability. Never mark "covered" without evidence in the transcript;
"next_probe": the single best follow-up question to ask RIGHT NOW, phrased for the interviewer to say aloud. Prefer probes that turn a vague claim into specifics (what exactly did you do, what was the result, what would you do differently) over new topics; "" if nothing is pending;
"probes_suggested": up to 3 other useful questions, each {"question": text, "why": one short reason, "competency": name};
"questions_asked": list of the interviewer's questions so far, paraphrased briefly, in order;
"candidate_questions": questions the candidate asked;
"strengths": list of evidence-backed strengths observed so far (short, each with the evidence in the same string);
"concerns": list of evidence-backed concerns (short, with evidence). Concerns are about the evidence, never about the person;
"inconsistencies": list of {"earlier": what was said before, "now": what was said later, "note": why they do not fit};
"bias_guard": list of objects {"issue": an interviewer question or remark that touches a protected characteristic or is not job-related (age, family plans, health, disability, religion, nationality, pregnancy, marital status, etc.), "quote": the words, "suggestion": a job-related way to get at the underlying need}. Empty when nothing of the kind was said. This is a guard for the interviewer, not an accusation;
"time_check": {"minutes": elapsed minutes as an integer, "note": one short line such as "half the time gone, technical not started" or ""};
"revisions": list of {"was", "now", "because"} for anything you changed since the previous view;
"assumptions": list of {"assumption", "would_change_if"};
"gaps": what the interviewer still needs to find out to make a decision, as short items;
"terms": technical terms, products, companies or names heard that the interviewer may want to note.

Rules. Every tick: list EVERY competency (from the job description, or the default seven) with its current status, even when it is "not yet"; re-derive next_probe from the transcript and drop it the moment the transcript shows it has been answered; add bias_guard items at the tick in which the remark is first heard. Everything in claims, competencies.evidence, questions_asked, strengths, concerns, inconsistencies, bias_guard.quote and terms must come from the transcript; suggestions live only in next_probe, probes_suggested and bias_guard.suggestion. In the first 90 seconds, or while the role and the stage are still unclear, output no concerns and no inconsistencies: let the candidate settle. Be calm and specific; never suggest questions about protected characteristics; never recommend a hiring decision. Prefer updating the previous view to rewriting it; keep earlier items unless contradicted. Re-derive the competency table from the whole transcript every tick. Short phrases."""

INTERVIEW_PANELS = [
    {"key": "summary", "title": "", "type": "headline", "zone": "top", "prefix_key": "role", "prefix_sep": " · ", "suffix_key": "stage"},
    {"key": "next_probe", "title": "Ask next", "type": "callout", "zone": "wide"},
    {"key": "competencies", "title": "Competencies", "type": "cards", "label": "name", "sub": "status", "lists": ["evidence"], "zone": "wide", "sub_status": True},
    {"key": "claims", "title": "Claims to verify", "type": "objlist", "label": "claim", "badge": "specificity", "badge_hot": "vague", "why": "quote", "zone": "wide"},
    {"key": "probes_suggested", "title": "Other probes", "type": "objlist", "label": "question", "sub": "why", "badge": "competency", "zone": "wide"},
    {"key": "strengths", "title": "Strengths (with evidence)", "type": "list", "zone": "left"},
    {"key": "concerns", "title": "Concerns (with evidence)", "type": "list", "zone": "right"},
    {"key": "inconsistencies", "title": "Inconsistencies", "type": "objlist", "label": "now", "why": "earlier", "why_prefix": "earlier: ", "sub": "note", "zone": "wide"},
    {"key": "bias_guard", "title": "Bias guard", "type": "objlist", "label": "issue", "why": "quote", "arrow": "suggestion", "zone": "wide", "hot": True},
    {"key": "questions_asked", "title": "Questions asked", "type": "list", "zone": "left"},
    {"key": "candidate_questions", "title": "Candidate asked", "type": "list", "zone": "right"},
    {"key": "assumptions", "title": "Assumptions under test", "type": "objlist", "label": "assumption", "sub": "would_change_if", "sub_prefix": "changes if: ", "zone": "left"},
    {"key": "gaps", "title": "Still to find out", "type": "list", "zone": "right"},
    {"key": "time_check", "title": "Time", "type": "kv", "label": "minutes", "sub": "note", "zone": "left"},
    {"key": "terms", "title": "Terms and names heard", "type": "list", "zone": "right"},
]


def _as_objlist(v, label: str) -> list:
    """Models sometimes emit {name: {...}} instead of [{name: ..., ...}]; normalise to the list form the UI expects."""
    if isinstance(v, dict):
        return [{label: k, **(x if isinstance(x, dict) else {"value": x})} for k, x in v.items()]
    return [x for x in (v or []) if x is not None] if isinstance(v, list) else []


def interview_hold(js: dict, upto_t: float) -> dict:
    if not isinstance(js, dict):
        return js
    js["competencies"] = _as_objlist(js.get("competencies"), "name")
    js["claims"] = _as_objlist(js.get("claims"), "claim")
    js["probes_suggested"] = _as_objlist(js.get("probes_suggested"), "question")
    js["bias_guard"] = _as_objlist(js.get("bias_guard"), "issue")
    js["inconsistencies"] = _as_objlist(js.get("inconsistencies"), "now")
    for k in ("strengths", "concerns", "questions_asked", "candidate_questions", "gaps", "terms"):
        v = js.get(k)
        js[k] = [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
    if isinstance(js.get("time_check"), str):
        js["time_check"] = {"minutes": int(upto_t // 60), "note": js["time_check"]}
    if upto_t < 90 or (js.get("stage") in (None, "", "unclear", "intro") and upto_t < 180):
        js["concerns"] = []
        js["inconsistencies"] = []
    for c in (js.get("claims") or []):
        if isinstance(c, dict) and c.get("specificity") not in ("specific", "vague", "unverified"):
            c["specificity"] = "unverified"
    for c in (js.get("competencies") or []):
        if isinstance(c, dict):
            if c.get("status") not in ("covered", "partial", "not yet"):
                c["status"] = "not yet"
            if c.get("status") == "covered" and not c.get("evidence"):
                c["status"] = "partial"
    return js


PACKS: dict[str, Pack] = {
    "clinical": Pack(
        name="clinical", title="Live decision support", lede="In-visit clinical view: differential, next question, tiered red flags, plan stated vs suggested, safety-netting; guideline lookups attached to the next tick.",
        system=LS.SYNTH_SYSTEM, panels=CLINICAL_PANELS, hold=LS.apply_hold, lookups=True, mishearings=True, activity_lookup_label="NHS + patient.info"),
    "interview": Pack(
        name="interview", title="Interview assistant", lede="For the interviewer, disclosed to the candidate: which competencies have evidence, which claims are still vague, the best next probe, and a guard against questions that are not job-related. It never recommends a decision.",
        system=INTERVIEW_SYSTEM, panels=INTERVIEW_PANELS, hold=interview_hold, max_tokens=2400,
        context_fields=[{"key": "job", "label": "Job description", "placeholder": "Paste the job description or the competencies you are hiring for (optional)."},
                        {"key": "cv", "label": "Candidate CV or notes", "placeholder": "Paste the CV, application or your notes (optional)."}]),
}


def pack_public(p: Pack) -> dict:
    return {"name": p.name, "title": p.title, "lede": p.lede, "panels": p.panels, "context_fields": p.context_fields, "lookups": p.lookups, "mishearings": p.mishearings}


_NONWORD = re.compile(r"[^a-z0-9'-]")
