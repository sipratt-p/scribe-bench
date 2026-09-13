"""Phase B (beast on official DeepSeek weights, Qwen stopped): regenerate the DeepSeek-dependent outputs with
the non-abliterated build so every published number rests on official weights.
  1. notes for best / best_cite on dev + test with note_model = dsv4flash_official (cached under the new model tag)
  2. judge-2 (official DeepSeek) attribution / plan scores for the vanilla and best notes on test
Phase C (Qwen up) then scores everything with judge 1 via verif_eval-style calls.
  python -m autoresearch.official_rerun"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoresearch import loop as L
from autoresearch.verif_eval import CONFIGS
recs, dev, test = L.load_recs()
qwen = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=2)   # only cached calls expected (role map)
ds = L.LLM("http://localhost:8000/v1", "DeepSeek-V4-Flash-DSpark", workers=4)
OFF = {k: {**v, "note_model": "dsv4flash_official"} for k, v in CONFIGS.items() if k in ("best", "best_cite")}
out = {}
for name, cfg in OFF.items():
    nm = L.note_llm_for(cfg, qwen)
    def one(cid):
        rec = recs[cid]; d = L.transcript_for(cfg, rec, qwen); n = L.note_for(cfg, rec, d, nm); return cid, len(n)
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, dev + test))
    out[name] = {c: n for c, n in rows}; print(name, "notes:", len(rows), "mean chars", sum(n for _, n in rows) // len(rows), flush=True)
# judge-2 with official weights on the test notes of vanilla, best(official), best_cite(official)
j2 = {}
for name in ("theirs", "best", "best_cite"):
    cfg = OFF.get(name, CONFIGS[name]); nm = L.note_llm_for(cfg, qwen)
    def one(cid):
        rec = recs[cid]; d = L.transcript_for(cfg, rec, qwen); n = L.note_for(cfg, rec, d, nm)
        m = L.score_note(rec, n, rec["dialogue"], ds); return cid, m
    with ThreadPoolExecutor(4) as ex:
        j2[name] = dict(ex.map(one, test))
    print(name, "judge-2 official: mean misattrib", round(sum(m["misattrib"] for m in j2[name].values()) / len(test), 3), flush=True)
json.dump({"notes": out, "judge2_official_test": j2}, open(ROOT / "runs/official_rerun_phaseB.json", "w"), indent=1)
print("DONE", flush=True)
