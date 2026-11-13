"""Run the fixed requests in test_cases.json with RAG on and off, through the same pipeline as the app.

For each case we record: recipes passing the filters, retrieved ids, valid suggestions,
invented ids, rule violations and (RAG off) unverified suggestions that are not in the database.

Run from the repo root:  python -m experiments.run_cases
Results: experiments/results/run_cases.md and .json
"""
from __future__ import annotations

import json
from pathlib import Path

import config
import pipeline
from rag.filters import HardFilters

CASES = Path(__file__).parent / "test_cases.json"
OUT_DIR = Path(__file__).parent / "results"


def main():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    rows, details = [], []
    for case in cases:
        f = HardFilters(case["diet"], case["allergies"], case["max_minutes"])
        for use_rag in (True, False):
            res = pipeline.suggest(case["message"], f, use_rag=use_rag)
            v = res.validation
            row = {"case": case["id"], "rag": "on" if use_rag else "off", "allowed": res.n_allowed,
                   "retrieved": len(res.hits), "valid": len(v.valid), "invented": len(v.invented),
                   "rule_violations": len(v.rule_violations), "unverified": len(v.unverified),
                   "parse_error": bool(v.parse_error)}
            rows.append(row)
            details.append({**row, "message": case["message"], "filters": f.describe(),
                            "retrieved_ids": [h.recipe_id for h in res.hits],
                            "valid": [(c.recipe["name"], c.reason) for c in res.cards],
                            "unverified": v.unverified, "invented_ids": v.invented,
                            "violations": v.rule_violations})
            print(row)

    OUT_DIR.mkdir(exist_ok=True)
    head = list(rows[0])
    md = ["# RAG on/off test cases", "",
          f"LLM `{config.LLM_MODEL}`, embeddings `{config.EMBED_MODEL}`, top-k {config.TOP_K}, "
          f"{config.N_SUGGESTIONS} meals requested.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head),
          *["| " + " | ".join(str(r[h]) for h in head) + " |" for r in rows], "",
          "`valid` = suggestions whose id exists in the retrieved recipes and that pass the hard filters. "
          "`unverified` (RAG off) = dishes the model named from its own knowledge; they cannot be checked "
          "against allergies/time or shown as recipe cards.", ""]
    for d in details:
        md.append(f"## {d['case']} (RAG {d['rag']}) – \"{d['message']}\" – {d['filters']}")
        md += [f"- valid: {name}: {reason}" for name, reason in d["valid"]]
        md += [f"- unverified: {name}: {reason}" for name, reason in d["unverified"]]
        md += [f"- invented: {x}" for x in d["invented_ids"]]
        md += [f"- violation: {x}" for x in d["violations"]]
        md.append("")
    (OUT_DIR / "run_cases.md").write_text("\n".join(md), encoding="utf-8")
    (OUT_DIR / "run_cases.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"Saved {OUT_DIR / 'run_cases.md'}")


if __name__ == "__main__":
    main()
