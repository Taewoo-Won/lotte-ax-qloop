"""E3 — 검증기 시험: 리포트 문장을 일부러 틀리게 바꿔도 걸러내는가, 맞는 문장은 통과시키는가.

문장 생성기와 검증기를 같은 사람이 만들었으므로 이것은 '기계적 보증' 시험이다.
실제 LLM 문장에 대한 차단율은 본선에서 OpenAI 모델 출력으로 다시 잰다.
사용: python3 experiments/e3_verifier.py → results/e3_verifier.json, results/sample_plan_seed0.md
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.e2_sim_loop import run_one          # noqa: E402
from qloop.report import build_tables, build_claims, plan_markdown, EVIDENCE_T   # noqa: E402
from qloop.verifier import verify, NUM               # noqa: E402


def perturb(claim: dict, T: dict, kind: str, rng, bank_texts: list[str]):
    text = claim["text"]
    nums = [(k, f) for k, f in claim["cite"] if f != "{}"]
    if kind == "shift" and nums:
        k, f = nums[rng.integers(len(nums))]
        old = f.format(T[k])
        for fac in rng.permutation([0.5, 0.8, 1.25, 1.6]):
            v = T[k] * fac if not isinstance(T[k], int) else int(round(T[k] * fac)) + (1 if T[k] == 0 else 0)
            new = f.format(v)
            if new != old:
                return text.replace(old, new, 1)
    if kind == "sign":
        for k, f in nums:
            old = f.format(T[k])
            if old.startswith("+") or old.startswith("-"):
                new = ("-" if old[0] == "+" else "+") + old[1:]
                if old.lstrip("+-").strip("0.%p") != "":
                    return text.replace(old, new, 1)
    if kind == "digit" and nums:
        k, f = nums[rng.integers(len(nums))]
        old = f.format(T[k])
        ds = [i for i, c in enumerate(old) if c.isdigit()]
        if len(ds) >= 2:
            i, j = ds[-1], ds[-2]
            if old[i] != old[j]:
                new = list(old); new[i], new[j] = new[j], new[i]
                return text.replace(old, "".join(new), 1)
        if ds:
            i = ds[-1]
            new = old[:i] + str((int(old[i]) + 3) % 10) + old[i + 1:]
            return text.replace(old, new, 1)
    if kind == "evidence":
        for k, f in claim["cite"]:
            if f == "{}":
                cur = T[k]
                others = [t for t in bank_texts if t != cur]
                if others:
                    return text.replace(cur, others[rng.integers(len(others))], 1)
    if kind == "extra":
        return text + f" 전 분기 대비 {rng.uniform(3, 40):.1f}% 늘었다."
    return None


def main():
    rng = np.random.default_rng(7)
    originals, results = [], {k: {"made": 0, "blocked": 0} for k in ["shift", "sign", "digit", "evidence", "extra"]}
    sample_md = None
    ev = {}                      # 근거 문항 정확도: 보여 준 근거 문항이 그 액션이 듣는 니즈를 재는 문항인가(세계 정답으로 채점)
    for seed in range(20):
        recs, agents, world = run_one(seed, "qloop", 8, None, return_world=True)
        for seg, ag in agents.items():
            rd = ag.history[-1]
            T = build_tables(ag, rd)
            need_of_action = {a.key: a.need for a in world.seg[seg].spec.actions}
            for a in ag.actions:
                q = T[f"top_q[{a}]"]
                hit = world.truth_need_of(seg, q) == need_of_action[a]
                big = abs(T[f"delta_t[{a},{q}]"]) >= EVIDENCE_T
                d = ev.setdefault(seg, {"all": [0, 0], "shown": [0, 0]})
                d["all"][0] += hit; d["all"][1] += 1
                if big:
                    d["shown"][0] += hit; d["shown"][1] += 1
            if seed >= 10:
                continue
            claims = build_claims(ag, rd, T)
            bank_texts = [i.text for i in ag.bank.items.values()]
            for c in claims:
                ok, why = verify(c, T)
                originals.append(ok)
                for kind in results:
                    bad = perturb(c, T, kind, rng, bank_texts)
                    if bad is None or bad == c["text"]:
                        continue
                    results[kind]["made"] += 1
                    ok2, _ = verify({"text": bad, "cite": c["cite"]}, T)
                    results[kind]["blocked"] += int(not ok2)
            if seed == 0:
                passed = [c for c in claims if verify(c, T)[0]]
                sample_md = (sample_md or "# 예시 액션플랜 (시드 0, 마지막 회차, 템플릿 문장 + 검증 통과분)\n\n") + \
                    plan_markdown(ag, rd, passed) + "\n\n"
    out = {
        "evidence_accuracy": {seg: {"all_top": f"{v['all'][0]}/{v['all'][1]}", "shown_t>=3": f"{v['shown'][0]}/{v['shown'][1]}"}
                              for seg, v in ev.items()},
        "evidence_seeds": 20, "verifier_seeds": 10,
        "original_claims": len(originals), "original_passed": int(sum(originals)),
        "perturbed": results,
        "perturbed_total": int(sum(v["made"] for v in results.values())),
        "perturbed_blocked": int(sum(v["blocked"] for v in results.values())),
    }
    (ROOT / "results/e3_verifier.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (ROOT / "results/sample_plan_seed0.md").write_text(sample_md)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
