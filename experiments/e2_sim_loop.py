"""E2 — 심은 정답 시뮬레이션에서 기존 방식 / 학습기만 / Q-Loop 를 같은 세계에 나란히 돌린다.

사용: python3 experiments/e2_sim_loop.py [--seeds 20] [--rounds 8] [--decay none|0]
결과: results/e2_runs.json (회차별 기록), results/e2_summary.json (요약), results/e2_bank_seed0.json (진화 원장 예시)
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qloop.world import World                     # noqa: E402  (하네스만 세계를 안다)
from qloop.agent import make_agent                # noqa: E402
from qloop.analyst import RoundData              # noqa: E402

BUDGET = {"b2b_photo": 8, "b2b_industrial": 7, "b2c_instax": 7}
HIDDEN = {"b2b_photo": ["stock"], "b2b_industrial": ["leadtime"], "b2c_instax": ["filmcost", "goods"]}
KINDS = ["static", "fixed", "qloop"]


def run_one(seed: int, kind: str, rounds: int, decay):
    world = World(seed, length_decay_override=decay)
    cat = world.public_catalog()
    agents = {seg: make_agent(kind, seg, cat[seg], seed, si, BUDGET[seg]) for si, seg in enumerate(cat)}
    for seg, ag in agents.items():
        ag.set_initial_sales(world.erp_last_sales(seg))
    registered: set[tuple[str, str]] = set()
    recs = []
    for t in range(rounds):
        for seg, ag in agents.items():
            for it in ag.bank.items.values():
                if it.theme and (seg, it.qid) not in registered:
                    world.register_question(seg, it.qid, it.theme)
                    registered.add((seg, it.qid))
            qids = ag.compose(t)
            resp, answers, texts = world.survey(seg, qids)
            coded = ag.coder.code(texts)
            assign, randomized = ag.assign(t, resp, answers, qids)
            ref = ag.ref.copy()
            sales = world.apply_actions(seg, assign)
            truth = world.last_truth
            ag.learn(RoundData(t, qids, resp, answers, texts, coded, assign, randomized, sales, ref))
            nonanchor = [q for q in qids if not ag.bank.items[q].anchor]
            covered = {h: any(world.truth_need_of(seg, q) == h and world.truth_informative(seg, q) for q in qids)
                       for h in HIDDEN[seg]}
            recs.append({
                "seed": seed, "kind": kind, "seg": seg, "round": t, "n_q": len(qids),
                "rr": len(resp) / ag.n, "net": float(truth["net"].sum()), "oracle": float(truth["oracle"].sum()),
                "informative_share": float(np.mean([world.truth_informative(seg, q) for q in nonanchor])) if nonanchor else 0.0,
                "hidden_covered": covered,
                "rollbacks": getattr(ag, "rollbacks", 0),
            })
        world.advance()
    # 진화 판단의 정오(세계 정답으로 채점): 폐기한 문항이 실제 신호 문항이었나, 승격한 새 문항이 실제 신호였나
    audit = {}
    for seg, ag in agents.items():
        for it in ag.bank.items.values():          # 마지막 회차에 막 만든 후보도 채점할 수 있게 등록
            if it.theme and (seg, it.qid) not in registered:
                world.register_question(seg, it.qid, it.theme)
                registered.add((seg, it.qid))
        kills = [e.qid for e in ag.bank.ledger if e.op == "kill"]
        proms = [e.qid for e in ag.bank.ledger if e.op == "promote"]
        cands = [e.qid for e in ag.bank.ledger if e.op == "add_candidate"]
        audit[seg] = {
            "kills": len(kills), "kills_informative": sum(world.truth_informative(seg, q) for q in kills),
            "promotes": len(proms), "promotes_informative": sum(world.truth_informative(seg, q) for q in proms),
            "candidates": len(cands), "candidates_informative": sum(world.truth_informative(seg, q) for q in cands),
        }
    recs[-1]["audit"] = audit
    return recs, agents


def summarize(runs):
    out = {}
    by = {}
    for r in runs:
        by.setdefault((r["kind"], r["seed"]), []).append(r)
    seeds = sorted({r["seed"] for r in runs})
    cap = {k: [] for k in KINDS}
    capseg = {k: {} for k in KINDS}
    for k in KINDS:
        for s in seeds:
            rs = by[(k, s)]
            cap[k].append(sum(r["net"] for r in rs) / sum(r["oracle"] for r in rs))
            for seg in BUDGET:
                rr = [r for r in rs if r["seg"] == seg]
                capseg[k].setdefault(seg, []).append(sum(r["net"] for r in rr) / sum(r["oracle"] for r in rr))
    rng = np.random.default_rng(0)

    def boot(d):
        d = np.asarray(d)
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    out["capture_total"] = {k: {"mean": float(np.mean(v)), "ci95": boot(v)} for k, v in cap.items()}
    out["capture_by_segment"] = {k: {seg: float(np.mean(v)) for seg, v in d.items()} for k, d in capseg.items()}
    out["paired_diff"] = {
        "qloop_minus_fixed": {"mean": float(np.mean(np.subtract(cap["qloop"], cap["fixed"]))),
                              "ci95": boot(np.subtract(cap["qloop"], cap["fixed"])),
                              "wins": int(np.sum(np.subtract(cap["qloop"], cap["fixed"]) > 0))},
        "fixed_minus_static": {"mean": float(np.mean(np.subtract(cap["fixed"], cap["static"]))),
                               "ci95": boot(np.subtract(cap["fixed"], cap["static"])),
                               "wins": int(np.sum(np.subtract(cap["fixed"], cap["static"]) > 0))},
        "qloop_minus_static": {"mean": float(np.mean(np.subtract(cap["qloop"], cap["static"]))),
                               "ci95": boot(np.subtract(cap["qloop"], cap["static"])),
                               "wins": int(np.sum(np.subtract(cap["qloop"], cap["static"]) > 0))},
        "n_seeds": len(seeds),
    }
    per_round = {}
    for k in KINDS:
        for seg in BUDGET:
            rows = [r for r in runs if r["kind"] == k and r["seg"] == seg]
            R = max(r["round"] for r in rows) + 1
            per_round.setdefault(k, {})[seg] = {
                "capture": [float(np.mean([r["net"] / r["oracle"] for r in rows if r["round"] == t])) for t in range(R)],
                "n_q": [float(np.mean([r["n_q"] for r in rows if r["round"] == t])) for t in range(R)],
                "rr": [float(np.mean([r["rr"] for r in rows if r["round"] == t])) for t in range(R)],
                "informative_share": [float(np.mean([r["informative_share"] for r in rows if r["round"] == t])) for t in range(R)],
                "hidden_covered": {h: [float(np.mean([r["hidden_covered"][h] for r in rows if r["round"] == t]))
                                       for t in range(R)] for h in HIDDEN[seg]},
            }
    out["per_round"] = per_round
    aud = [r["audit"] for r in runs if r["kind"] == "qloop" and "audit" in r]
    out["evolution_audit"] = {seg: {k: int(sum(a[seg][k] for a in aud)) for k in aud[0][seg]} for seg in BUDGET}
    out["rollbacks_total"] = int(sum(max(r["rollbacks"] for r in runs if r["kind"] == "qloop" and r["seed"] == s and r["seg"] == seg)
                                   for s in seeds for seg in BUDGET))
    return out


def dump_bank(agents, path):
    data = {}
    for seg, ag in agents.items():
        data[seg] = {
            "items": [{"qid": i.qid, "text": i.text, "source": i.source, "status": i.status, "anchor": i.anchor,
                       "theme": i.theme, "values": [[r, round(v, 2)] for r, v in i.values],
                       "kill_reason": i.kill_reason, "revival": i.revival} for i in ag.bank.items.values()],
            "ledger": [{"round": e.round, "op": e.op, "qid": e.qid, "reason": e.reason, "evidence": e.evidence}
                       for e in ag.bank.ledger],
        }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--decay", default="none")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    decay = None if a.decay == "none" else float(a.decay)
    t0 = time.time()
    runs = []
    for s in range(a.seeds):
        for k in KINDS:
            recs, agents = run_one(s, k, a.rounds, decay)
            runs.extend(recs)
            if s == 0 and k == "qloop":
                dump_bank(agents, ROOT / f"results/e2_bank_seed0{a.tag}.json")
    summ = summarize(runs)
    summ["config"] = {"seeds": a.seeds, "rounds": a.rounds, "length_decay_override": decay,
                      "seconds": round(time.time() - t0, 1)}
    (ROOT / f"results/e2_runs{a.tag}.json").write_text(json.dumps(runs, ensure_ascii=False))
    (ROOT / f"results/e2_summary{a.tag}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    ct = summ["capture_total"]
    print({k: round(v["mean"], 3) for k, v in ct.items()}, summ["paired_diff"]["qloop_minus_fixed"], summ["config"])


if __name__ == "__main__":
    main()
