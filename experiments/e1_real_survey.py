"""E1 — 실제 설문 응답(공개 데이터 13만 명)에서 Q-Loop 폐기 규칙이 정보를 지키며 설문을 줄이는가.

사용: python3 experiments/e1_real_survey.py  (data/raw/airline/{train,test}.csv 필요 — fetch_data.sh)
결과: results/e1_real_survey.json
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/airline"
ITEMS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
         "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
         "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"]
KO = {"Inflight wifi service": "기내 와이파이", "Departure/Arrival time convenient": "출도착 시간 편의",
      "Ease of Online booking": "온라인 예약 편의", "Gate location": "게이트 위치", "Food and drink": "기내식·음료",
      "Online boarding": "온라인 탑승수속", "Seat comfort": "좌석 편안함", "Inflight entertainment": "기내 엔터테인먼트",
      "On-board service": "탑승 서비스", "Leg room service": "다리 공간", "Baggage handling": "수하물 처리",
      "Checkin service": "체크인 서비스", "Inflight service": "기내 서비스", "Cleanliness": "청결"}
EPS, MIN_EV, MAX_KILLS, CORR_T, BUDGET, WAVES = 0.002, 2, 2, 0.7, 7, 6


def load(path):
    df = pd.read_csv(path)
    df["y"] = (df["satisfaction"] == "satisfied").astype(int)
    for c in ITEMS:
        df[c] = df[c].replace(0, np.nan)
    return df


def xy(df, items, med):
    X = df[items].fillna(med[items]).to_numpy(float)
    return X, df["y"].to_numpy()


def fit_auc(tr, te, items):
    med = tr[ITEMS].median()
    Xtr, ytr = xy(tr, items, med)
    Xte, yte = xy(te, items, med)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
    p = m.predict_proba((Xte - mu) / sd)[:, 1]
    return roc_auc_score(yte, p), m, mu, sd, med


def perm_importance(m, mu, sd, med, te, items, rng, reps=5):
    X, y = xy(te, items, med)
    Z = (X - mu) / sd
    base = roc_auc_score(y, m.predict_proba(Z)[:, 1])
    out = {}
    for j, it in enumerate(items):
        drops = []
        for _ in range(reps):
            Zp = Z.copy()
            Zp[:, j] = rng.permutation(Zp[:, j])
            drops.append(base - roc_auc_score(y, m.predict_proba(Zp)[:, 1]))
        out[it] = float(np.mean(drops))
    return base, out


def qloop_select(train, rng):
    order = rng.permutation(len(train))
    waves = [train.iloc[ix] for ix in np.array_split(order, WAVES)]
    active = list(ITEMS)
    hist = {it: [] for it in ITEMS}
    log = []
    for w in range(1, WAVES):
        tr = pd.concat(waves[:w])
        te = waves[w]
        _, m, mu, sd, med = fit_auc(tr, te, active)
        base, imp = perm_importance(m, mu, sd, med, te, active, rng)
        for it in active:
            hist[it].append(imp[it])
        kills = []
        for it in active:
            last = hist[it][-MIN_EV:]
            if len(last) == MIN_EV and all(v < EPS for v in last):
                kills.append((max(last), it, f"기여도 {MIN_EV}회 연속 <{EPS}"))
        C = te[active].corr().abs()
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if C.loc[a, b] >= CORR_T:
                    lo = a if np.mean(hist[a][-2:]) < np.mean(hist[b][-2:]) else b
                    if all(k[1] != lo for k in kills):
                        kills.append((np.mean(hist[lo][-2:]), lo, f"응답 상관 {C.loc[a, b]:.2f} 중복"))
        kills.sort()
        for v, it, why in kills[:MAX_KILLS]:
            active.remove(it)
            log.append({"wave": w, "item": KO[it], "reason": why, "value": round(float(v), 4)})
        log.append({"wave": w, "base_auc": round(base, 4), "active": len(active)})
    if len(active) > BUDGET:
        rest = sorted(active, key=lambda it: -np.mean(hist[it][-2:]))[BUDGET:]
        for it in rest:
            log.append({"wave": "final", "item": KO[it], "reason": f"상한 {BUDGET}문항 — 휴면(폐기 아님)"})
        active = sorted(active, key=lambda it: -np.mean(hist[it][-2:]))[:BUDGET]
    return active, log


def main():
    tr_all, te_all = load(RAW / "train.csv"), load(RAW / "test.csv")
    out = {}
    for seg, lab in [("Business travel", "출장 고객(B2B 성격)"), ("Personal Travel", "개인 여행 고객(B2C 성격)")]:
        rng = np.random.default_rng(0)
        tr = tr_all[tr_all["Type of Travel"] == seg].reset_index(drop=True)
        te = te_all[te_all["Type of Travel"] == seg].reset_index(drop=True)
        sel, log = qloop_select(tr, rng)
        k = len(sel)
        full, *_ = fit_auc(tr, te, ITEMS)
        ql, *_ = fit_auc(tr, te, sel)
        rnd = [fit_auc(tr, te, list(rng.choice(ITEMS, k, replace=False)))[0] for _ in range(200)]
        low = tr[ITEMS].mean().sort_values().index[:k].tolist()
        lowa, *_ = fit_auc(tr, te, low)
        cor = tr[ITEMS + ["y"]].corr()["y"].drop("y").abs().sort_values(ascending=False).index[:k].tolist()
        cora, *_ = fit_auc(tr, te, cor)
        out[seg] = {
            "label": lab, "n_train": len(tr), "n_test": len(te), "k": k,
            "auc_full14": round(full, 4), "auc_qloop": round(ql, 4),
            "auc_random_mean": round(float(np.mean(rnd)), 4), "auc_random_sd": round(float(np.std(rnd)), 4),
            "auc_random_p95": round(float(np.percentile(rnd, 95)), 4),
            "auc_lowest_mean": round(lowa, 4), "auc_top_corr": round(cora, 4),
            "selected": [KO[i] for i in sel], "lowest_mean_items": [KO[i] for i in low],
            "top_corr_items": [KO[i] for i in cor], "log": log,
        }
        print(seg, {kk: out[seg][kk] for kk in ["k", "auc_full14", "auc_qloop", "auc_random_mean", "auc_lowest_mean", "auc_top_corr"]})
    (ROOT / "results/e1_real_survey.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
