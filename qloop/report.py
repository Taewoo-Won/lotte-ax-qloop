"""액션플랜 리포트 — 영업·마케팅 담당자가 받는 결과물.

원칙: 숫자는 코드가 계산하고, 문장(LLM 또는 템플릿)은 그 숫자를 '인용'만 한다.
근거 문항은 상호작용 |t| 가 EVIDENCE_T 이상일 때만 보여 준다. 미만이면 "근거 문항 없음 — 평균 효과로 배정".
문장마다 citations 를 달아 두고, verifier 가 표에서 다시 계산해 맞는 문장만 내보낸다.
"""
from __future__ import annotations

import numpy as np

from .analyst import RoundData

EVIDENCE_T = 3.0


def build_tables(agent, rd: RoundData) -> dict:
    """리포트가 인용할 수 있는 숫자 표. 키 → 값."""
    fit = agent.fit
    T: dict[str, float | str] = {}
    T["n_q"] = len(rd.qids)
    T["n_resp"] = len(rd.resp)
    T["rr"] = len(rd.resp) / agent.n
    assign = rd.assign
    for a in agent.actions:
        T[f"alpha[{a}]"] = fit.alpha[a]
        T[f"alpha_t[{a}]"] = fit.alpha_t[a]
        T[f"n_assigned[{a}]"] = sum(1 for x in assign if x == a)
        j = int(np.argmax(np.abs(fit.delta_t[fit.actions.index(a)])))
        q = fit.qids[j]
        T[f"top_q[{a}]"] = q
        T[f"top_q_text[{a}]"] = agent.bank.items[q].text
        T[f"delta[{a},{q}]"] = float(fit.delta[fit.actions.index(a), j])
        T[f"delta_t[{a},{q}]"] = float(fit.delta_t[fit.actions.index(a), j])
    return T


def build_claims(agent, rd: RoundData, T: dict) -> list[dict]:
    """템플릿 문장 + 인용. (LLM 백엔드에서는 같은 인용 키를 받아 문장만 다르게 쓴다.)"""
    seg = agent.seg
    claims = [{
        "text": f"[{seg}] 이번 회차 {T['n_q']}문항, 응답 {T['n_resp']}곳(응답률 {T['rr']:.1%}).",
        "cite": [("n_q", "{:d}"), ("n_resp", "{:d}"), ("rr", "{:.1%}")],
    }]
    for a in agent.actions:
        q = T[f"top_q[{a}]"]
        lab = agent.action_label[a]
        claims.append({
            "text": f"'{lab}' 평균 효과 {T[f'alpha[{a}]']:+.1%}(t={T[f'alpha_t[{a}]']:.1f}).",
            "cite": [(f"alpha[{a}]", "{:+.1%}"), (f"alpha_t[{a}]", "{:.1f}")],
        })
        if abs(T[f"delta_t[{a},{q}]"]) >= EVIDENCE_T:
            claims.append({
                "text": f"'{lab}' 효과는 「{T[f'top_q_text[{a}]']}」 응답이 1표준편차 높을수록 "
                        f"{T[f'delta[{a},{q}]']:+.1%}p 달라진다(t={T[f'delta_t[{a},{q}]']:.1f}).",
                "cite": [(f"top_q_text[{a}]", "{}"), (f"delta[{a},{q}]", "{:+.1%}"), (f"delta_t[{a},{q}]", "{:.1f}")],
            })
        else:
            claims.append({"text": f"'{lab}'은(는) 근거 문항 없음 — 평균 효과로 배정.", "cite": []})
        claims.append({
            "text": f"이번 회차 '{lab}' 배정 {T[f'n_assigned[{a}]']}곳.",
            "cite": [(f"n_assigned[{a}]", "{:d}")],
        })
    return claims


def plan_markdown(agent, rd: RoundData, claims_ok: list[dict]) -> str:
    """검증을 통과한 문장 + 진화 원장으로 담당자용 액션플랜을 만든다."""
    lines = [f"## {agent.seg} — R{rd.round} 액션플랜", ""]
    lines += [f"- {c['text']}" for c in claims_ok]
    ch = [e for e in agent.bank.ledger if e.round == rd.round]
    if ch:
        lines += ["", "### 다음 회차 설문 변경(근거 포함)"]
        for e in ch:
            it = agent.bank.items.get(e.qid)
            lines.append(f"- {e.op} {e.qid} 「{it.text if it else ''}」 — {e.reason}")
    return "\n".join(lines)
