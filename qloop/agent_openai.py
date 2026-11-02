"""OpenAI Agents SDK(0.24.0) 배선 — 본선용. 키가 없으면 --build-only 로 구성과 가드레일만 확인한다.

역할 분담
- LLM: 무엇을 할지 계획, 개방형 응답 해석, 새 문항 문구, 액션플랜 문장(숫자는 인용만).
- 코드: 결정 기여도·효과 추정·배정·폐기/승격 판단, 문장 검증(불변).
- 사람: 설문 발송과 액션플랜 배포를 승인(needs_approval=True).

모델 배분(2026-11 기준 OpenAI 공식 라인업, 환경변수로 바꿀 수 있음)
- 개방형 응답 1차 코딩: QLOOP_MODEL_BULK  (기본 gpt-6-luna, 대량·저비용, 사람 표본과 일치도 κ 로 검수)
- 코딩 불일치 판정·새 문항 문구·액션플랜 문장: QLOOP_MODEL_MAIN (기본 gpt-6.1-sol)
- 누적 자료 중 스캔본·표 이미지 읽기: QLOOP_MODEL_VISION (기본 gpt-6.1-terra)
Agent Builder 는 2026-11-30 종료 예정이라 쓰지 않는다(Agents SDK + Responses API 로만 구성).
"""
from __future__ import annotations

import argparse, json, os, sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:   # 선택 의존성: 본선 환경(openai-agents==0.24.0)에서만 필요
    from agents import Agent, function_tool, output_guardrail, GuardrailFunctionOutput, RunContextWrapper
except ImportError:  # pragma: no cover
    Agent = None

from qloop.verifier import verify            # noqa: E402
from qloop.report import build_tables, build_claims   # noqa: E402

MODEL_MAIN = os.environ.get("QLOOP_MODEL_MAIN", "gpt-6.1-sol")
MODEL_BULK = os.environ.get("QLOOP_MODEL_BULK", "gpt-6-luna")
MODEL_VISION = os.environ.get("QLOOP_MODEL_VISION", "gpt-6.1-terra")


class Cite(BaseModel):
    key: str
    fmt: str


class Claim(BaseModel):
    text: str
    cite: list[Cite]


class PlanReport(BaseModel):
    segment: str
    claims: list[Claim]


@dataclass
class QLoopContext:
    agents: dict            # seg -> QLoopAgent
    last: dict              # seg -> RoundData


def check_report(ctx: QLoopContext, report: PlanReport) -> tuple[bool, list[str]]:
    """가드레일 본체: 문장마다 표에서 다시 계산. 하나라도 틀리면 배포 막음."""
    ag = ctx.agents[report.segment]
    T = build_tables(ag, ctx.last[report.segment])
    bad = []
    for c in report.claims:
        ok, why = verify({"text": c.text, "cite": [(x.key, x.fmt) for x in c.cite]}, T)
        if not ok:
            bad.append(f"{c.text} → {why}")
    return (not bad), bad


def build(ctx: QLoopContext):
    if Agent is None:
        raise RuntimeError("pip install openai-agents==0.24.0 필요")

    @function_tool
    def round_tables(w: RunContextWrapper[QLoopContext], segment: str) -> str:
        """세그먼트의 이번 회차 숫자 표(인용 키 → 값)를 돌려준다. 문장의 숫자는 반드시 여기서 인용한다."""
        ag = w.context.agents[segment]
        return json.dumps(build_tables(ag, w.context.last[segment]), ensure_ascii=False, default=float)

    @function_tool
    def bank_changes(w: RunContextWrapper[QLoopContext], segment: str) -> str:
        """이번 회차 진화기 결정(폐기·승격·새 문항 후보)과 근거를 돌려준다."""
        ag = w.context.agents[segment]
        rnd = w.context.last[segment].round
        return json.dumps([{"op": e.op, "qid": e.qid, "text": ag.bank.items[e.qid].text if e.qid in ag.bank.items else "",
                            "reason": e.reason, "evidence": e.evidence} for e in ag.bank.ledger if e.round == rnd],
                          ensure_ascii=False)

    @function_tool
    def next_survey(w: RunContextWrapper[QLoopContext], segment: str) -> str:
        """다음 회차 설문 초안(문항 ID·문구·출처)을 돌려준다."""
        ag = w.context.agents[segment]
        qids = ag.compose(w.context.last[segment].round + 1)
        return json.dumps([{"qid": q, "text": ag.bank.items[q].text, "source": ag.bank.items[q].source} for q in qids],
                          ensure_ascii=False)

    @function_tool(needs_approval=True)
    def send_survey(w: RunContextWrapper[QLoopContext], segment: str) -> str:
        """설문을 발송한다. 사람 승인 없이는 실행되지 않는다."""
        return f"{segment} 설문 발송 요청 접수(데모: 실제 발송 없음)"

    @output_guardrail
    def numbers_must_match(w: RunContextWrapper[QLoopContext], agent, output: PlanReport) -> GuardrailFunctionOutput:
        ok, bad = check_report(w.context, output)
        return GuardrailFunctionOutput(output_info={"blocked": bad}, tripwire_triggered=not ok)

    planner = Agent[QLoopContext](
        name="Q-Loop 액션플랜 작성",
        model=MODEL_MAIN,
        instructions=(
            "너는 한국후지필름 영업·마케팅 담당자에게 분기 액션플랜을 쓰는 에이전트다. "
            "숫자는 절대 만들지 말고 round_tables 의 값을 인용 키와 형식(fmt)으로만 쓴다. "
            "bank_changes 로 다음 설문이 왜 바뀌는지 근거를 설명하고, next_survey 로 초안을 확인한다. "
            "설문 발송은 send_survey 로 요청하되 사람 승인을 기다린다. 출력은 PlanReport 형식."
        ),
        tools=[round_tables, bank_changes, next_survey, send_survey],
        output_type=PlanReport,
        output_guardrails=[numbers_must_match],
    )
    return planner


def demo_context(seed: int = 0, rounds: int = 8) -> QLoopContext:
    from experiments.e2_sim_loop import run_one
    _, agents = run_one(seed, "qloop", rounds, None)
    return QLoopContext(agents=agents, last={k: a.history[-1] for k, a in agents.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true", help="키 없이 구성과 가드레일만 확인")
    ap.add_argument("--segment", default="b2c_instax")
    a = ap.parse_args()
    ctx = demo_context()
    planner = build(ctx)
    print("에이전트:", planner.name, "| 모델:", MODEL_MAIN, "| 도구:", [t.name for t in planner.tools])
    ag = ctx.agents[a.segment]
    T = build_tables(ag, ctx.last[a.segment])
    claims = build_claims(ag, ctx.last[a.segment], T)
    good = PlanReport(segment=a.segment, claims=[Claim(text=c["text"], cite=[Cite(key=k, fmt=f) for k, f in c["cite"]])
                                                  for c in claims[:3]])
    bad = good.model_copy(deep=True)
    bad.claims[1].text = bad.claims[1].text.replace("%", "0%", 1)      # 숫자 하나를 틀리게
    print("가드레일(맞는 리포트):", check_report(ctx, good))
    print("가드레일(틀린 리포트):", check_report(ctx, bad))
    if a.build_only or not os.environ.get("OPENAI_API_KEY"):
        print("--build-only: 모델 호출 없이 종료")
        return
    from agents import Runner
    res = Runner.run_sync(planner, f"{a.segment} 세그먼트의 이번 분기 액션플랜을 써라.", context=ctx)
    print(res.final_output)


if __name__ == "__main__":
    main()
