"""E5 — 실제 OpenAI 모델 출력으로 두 가지를 잰다(VPS 의 Codex CLI, ChatGPT 구독 로그인).

T1. 액션플랜 문장: 모델에게 숫자표(인용 키 → 값)를 주고 플랜 문장 + 인용 키를 쓰게 한 뒤, 검증기로 채점.
    → '같은 저자가 만든 생성기' 가 아니라 실제 LLM 문장에서 검증기가 무엇을 통과·차단하는가.
T2. 개방형 응답 분류: 니즈가 알려진 응답 문장을 주고 주제를 고르게 해, 정답·오프라인 사전 코더와 비교.

사용(VPS): bash live/run_codex.sh   (codex 로그인 필요, 키 불필요)
결과: results/e5_codex_live.json (프롬프트·원문 출력·채점 전부), 화면에 800자 요약
"""
from __future__ import annotations

import json, os, re, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.e2_sim_loop import run_one            # noqa: E402
from qloop.report import build_tables, EVIDENCE_T       # noqa: E402
from qloop.verifier import verify                       # noqa: E402
from qloop.coder import LexiconCoder, THEMES            # noqa: E402
from qloop.world import TXT, IRRELEVANT                 # noqa: E402  (정답 채점용 — 하네스만 사용)

CODEX = os.environ.get("CODEX") or shutil.which("codex") or "/root/.local/bin/codex"
NEED2THEME = {"stock": "재고·결품", "training": "교육·사용법", "margin": "가격·마진", "newbiz": "신사업·신상품",
              "spec": "기술지원·사양", "price": "가격·마진", "leadtime": "납기", "filmcost": "필름가격",
              "gift": "선물", "upgrade": "신모델", "print": "인화", "goods": "굿즈·포토카드"}


def fmt_of(key: str) -> str:
    if key.startswith(("alpha_t", "delta_t")):
        return "{:.1f}"
    if key.startswith(("alpha[", "delta[")):
        return "{:+.1%}"
    if key.startswith(("n_assigned", "n_q", "n_resp")):
        return "{:d}"
    if key == "rr":
        return "{:.1%}"
    return "{}"


def codex(prompt: str, timeout: int = 420):
    work = tempfile.mkdtemp(prefix="qloop_codex_")
    t0 = time.time()
    p = subprocess.run([CODEX, "exec", "--skip-git-repo-check", "--sandbox", "read-only", prompt],
                       cwd=work, capture_output=True, text=True, timeout=timeout)
    return {"stdout": p.stdout, "stderr_tail": p.stderr[-3000:], "rc": p.returncode, "sec": round(time.time() - t0, 1)}


def grab_json(out: str):
    m = re.findall(r"<<<JSON(.*?)JSON>>>", out, re.S)
    raw = m[-1] if m else out[out.find("{"): out.rfind("}") + 1]
    return json.loads(raw)


def model_line(stderr: str, stdout: str) -> str:
    for line in (stderr + "\n" + stdout).splitlines():
        if re.match(r"\s*model\s*[:=]", line, re.I):
            return line.strip()[:80]
    return "unknown"


def t1_plans(agents):
    out = []
    for seg, ag in agents.items():
        rd = ag.history[-1]
        T = build_tables(ag, rd)
        table = {k: {"표기": fmt_of(k).format(v), "값": v if not isinstance(v, float) else round(v, 6)} for k, v in T.items()}
        labels = {a: ag.action_label[a] for a in ag.actions}
        prompt = (
            "너는 한국후지필름 영업·마케팅 담당자에게 이번 분기 액션플랜을 쓰는 에이전트다. 도구나 명령을 쓰지 말고 바로 답하라.\n"
            f"세그먼트: {seg}. 액션 이름: {json.dumps(labels, ensure_ascii=False)}\n"
            "아래 숫자표만 근거로 한국어 문장 6~10개를 써라. 규칙:\n"
            "1) 문장 속 숫자는 반드시 숫자표의 '표기' 문자열을 그대로 쓴다(자릿수·부호·% 그대로).\n"
            "2) 숫자표에 없는 숫자(날짜, 개수, 비율, 분기 번호 등)는 절대 쓰지 않는다.\n"
            f"3) top_q_text 문항을 근거로 말할 때는 그 액션의 delta_t 절댓값이 {EVIDENCE_T:.0f} 이상일 때만 쓰고, 미만이면 '근거 문항 없음 — 평균 효과로 배정'이라고 쓴다.\n"
            "4) 문장마다 쓴 숫자표 키를 cite 에 모두 나열한다. '1표준편차' 라는 표현은 써도 된다.\n"
            "출력은 아래 형식 하나만, 마커 사이에 JSON 으로:\n"
            '<<<JSON {"claims":[{"text":"...","cite":["키1","키2"]}]} JSON>>>\n'
            f"숫자표: {json.dumps(table, ensure_ascii=False)}"
        )
        r = codex(prompt)
        rec = {"segment": seg, "prompt": prompt, **r}
        try:
            claims = grab_json(r["stdout"])["claims"]
            res = []
            for c in claims:
                cl = {"text": c["text"], "cite": [(k, fmt_of(k)) for k in c.get("cite", [])]}
                ok, why = verify(cl, T)
                res.append({"text": c["text"], "cite": c.get("cite", []), "pass": ok, "why": why})
            rec["claims"] = res
        except Exception as ex:  # 파싱 실패도 기록
            rec["parse_error"] = repr(ex)[:300]
        out.append(rec)
    return out


def t2_coding():
    items, truth = [], {}
    i = 0
    for need, th in NEED2THEME.items():
        for t in TXT[need]:
            items.append((f"r{i:03d}", t)); truth[f"r{i:03d}"] = th; i += 1
    for t in IRRELEVANT:
        items.append((f"r{i:03d}", t)); truth[f"r{i:03d}"] = "없음"; i += 1
    themes = sorted(set(NEED2THEME.values()) | {"배송", "서비스·AS", "앱·온라인몰"})
    prompt = (
        "도구나 명령을 쓰지 말고 바로 답하라. 아래는 한국후지필름 고객(사진관 파트너·산업재 고객·인스탁스 회원)의 개방형 설문 응답이다. "
        f"각 응답에 가장 맞는 주제 하나를 이 목록에서 고르고, 해당 없으면 '없음': {json.dumps(themes, ensure_ascii=False)}\n"
        '출력은 마커 사이 JSON 하나: <<<JSON {"labels":{"r000":"주제",...}} JSON>>>\n'
        + "\n".join(f"{k}|{t}" for k, t in items)
    )
    r = codex(prompt)
    rec = {"prompt": prompt, **r, "n": len(items)}
    lex = LexiconCoder().code([t for _, t in items])
    lex_ok = sum((truth[k] == "없음" and not labs) or (truth[k] in labs[:1]) for (k, _), labs in zip(items, lex))
    rec["lexicon_correct"] = int(lex_ok)
    try:
        labels = grab_json(r["stdout"])["labels"]
        rec["llm_correct"] = int(sum(labels.get(k, "") == truth[k] for k, _ in items))
        rec["llm_missing"] = int(sum(k not in labels for k, _ in items))
        rec["llm_wrong_examples"] = [{"text": t, "truth": truth[k], "llm": labels.get(k)} for k, t in items
                                     if labels.get(k, "") != truth[k]][:12]
    except Exception as ex:
        rec["parse_error"] = repr(ex)[:300]
    return rec


def main():
    started = datetime.now(timezone.utc).isoformat()
    _, agents = run_one(0, "qloop", 8, None)
    t1 = t1_plans(agents)
    t2 = t2_coding()
    allc = [c for r in t1 for c in r.get("claims", [])]
    res = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "codex_version": subprocess.run([CODEX, "--version"], capture_output=True, text=True).stdout.strip(),
        "model_line": model_line(t1[0]["stderr_tail"], t1[0]["stdout"]) if t1 else "",
        "t1_plans": t1, "t2_coding": t2,
        "summary": {
            "plan_claims": len(allc), "plan_pass": sum(c["pass"] for c in allc),
            "plan_blocked": [c for c in allc if not c["pass"]][:6],
            "coding_n": t2.get("n"), "coding_llm_correct": t2.get("llm_correct"),
            "coding_lexicon_correct": t2.get("lexicon_correct"),
        },
    }
    (ROOT / "results/e5_codex_live.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    s = res["summary"]
    print("E5 요약 |", res["codex_version"], "|", res["model_line"])
    print(f"T1 플랜 문장 {s['plan_claims']}개 중 검증 통과 {s['plan_pass']}개")
    for c in s["plan_blocked"][:3]:
        print(" 차단:", c["text"][:60], "→", c["why"][:40])
    print(f"T2 분류 {s['coding_n']}건: LLM 정답 {s['coding_llm_correct']} / 사전 코더 {s['coding_lexicon_correct']}")
    for r in t1:
        if r.get("parse_error") or r["rc"] != 0:
            print(" T1 오류", r["segment"], r["rc"], r.get("parse_error", "")[:80])
    if t2.get("parse_error") or t2["rc"] != 0:
        print(" T2 오류", t2["rc"], t2.get("parse_error", "")[:80])


if __name__ == "__main__":
    main()
