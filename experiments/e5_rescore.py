"""E5 재채점 — VPS 에서 받은 실제 모델 출력(results/e5_codex_live.json)을 다시 부르지 않고 검증기 v1·v2 로 채점한다.

숫자표는 모델에게 실제로 보낸 프롬프트 안의 표를 그대로 꺼내 쓴다(재계산 차이 없음).
'맞는 문장인가'의 정답은 사람이 아니라 표 대조로 정한다: 문장의 모든 숫자가 표 값과 같고, 근거 없음·순위 주장이 표와 맞으면 맞는 문장.
그 판정을 코드로 다시 하지 않기 위해, 결과 파일에 문장별 v1·v2 판정과 사유를 남기고 사람이 읽어 확인한다.
사용: python3 experiments/e5_rescore.py → results/e5_rescored.json
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qloop.verifier import verify, verify_v1          # noqa: E402
from experiments.e5_codex_live import fmt_of           # noqa: E402


def table_from_prompt(prompt: str) -> dict:
    tbl = json.loads(prompt.split("숫자표: ", 1)[1])
    T = {k: v["값"] for k, v in tbl.items()}
    labels = json.loads(re.search(r"액션 이름: (\{.*?\})\n", prompt).group(1))
    for a, lab in labels.items():
        T[f"label[{a}]"] = lab
    return T


def main():
    d = json.loads((ROOT / "results/e5_codex_live.json").read_text())
    rows = []
    for r in d["t1_plans"]:
        T = table_from_prompt(r["prompt"])
        for c in r.get("claims", []):
            cl = {"text": c["text"], "cite": [(k, fmt_of(k)) for k in c["cite"]]}
            v1, w1 = verify_v1(cl, T)
            v2, w2 = verify(cl, T)
            rows.append({"segment": r["segment"], "text": c["text"], "cite": c["cite"],
                         "v1": v1, "v1_why": w1, "v2": v2, "v2_why": w2})
    out = {
        "source": "results/e5_codex_live.json", "codex_version": d["codex_version"],
        "run_utc": [d["started_utc"], d["finished_utc"]],
        "claims": len(rows), "v1_pass": sum(x["v1"] for x in rows), "v2_pass": sum(x["v2"] for x in rows),
        "v1_blocked_types": {
            "근거 없음(t값 인용·숫자 안 씀)": sum((not x["v1"]) and "근거 문항 없음" in x["text"] for x in rows),
            "순위 주장(숫자 안 씀)": sum((not x["v1"]) and "가장 높은" in x["text"] for x in rows),
        },
        "coding": {k: d["t2_coding"].get(k) for k in ("n", "llm_correct", "lexicon_correct", "llm_missing", "llm_wrong_examples")},
        "rows": rows,
    }
    (ROOT / "results/e5_rescored.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print({k: out[k] for k in ("claims", "v1_pass", "v2_pass", "v1_blocked_types")})
    for x in rows:
        if not x["v2"]:
            print("v2 차단:", x["text"][:70], "→", x["v2_why"])


if __name__ == "__main__":
    main()
