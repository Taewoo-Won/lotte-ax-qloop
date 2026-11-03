"""검증기 — 리포트 문장의 숫자·근거를 표에서 다시 계산해, 맞는 문장만 통과시킨다(불변 영역).

v2 (2026-11-03): 실제 OpenAI 모델 출력(E5)에서 v1 이 맞는 문장 9개를 막았다. 모델은 숫자를 지어내지 않았지만
'근거 문항 없음'(t값을 근거로 인용하되 숫자는 안 씀)과 '평균 효과가 가장 높은 X'(순위 주장)를 썼고,
v1 은 "인용한 값이 문장에 그대로 있어야 한다"는 한 가지 문장 유형만 알았다. v2 는 문장 유형 셋을 구분한다.

판정 규칙
1) 값 주장: 인용한 값을 표에서 다시 꺼내 같은 형식으로 쓴 문자열이 문장에 그대로 있어야 한다.
2) 문장 속 숫자는 전부 인용한 값에서 나온 것이어야 한다(인용 없는 숫자 = 지어낸 숫자).
3) 인용한 근거 문항 문구가 표의 문구와 같아야 한다(규칙 1의 문자열 일치로 확인).
4) 근거 없음 주장: 문장에 '근거 문항 없음'이 있으면 인용한 delta_t 가 있어야 하고, 그 |t| 가 EVIDENCE_T 미만이어야 한다.
   숫자로 안 쓴 delta_t 인용은 이 경우에만 허용한다.
5) 순위 주장: 문장에 '가장 높은/큰' 이 있으면, 그 말 바로 뒤(없으면 앞)에 나오는 액션이 같은 세그먼트 모든 액션 중
   평균 효과(alpha)가 실제로 가장 커야 하고, 그 액션의 alpha 를 인용해야 한다. 숫자로 안 쓴 alpha 인용은 이 경우에만 허용한다.
검증기가 못 잡는 것: 숫자에 붙인 이름·의미(예: t값을 '효과 지표'라고 부름). 이건 사람 검토 몫으로 남긴다.
"""
from __future__ import annotations

import re

NUM = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?p?")
EVIDENCE_T = 3.0
NO_EVIDENCE = "근거 문항 없음"
RANK_WORDS = ("가장 높은", "가장 큰", "가장 높고", "최고")


def _fmt(v, f: str) -> str:
    return f.format(v)


def _rank_subject(text: str, T: dict) -> str | None:
    labels = {k[6:-1]: v for k, v in T.items() if k.startswith("label[")}
    pos = min((text.find(w) for w in RANK_WORDS if w in text), default=-1)
    if pos < 0 or not labels:
        return None
    after = [(text.find(v, pos), a) for a, v in labels.items() if text.find(v, pos) >= 0]
    if after:
        return min(after)[1]
    before = [(pos - text.rfind(v, 0, pos), a) for a, v in labels.items() if text.rfind(v, 0, pos) >= 0]
    return min(before)[1] if before else None


def verify(claim: dict, T: dict) -> tuple[bool, str]:
    text = claim["text"]
    allowed: list[str] = []
    unwritten: list[str] = []
    for key, f in claim["cite"]:
        if key not in T:
            return False, f"표에 없는 인용 키 {key}"
        s = _fmt(T[key], f)
        if s in text:
            allowed.append(s)
        else:
            unwritten.append(key)
    no_ev = NO_EVIDENCE in text
    rank = any(w in text for w in RANK_WORDS)
    if no_ev:
        dts = [k for k, _ in claim["cite"] if k.startswith("delta_t[")]
        if not dts:
            return False, "근거 없음 주장에 근거 t값 인용이 없음"
        big = [k for k in dts if abs(float(T[k])) >= EVIDENCE_T]
        if big:
            return False, f"근거가 있는데(|t|≥{EVIDENCE_T:.0f}) 없다고 씀: {big[0]}"
    if rank:
        subj = _rank_subject(text, T)
        alphas = {k[6:-1]: float(v) for k, v in T.items() if k.startswith("alpha[")}
        if subj is None:
            return False, "순위 주장의 대상 액션을 못 찾음"
        if f"alpha[{subj}]" not in [k for k, _ in claim["cite"]]:
            return False, f"순위 주장 대상 alpha[{subj}] 미인용"
        if alphas[subj] < max(alphas.values()):
            return False, f"순위 틀림: {subj} 는 평균 효과 최대가 아님"
    for key in unwritten:
        if no_ev and key.startswith("delta_t["):
            continue
        if rank and key.startswith("alpha["):
            continue
        return False, f"{key} 값 불일치(표: {_fmt(T[key], dict(claim['cite'])[key])})"
    rest = text
    for s in sorted(allowed, key=len, reverse=True):
        rest = rest.replace(s, " ")
    for k, v in T.items():                            # 액션 이름 속 문자는 숫자로 치지 않음
        if k.startswith("label["):
            rest = rest.replace(str(v), " ")
    rest = re.sub(r"\[[^\]]*\]", " ", rest)          # 세그먼트 태그
    rest = rest.replace("1표준편차", " ")             # 고정 문구
    stray = [m for m in NUM.findall(rest) if m.strip("+-")]
    if stray:
        return False, f"인용 없는 숫자 {stray}"
    return True, "ok"


def verify_v1(claim: dict, T: dict) -> tuple[bool, str]:
    """비교용 v1(값 주장만 앎). E5 재채점에서 v1·v2 를 나란히 보고한다."""
    text = claim["text"]
    allowed: list[str] = []
    for key, f in claim["cite"]:
        if key not in T:
            return False, f"표에 없는 인용 키 {key}"
        s = _fmt(T[key], f)
        if s not in text:
            return False, f"{key} 값 불일치(표: {s})"
        allowed.append(s)
    rest = text
    for s in sorted(allowed, key=len, reverse=True):
        rest = rest.replace(s, " ")
    rest = re.sub(r"\[[^\]]*\]", " ", rest)
    rest = rest.replace("1표준편차", " ")
    stray = [m for m in NUM.findall(rest) if m.strip("+-")]
    if stray:
        return False, f"인용 없는 숫자 {stray}"
    return True, "ok"
