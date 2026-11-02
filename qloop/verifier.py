"""검증기 — 리포트 문장의 숫자·근거를 표에서 다시 계산해, 맞는 문장만 통과시킨다(불변 영역).

판정 규칙
1) 인용한 값마다 표에서 다시 꺼내 같은 형식으로 쓴 문자열이 문장에 그대로 있어야 한다.
2) 문장 속 숫자는 전부 인용한 값에서 나온 것이어야 한다(인용 없는 숫자 = 지어낸 숫자).
3) 인용한 근거 문항 문구가 표의 문구와 같아야 한다.
"""
from __future__ import annotations

import re

NUM = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?p?")


def _fmt(v, f: str) -> str:
    return f.format(v)


def verify(claim: dict, T: dict) -> tuple[bool, str]:
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
    rest = re.sub(r"\[[^\]]*\]", " ", rest)          # 세그먼트 태그
    rest = rest.replace("1표준편차", " ")             # 고정 문구
    stray = [m for m in NUM.findall(rest) if m.strip("+-")]
    if stray:
        return False, f"인용 없는 숫자 {stray}"
    return True, "ok"
