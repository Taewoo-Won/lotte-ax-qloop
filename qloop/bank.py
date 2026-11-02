"""문항 은행 — 누적 설문·컨설팅 자료에서 온 문항과 에이전트가 새로 만든 문항을 한곳에서 관리한다.

문항마다 출처(계보), 상태(active/candidate/killed), 회차별 '결정 기여도' 이력, 폐기 사유와 부활 조건을 남긴다.
모든 변경은 원장(ledger)에 근거와 함께 기록되고, 스냅샷으로 되돌릴 수 있다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class Item:
    qid: str
    text: str
    source: str                      # 예: "2024 파트너 만족도 조사", "개방형 응답 주제 '재고·결품'(R1)"
    status: str = "active"           # active | candidate | killed
    anchor: bool = False             # 시계열 추적용 고정 문항 — 불변 영역, 지우지 않는다
    theme: str | None = None         # 개방형 응답에서 태어난 문항이면 그 주제
    polarity: str = "+"
    linked_action: str | None = None # 사람이 붙여 둔 연결(기존 방식용 메타데이터)
    born_round: int = 0
    values: list[tuple[int, float]] = field(default_factory=list)  # (회차, 결정 기여도)
    killed_round: int | None = None
    kill_reason: str | None = None
    revival: str | None = None       # 부활 조건(사람이 읽는 문장)
    theme_share_at_kill: float | None = None

    def last_values(self, k: int) -> list[float]:
        return [v for _, v in self.values[-k:]]


@dataclass
class LedgerEntry:
    round: int
    op: str                          # kill | add_candidate | promote | retest | rollback
    qid: str
    reason: str
    evidence: dict


class Bank:
    def __init__(self, segment: str):
        self.segment = segment
        self.items: dict[str, Item] = {}
        self.ledger: list[LedgerEntry] = []
        self._next = 1

    # ── 생성 ──
    @classmethod
    def from_legacy(cls, segment: str, legacy: list[dict], anchors: set[str], tagger=None) -> "Bank":
        """누적 문항을 은행에 넣으면서 같은 주제 어휘로 태깅한다(이미 재는 주제에 새 문항을 또 만들지 않게)."""
        b = cls(segment)
        for q in legacy:
            th = None
            if tagger is not None:
                labels = tagger.code([q["text"]])[0]
                th = labels[0] if labels else None
            b.items[q["qid"]] = Item(qid=q["qid"], text=q["text"], source=q["source"], anchor=q["qid"] in anchors,
                                     polarity=q.get("polarity", "+"), linked_action=q.get("linked_action"), theme=th)
        return b

    def new_qid(self) -> str:
        qid = f"N{self.segment.split('_')[-1][:3].upper()}{self._next:02d}"
        self._next += 1
        return qid

    # ── 조회 ──
    def by_status(self, status: str) -> list[Item]:
        return [i for i in self.items.values() if i.status == status]

    def themes_in_play(self) -> set[str]:
        return {i.theme for i in self.items.values() if i.theme and i.status in ("active", "candidate")}

    # ── 변경(전부 원장에 남김) ──
    def log(self, rnd: int, op: str, qid: str, reason: str, **evidence):
        self.ledger.append(LedgerEntry(rnd, op, qid, reason, evidence))

    def snapshot(self) -> dict:
        return {"items": copy.deepcopy(self.items), "next": self._next}

    def restore(self, snap: dict):
        self.items = copy.deepcopy(snap["items"])
        self._next = snap["next"]
