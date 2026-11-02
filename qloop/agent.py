"""Q-Loop 세그먼트 에이전트 — 설문 구성 → 응답 취합 → 분석 → 액션 배정 → 결과 학습 → 문항 은행 진화.

정책 셋을 같은 인터페이스로 둔다(실험에서 같은 세계에 나란히 돌린다):
- StaticAgent  (기존 방식) : 누적 문항 전부를 매번 묻고, 매출 '수준'과 상관이 큰 문항 중 점수가 나쁜 곳에
                            사람이 붙여 둔 액션을 건다. 무작위 대조군·학습 없음.
- QLoopAgent(evolve=False) : 같은 학습기(상호작용 회귀 + 무작위 대조군)를 쓰되 문항은 고정.
- QLoopAgent(evolve=True)  : 전체 Q-Loop. 결정 기여도로 문항을 쉬게/폐기하고, 개방형 응답에서 새 문항을 만들어 시험한다.

불변 영역(IMMUTABLE)은 진화기가 건드리지 못한다: 앵커 문항, 무작위 대조군 하한, 검증기, 개인정보 가명처리.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .bank import Bank, Item
from .coder import LexiconCoder, theme_shares, QUESTION_TEMPLATES
from .analyst import RoundData, fit_interaction, ips_gain, level_drivers

IMMUTABLE = {
    "anchors_never_removed": True,
    "min_random_share": 0.10,
    "verifier_required": True,
    "pii_pseudonymize": True,
}


@dataclass
class Config:
    budget: int                      # 회차당 객관식 문항 상한(개방형 1개 별도)
    evolve: bool = True
    pilot_rounds: int = 2            # 처음 두 회차는 누적 문항 전부를 측정
    rand_share: float = 0.15         # 무작위 대조군 비율(불변 하한 0.10)
    kill_t: float = 1.0              # 결정 기여도 |t| 가 이 아래로
    promote_t: float = 2.0           # 새 문항 승격 기준
    min_ev: int = 2                  # 연속 이만큼 근거가 쌓여야 바꾼다
    max_kills: int = 2               # 회차당 폐기 상한
    max_adds: int = 2                # 회차당 새 문항 상한
    explore_slots: int = 2           # 새 문항·재시험 자리
    theme_share: float = 0.06        # 개방형 주제가 응답자의 이 비율 이상이면 새 문항 후보
    corr_t: float = 0.7              # 중복 문항 판정
    retest_after: int = 4            # 폐기된 누적 문항을 이 회차 뒤 한 번 재시험(시장 변화 대비)
    window: int = 3                  # 학습에 쓰는 최근 회차 수

    def __post_init__(self):
        if self.rand_share < IMMUTABLE["min_random_share"]:
            raise ValueError("무작위 대조군 비율은 불변 하한 아래로 내릴 수 없다")


class _Base:
    def __init__(self, seg_key: str, cat: dict, seed: int, seg_index: int, anchors: set[str]):
        self.seg = seg_key
        self.n = cat["n"]
        self.actions = [a["key"] for a in cat["actions"]]
        self.action_label = {a["key"]: a["label"] for a in cat["actions"]}
        self.cost = {a["key"]: a["cost"] for a in cat["actions"]}
        self.bank = Bank.from_legacy(seg_key, cat["legacy"], anchors)
        self.rng = np.random.default_rng([seed, 99, seg_index])
        self.history: list[RoundData] = []
        self.coder = LexiconCoder()
        self.ref = None
        self._sales_sum = None
        self._sales_cnt = 0

    def set_initial_sales(self, sales: np.ndarray):
        self._sales_sum = sales.astype(float).copy()
        self._sales_cnt = 1
        self.ref = sales.astype(float).copy()

    def _update_ref(self, sales: np.ndarray):
        self._sales_sum += sales
        self._sales_cnt += 1
        self.ref = self._sales_sum / self._sales_cnt


class StaticAgent(_Base):
    """기존 방식(비교 기준선)."""
    name = "기존 방식(정적 설문+상관 동인)"

    def compose(self, rnd: int) -> list[str]:
        return [i.qid for i in self.bank.items.values() if i.status == "active"]

    def assign(self, rnd, resp, answers, qids):
        n = self.n
        assign: list[str | None] = [None] * n
        drivers = level_drivers(answers, self.ref[resp], top=5)
        dq = [q for q, _ in drivers]
        for k, i in enumerate(resp):
            worst, wq = 0, None
            for q in dq:
                it = self.bank.items[q]
                a = int(answers[q][k])
                prob = a if it.polarity == "-" else 6 - a      # 높을수록 문제/기회
                if prob > worst:
                    worst, wq = prob, q
            if wq is not None and worst >= 4:
                assign[i] = self.bank.items[wq].linked_action
        return assign, np.zeros(n, bool)

    def learn(self, rd: RoundData):
        self.history.append(rd)
        self._update_ref(rd.sales)


class QLoopAgent(_Base):
    def __init__(self, seg_key, cat, seed, seg_index, anchors, cfg: Config):
        super().__init__(seg_key, cat, seed, seg_index, anchors)
        self.cfg = cfg
        self.fit = None
        self.name = "Q-Loop" if cfg.evolve else "학습기만(문항 고정)"
        self.pending = None            # (변경 회차, 스냅샷, 변경 전 이득)
        self.frozen: dict[str, int] = {}
        self.rollbacks = 0
        self.last_policy_choice: list[str | None] | None = None

    # ── 설문 구성 ──
    def compose(self, rnd: int) -> list[str]:
        items = self.bank.items.values()
        anchors = [i.qid for i in items if i.status == "active" and i.anchor]
        cands = sorted([i for i in items if i.status == "candidate"], key=lambda i: -i.born_round)
        if not self.cfg.evolve or rnd < self.cfg.pilot_rounds:
            act = [i.qid for i in items if i.status == "active" and not i.anchor]
            return anchors + act + [c.qid for c in cands][: self.cfg.explore_slots]
        explore = [c.qid for c in cands][: self.cfg.explore_slots]
        if len(explore) < self.cfg.explore_slots:
            dead = sorted([i for i in items if i.status == "killed" and i.theme is None
                           and i.killed_round is not None and rnd - i.killed_round >= self.cfg.retest_after],
                          key=lambda i: i.killed_round)
            for d in dead[: self.cfg.explore_slots - len(explore)]:
                explore.append(d.qid)
                self.bank.log(rnd, "retest", d.qid, f"폐기 후 {rnd - d.killed_round}회차 경과 — 시장 변화 확인용 재측정")
        core = self.cfg.budget - len(anchors) - len(explore)
        act = sorted([i for i in items if i.status == "active" and not i.anchor],
                     key=lambda i: -(np.mean(i.last_values(2)) if i.values else 99.0))
        return anchors + [i.qid for i in act[:max(core, 0)]] + explore

    # ── 액션 배정 ──
    def _x_now(self, resp, answers, qids_model):
        X = np.zeros((self.n, len(qids_model)))
        for j, q in enumerate(qids_model):
            if q in answers:
                a = answers[q].astype(float)
                if a.std() > 0:
                    X[resp, j] = (a - a.mean()) / a.std()
        return X

    def assign(self, rnd, resp, answers, qids):
        n = self.n
        pilot = self.fit is None
        randomized = np.ones(n, bool) if pilot else (self.rng.random(n) < self.cfg.rand_share)
        menu = self.actions + [None]
        rand_pick = [menu[k] for k in self.rng.integers(len(menu), size=n)]
        policy: list[str | None] = [None] * n
        if not pilot:
            X = self._x_now(resp, answers, self.fit.qids)
            U = self.fit.uplift(X)                                   # 상대 효과
            net = U * self.ref[:, None] - np.array([self.cost[a] for a in self.actions])[None, :]
            best = net.argmax(1)
            for i in range(n):
                policy[i] = self.actions[best[i]] if net[i, best[i]] > 0 else None
        self.last_policy_choice = policy
        assign = [rand_pick[i] if randomized[i] else policy[i] for i in range(n)]
        return assign, randomized

    # ── 학습 + 진화 ──
    def learn(self, rd: RoundData):
        self.history.append(rd)
        qset: list[str] = []
        for h in self.history[-self.cfg.window:]:
            for q in h.qids:
                if q not in qset:
                    qset.append(q)
        prev_fit = self.fit
        self.fit = fit_interaction(self.history, self.actions, qset, self.n, window=self.cfg.window)
        for q in rd.qids:
            v = self.fit.value(q)
            if v is not None:
                self.bank.items[q].values.append((rd.round, v))
        gain = None
        if prev_fit is not None and self.last_policy_choice is not None:
            gain = ips_gain(rd, self.last_policy_choice, self.actions, self.cost)
        if self.cfg.evolve:
            self._evolve(rd, gain)
        self._update_ref(rd.sales)

    def _evolve(self, rd: RoundData, gain):
        rnd, cfg, bank = rd.round, self.cfg, self.bank
        # 1) 직전 회차 변경의 롤백 판단: 변경 뒤 무작위 대조군 기준 이득이 2표준오차 넘게 떨어졌으면 되돌린다
        if self.pending is not None and gain is not None:
            ch_round, snap, before = self.pending
            if before is not None and np.isfinite(before[1]) and np.isfinite(gain[1]):
                drop = before[0] - gain[0]
                se = float(np.hypot(before[1], gain[1]))
                if drop > 2 * se:
                    changed = [e.qid for e in bank.ledger if e.round == ch_round and e.op in ("kill", "promote")]
                    bank.restore(snap)
                    for q in changed:
                        self.frozen[q] = rnd + 2
                        bank.log(rnd, "rollback", q, "변경 뒤 대조군 기준 이득이 2SE 넘게 하락",
                                 before=round(before[0], 2), after=round(gain[0], 2), se=round(se, 2))
                    self.rollbacks += 1
                    self.pending = None
                    return
            self.pending = None
        if rnd < cfg.pilot_rounds - 1:
            self._discover(rd)
            return
        snap = bank.snapshot()
        kills: list[tuple[float, str, str]] = []
        promotes: list[str] = []
        for it in list(bank.items.values()):
            if it.anchor or self.frozen.get(it.qid, -1) > rnd or it.qid not in rd.qids:
                continue
            last = it.last_values(cfg.min_ev)
            if len(last) < cfg.min_ev:
                continue
            if it.status == "candidate":
                if all(v >= cfg.promote_t for v in last):
                    promotes.append(it.qid)
                elif all(v < cfg.kill_t for v in last):
                    kills.append((max(last), it.qid, "새 문항 시험 결과 결정 기여도 부족"))
            elif it.status == "active":
                if all(v < cfg.kill_t for v in last):
                    kills.append((max(last), it.qid, f"최근 {cfg.min_ev}회 결정 기여도 |t|<{cfg.kill_t}"))
            elif it.status == "killed":    # 재시험 대상
                if all(v >= cfg.promote_t for v in it.last_values(1)):
                    promotes.append(it.qid)
                else:
                    it.killed_round = rnd   # 재시험 실패 — 다음 재시험까지 다시 기다린다
        # 중복: 응답 상관이 높은 활성 문항 쌍은 기여도가 낮은 쪽을 정리
        act = [q for q in rd.qids if bank.items[q].status == "active" and not bank.items[q].anchor
               and q in rd.answers]
        for a_i in range(len(act)):
            for b_i in range(a_i + 1, len(act)):
                qa, qb = act[a_i], act[b_i]
                xa, xb = rd.answers[qa], rd.answers[qb]
                if xa.std() == 0 or xb.std() == 0:
                    continue
                r = abs(np.corrcoef(xa, xb)[0, 1])
                if r >= cfg.corr_t:
                    va = np.mean(bank.items[qa].last_values(2)) if bank.items[qa].values else 0
                    vb = np.mean(bank.items[qb].last_values(2)) if bank.items[qb].values else 0
                    lo = qa if va < vb else qb
                    if all(k[1] != lo for k in kills):
                        kills.append((min(va, vb), lo, f"다른 문항과 응답 상관 {r:.2f} — 중복"))
        kills.sort()
        changed = False
        for v, q, why in kills[: cfg.max_kills]:
            it = bank.items[q]
            it.status, it.killed_round, it.kill_reason = "killed", rnd, why
            if it.theme:
                share = theme_shares(rd.coded).get(it.theme, 0.0)
                it.theme_share_at_kill = share
                it.revival = f"개방형 응답에서 '{it.theme}' 비율이 {max(2 * share, cfg.theme_share):.0%} 이상으로 오르면 재제안"
            else:
                it.revival = f"{cfg.retest_after}회차 뒤 탐색 자리에서 한 번 재측정해 기여도 |t|≥{cfg.promote_t}이면 복귀"
            bank.log(rnd, "kill", q, why, value=round(v, 2))
            changed = True
        for q in promotes:
            it = bank.items[q]
            was = it.status
            it.status, it.killed_round = "active", None
            bank.log(rnd, "promote", q, f"{'재시험' if was == 'killed' else '새 문항 시험'} 통과 — 결정 기여도 연속 |t|≥{cfg.promote_t}",
                     values=[round(v, 2) for v in it.last_values(cfg.min_ev)])
            changed = True
        self._discover(rd)
        if changed:
            self.pending = (rnd, snap, gain)

    def _discover(self, rd: RoundData):
        cfg, bank, rnd = self.cfg, self.bank, rd.round
        shares = theme_shares(rd.coded)
        in_play = bank.themes_in_play()
        added = 0
        for th, sh in shares.items():
            if added >= cfg.max_adds or sh < cfg.theme_share or th in in_play:
                continue
            dead = [i for i in bank.items.values() if i.theme == th and i.status == "killed"]
            if dead:
                d = dead[-1]
                need = max(2 * (d.theme_share_at_kill or 0), cfg.theme_share)
                if sh < need:
                    continue
            qid = bank.new_qid()
            bank.items[qid] = Item(qid=qid, text=QUESTION_TEMPLATES.get(th, f"'{th}' 관련 불편이 있다"),
                                   source=f"개방형 응답 주제 '{th}' (R{rnd}, 응답자의 {sh:.0%})",
                                   status="candidate", theme=th, polarity="-", born_round=rnd + 1)
            bank.log(rnd, "add_candidate", qid, f"개방형 응답에서 '{th}' 가 {sh:.0%} — 이를 재는 문항이 없음", share=round(sh, 3))
            added += 1


def make_agent(kind: str, seg_key: str, cat: dict, seed: int, seg_index: int, budget: int):
    anchors = {cat["legacy"][0]["qid"]}           # 각 세그먼트 첫 문항(전반 만족)을 시계열 앵커로 고정
    if kind == "static":
        return StaticAgent(seg_key, cat, seed, seg_index, anchors)
    if kind == "fixed":
        return QLoopAgent(seg_key, cat, seed, seg_index, anchors, Config(budget=budget, evolve=False))
    if kind == "qloop":
        return QLoopAgent(seg_key, cat, seed, seg_index, anchors, Config(budget=budget, evolve=True))
    raise ValueError(kind)
