"""분석기 — 문항이 '누가 어떤 액션에 반응하는가'를 가려 주는지 잰다.

흔한 설문 분석은 문항과 매출 '수준'의 상관을 본다. 그러면 큰 고객이 대체로 만족도도 높다는 사실 때문에
만족도 문항이 핵심 동인처럼 보인다. 하지만 그 문항은 어떤 액션을 누구에게 줄지 정하는 데 아무 도움이 안 된다.

여기서는 매출 '변화'를 액션과 문항의 상호작용으로 회귀한다:
    g = μ + ρ·응답여부 + Σ_a α_a·A_a + Σ_q γ_q·x_q + Σ_a Σ_q δ_aq·A_a·x_q + ε
- α_a  : 액션 a 의 평균 효과(세그먼트 전체)
- δ_aq : 문항 q 의 응답에 따라 액션 a 의 효과가 달라지는 정도 = 그 문항이 액션 배정에 주는 정보
- 문항의 결정 기여도 = max_a |t(δ_aq)|
무작위 대조군(고객의 일부에 액션을 무작위 배정)이 있어서 이 계수가 상관이 아니라 효과로 읽힌다.
에이전트의 배정도 관측된 응답에만 의존하므로, 응답을 통제한 회귀에서 배정 편향이 생기지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RoundData:
    round: int
    qids: list[str]
    resp: np.ndarray                 # 응답자 인덱스
    answers: dict[str, np.ndarray]   # qid -> 응답자별 점수
    texts: list[str]
    coded: list[list[str]]
    assign: list[str | None]         # 고객별 실행 액션
    randomized: np.ndarray           # 고객별 무작위 배정 여부
    sales: np.ndarray                # 이번 분기 매출(액션 후)
    ref: np.ndarray                  # 기준 매출(에이전트가 아는 직전 평균)

    def x_matrix(self, feature_qids: list[str], n: int) -> np.ndarray:
        """고객 × 문항 표준화 점수. 안 물은 문항·무응답자는 0(평균)."""
        X = np.zeros((n, len(feature_qids)))
        for j, q in enumerate(feature_qids):
            if q in self.answers and len(self.resp) > 2:
                a = self.answers[q].astype(float)
                sd = a.std()
                if sd > 0:
                    X[self.resp, j] = (a - a.mean()) / sd
        return X


@dataclass
class Fit:
    actions: list[str]
    qids: list[str]
    alpha: dict[str, float]
    alpha_t: dict[str, float]
    delta: np.ndarray                # K × Q
    delta_t: np.ndarray              # K × Q
    n_rows: int
    n_resp_rows: int
    asked_rounds: dict[str, int] = field(default_factory=dict)

    def value(self, q: str) -> float | None:
        if self.asked_rounds.get(q, 0) == 0:
            return None
        j = self.qids.index(q)
        return float(np.max(np.abs(self.delta_t[:, j])))

    def uplift(self, X: np.ndarray) -> np.ndarray:
        """고객별 액션별 상대 효과 예측 (n × K)."""
        a = np.array([self.alpha[k] for k in self.actions])
        return a[None, :] + X @ self.delta.T


def fit_interaction(history: list[RoundData], actions: list[str], qids: list[str], n: int,
                    window: int = 3, lam_x: float = 1.0, lam_int: float = 4.0) -> Fit:
    rows_X, rows_g = [], []
    asked = {q: 0 for q in qids}
    n_resp_rows = 0
    K, Q = len(actions), len(qids)
    for rd in history[-window:]:
        X = rd.x_matrix(qids, n)
        r = np.zeros(n)
        r[rd.resp] = 1
        A = np.zeros((n, K))
        for i, a in enumerate(rd.assign):
            if a is not None:
                A[i, actions.index(a)] = 1
        inter = (A[:, :, None] * X[:, None, :]).reshape(n, K * Q)
        D = np.hstack([np.ones((n, 1)), r[:, None], A, X, inter])
        g = rd.sales / rd.ref - 1.0
        rows_X.append(D)
        rows_g.append(g)
        n_resp_rows += len(rd.resp)
        for q in qids:
            if q in rd.answers:
                asked[q] += 1
    D = np.vstack(rows_X)
    g = np.concatenate(rows_g)
    p = D.shape[1]
    pen = np.zeros(p)
    pen[2 + K: 2 + K + Q] = lam_x
    pen[2 + K + Q:] = lam_int
    XtX = D.T @ D
    M = XtX + np.diag(pen)
    Minv = np.linalg.pinv(M)
    beta = Minv @ (D.T @ g)
    resid = g - D @ beta
    dof = max(len(g) - float(np.trace(Minv @ XtX)), 1.0)
    s2 = float(resid @ resid) / dof
    cov = s2 * (Minv @ XtX @ Minv)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    t = beta / se
    al = beta[2:2 + K]
    alt = t[2:2 + K]
    delta = beta[2 + K + Q:].reshape(K, Q)
    delta_t = t[2 + K + Q:].reshape(K, Q)
    return Fit(actions, qids, dict(zip(actions, al)), dict(zip(actions, alt)), delta, delta_t,
               n_rows=len(g), n_resp_rows=n_resp_rows, asked_rounds=asked)


def ips_gain(rd: RoundData, policy_choice: list[str | None], actions: list[str], cost: dict[str, float]):
    """무작위 배정 고객만으로 '정책이 고른 액션을 했다면' 대비 '아무것도 안 했다면'의 순이익 차이를 추정(IPS).
    반환: (추정치, 표준오차, 표본 수). 롤백 판단에 쓴다."""
    idx = np.flatnonzero(rd.randomized)
    if len(idx) < 10:
        return 0.0, float("inf"), len(idx)
    p = 1.0 / (len(actions) + 1)
    contrib = []
    for i in idx:
        d = rd.sales[i] - rd.ref[i]
        took = rd.assign[i]
        v = 0.0
        if took == policy_choice[i]:
            v += (d - (cost[took] if took else 0.0)) / p
        if took is None:
            v -= d / p
        contrib.append(v)
    c = np.array(contrib)
    return float(c.mean()), float(c.std(ddof=1) / np.sqrt(len(c))), len(c)


def level_drivers(rd_answers: dict[str, np.ndarray], level: np.ndarray, top: int = 5) -> list[tuple[str, float]]:
    """기존 방식: 문항 점수와 매출 '수준'의 상관으로 동인 순위를 매긴다(비교 기준선용)."""
    out = []
    for q, a in rd_answers.items():
        if a.std() > 0 and level.std() > 0:
            out.append((q, float(np.corrcoef(a, level)[0, 1])))
    out.sort(key=lambda kv: -abs(kv[1]))
    return out[:top]
