"""심은 정답 시뮬레이션 세계 (한국후지필름형 B2B/B2C 고객).

이 모듈은 '정답'을 가진다: 고객별 잠재 니즈, 문항의 실제 측정 성질, 액션의 실제 효과.
에이전트(qloop.agent 이하)는 이 모듈을 import 하지 않는다. 에이전트가 받는 것은
World.survey() 와 World.apply_actions() 가 돌려주는 관측값뿐이다.
정답은 실험 하네스가 채점할 때만 World.truth_* 로 꺼낸다.

세계 파라미터는 비교 실행 전에 docs/PREREG.md 에 고정했다. 바꾸면 그 문서에 기록한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import zlib
import numpy as np


@dataclass(frozen=True)
class Need:
    key: str
    label: str
    prevalence: float
    start_round: int = 0          # 이 회차부터 생기는 니즈(시장 변화)


@dataclass(frozen=True)
class Action:
    key: str
    label: str
    need: str                     # 이 액션이 듣는 니즈
    effect_pct: float             # 니즈가 있는 고객의 분기 매출 증가율(%)
    cost: float                   # 고객 1명당 비용(매출 지수 단위)


@dataclass(frozen=True)
class QSpec:
    """문항의 '진짜' 측정 성질 — 세계만 안다."""
    need: str | None              # 어느 니즈를 재는가 (None 이면 아무것도 안 잼)
    loading: float                # 니즈 신호 세기 (부호는 문항 방향)
    sat: float = 0.0              # 전반 만족(매출 수준과 엮인 혼동 요인) 적재


@dataclass
class Segment:
    key: str
    label: str
    n: int
    rr_max: float                 # 8문항일 때 응답률
    length_decay: float           # 문항 1개 늘 때 응답률 감쇠 (지수)
    base_sigma: float             # 고객 매출 규모 분산 (로그정규)
    needs: list[Need]
    actions: list[Action]
    legacy: dict[str, tuple[str, str, str, QSpec]]   # qid -> (문항, 출처, 사람이 붙인 연결 액션, 진짜 성질)
    catalog: dict[str, QSpec]     # 주제(theme) -> 그 주제로 새 문항을 만들면 갖게 될 진짜 성질
    texts: dict[str, list[str]]   # need -> 개방형 응답 문장들


# ───────────────────────── 개방형 응답 문장 ─────────────────────────
# 일부러 사전(lexicon)에 안 걸리는 우회 표현을 섞었다(약 30%). 오프라인 코더 정확도 과대평가를 줄이려는 것.
TXT = {
    "stock": ["인화지가 자주 떨어져서 손님을 돌려보낸 적이 있어요", "재고가 부족할 때 주문하면 너무 늦게 와요",
              "필름 결품 때문에 주말 장사를 놓쳤습니다", "물량 확보가 제일 어렵네요", "주문 넣으면 일주일 넘게 걸려요",
              "성수기에 소모품이 바닥나요", "결품이 잦아요", "재고 관리가 힘듭니다", "급하게 필요할 때 받을 방법이 없어요"],
    "training": ["새 장비 조작법을 잘 모르겠어요", "직원 교육이 필요합니다", "사용법 설명을 다시 듣고 싶어요",
                 "신규 프린터 세팅이 어렵습니다", "알바생이 기계를 다루기 어려워해요", "보정 소프트웨어를 제대로 못 써요",
                 "교육 자료가 있으면 좋겠어요", "기능이 많은데 활용을 못 해요"],
    "margin": ["공급가가 너무 올랐어요", "마진이 거의 안 남습니다", "가격 경쟁이 심해서 남는 게 없어요",
               "단가 인하가 필요합니다", "매입가 부담이 큽니다", "요즘 남는 게 별로 없네요", "할인 지원이 있으면 좋겠어요"],
    "newbiz": ["포토카드 같은 새 서비스를 해보고 싶어요", "셀피 부스 들이는 것 관심 있습니다", "신규 상품을 들여보고 싶어요",
               "요즘 젊은 손님용 상품이 필요해요", "네컷 사진기 도입 문의드려요", "새로운 매출원이 필요합니다"],
    "spec": ["기술 사양 문의에 답이 늦어요", "측정 결과 해석을 도와줄 엔지니어가 필요합니다", "적용 사례 자료가 부족해요",
             "샘플 테스트를 해보고 싶습니다", "기술 지원이 아쉬워요", "우리 공정에 맞는지 검증하고 싶어요"],
    "price": ["단가가 경쟁사보다 높아요", "연간 계약하면 할인되나요", "가격 부담이 있습니다", "예산 맞추기가 어려워요",
              "구매 단가를 낮추고 싶어요"],
    "leadtime": ["납기가 들쭉날쭉해요", "입고 일정이 늦어서 라인이 멈춘 적 있어요", "납기 지연이 제일 문제입니다",
                 "급한 발주 대응이 안 돼요", "리드타임이 너무 깁니다", "언제 들어오는지 알 수가 없어요"],
    "filmcost": ["필름값이 너무 비싸요", "필름 가격 때문에 자주 못 찍어요", "카메라보다 필름이 부담이에요",
                 "한 장에 천 원 넘는 게 아까워요", "필름 할인 좀 해주세요", "찍고 싶은데 아껴 쓰게 돼요",
                 "필름 정기구매 혜택 있으면 좋겠어요", "소모품 비용이 커요"],
    "gift": ["친구 생일 선물로 샀어요", "선물 포장이 있으면 좋겠어요", "기념일 선물용으로 좋아요", "선물 세트가 더 다양했으면",
             "선물하기 좋은 구성이 필요해요"],
    "upgrade": ["신모델 나오면 바로 사고 싶어요", "새 기종 정보가 궁금해요", "업그레이드할 만한 모델이 있나요",
                "신제품 출시 알림 받고 싶어요", "더 좋은 모델로 바꾸고 싶어요"],
    "print": ["폰 사진도 인화하고 싶어요", "인화 서비스 가격이 궁금해요", "사진을 뽑아서 앨범 만들고 싶어요",
              "인화 쿠폰 있으면 쓸 것 같아요", "가까운 데서 인화하고 싶어요"],
    "goods": ["포토카드 만들고 싶어요", "아이돌 포토카드처럼 굿즈로 만들고 싶어요", "굿즈 제작 서비스 있나요",
              "포카 꾸미기 세트가 있었으면", "야구장 포토카드 너무 좋았어요", "사진으로 굿즈 만들기 관심 있어요"],
}
IRRELEVANT = ["특별히 없습니다", "만족합니다", "잘 쓰고 있어요", "없음", "그냥 그래요", "포장이 깔끔했으면 좋겠어요",
              "앱이 조금 느려요", "매장 직원분이 친절했어요", "배송이 조금 늦었어요", "광고를 더 했으면", "ㅎㅎ 좋아요"]


# 문항 방향(회사가 원래 아는 정보): '+' 는 동의할수록 좋은 상태, '-' 는 동의할수록 문제·기회가 있다는 뜻
POLARITY_NEG = {"A05", "A07", "A08", "C05", "C06", "C07", "C09", "C14"}


def build_segments() -> list[Segment]:
    A = Segment(
        key="b2b_photo", label="B2B 사진관·인화점 파트너", n=400, rr_max=0.65, length_decay=0.035, base_sigma=0.6,
        needs=[Need("stock", "재고·결품", 0.30), Need("training", "교육·사용법", 0.25),
               Need("margin", "가격·마진", 0.35), Need("newbiz", "신사업 관심", 0.20)],
        actions=[Action("restock", "자동 재고보충 계약", "stock", 9.0, 2.0),
                 Action("visit_train", "운영 교육 방문", "training", 7.0, 3.0),
                 Action("rebate", "볼륨 리베이트", "margin", 6.0, 3.5),
                 Action("newbiz_kit", "신사업 키트 시연", "newbiz", 10.0, 4.0)],
        legacy={
            "A01": ("전반적으로 거래에 만족한다", "2024 파트너 만족도 조사", "rebate", QSpec(None, 0.0, 1.0)),
            "A02": ("다른 사업자에게 추천할 의향이 있다", "2024 파트너 만족도 조사", "rebate", QSpec(None, 0.0, 1.1)),
            "A03": ("담당 영업사원이 친절하다", "2024 파트너 만족도 조사", "visit_train", QSpec(None, 0.0, 0.6)),
            "A04": ("장비 사용 교육이 충분했다", "2023 장비 도입 컨설팅", "visit_train", QSpec("training", -1.2)),
            "A05": ("신규 장비 조작이 어렵다", "2025 컨설팅 노트", "visit_train", QSpec("training", 0.8)),
            "A06": ("공급가가 경쟁사 대비 합리적이다", "2024 파트너 만족도 조사", "rebate", QSpec("margin", -1.1)),
            "A07": ("최근 마진이 줄었다", "2025 컨설팅 노트", "rebate", QSpec("margin", 0.6)),
            "A08": ("포토카드·셀피 등 신규 서비스에 관심 있다", "2025 신사업 수요조사", "newbiz_kit", QSpec("newbiz", 1.3)),
            "A09": ("매장 인테리어 지원에 만족한다", "2023 장비 도입 컨설팅", "rebate", QSpec(None, 0.0)),
            "A10": ("후지필름 브랜드 이미지가 좋다", "2024 파트너 만족도 조사", "rebate", QSpec(None, 0.0, 0.5)),
            "A11": ("주문 시스템이 편리하다", "2024 파트너 만족도 조사", "restock", QSpec("stock", -0.25)),
            "A12": ("홍보물 품질에 만족한다", "2023 장비 도입 컨설팅", "rebate", QSpec(None, 0.0)),
        },
        catalog={"재고·결품": QSpec("stock", 1.3), "교육·사용법": QSpec("training", 1.0),
                 "가격·마진": QSpec("margin", 0.9), "신사업·신상품": QSpec("newbiz", 1.1),
                 "배송": QSpec(None, 0.0), "서비스·AS": QSpec(None, 0.0)},
        texts={k: TXT[k] for k in ("stock", "training", "margin", "newbiz")},
    )
    B = Segment(
        key="b2b_industrial", label="B2B 산업재 고객(LTO·프리스케일)", n=160, rr_max=0.55, length_decay=0.035,
        base_sigma=0.8,
        needs=[Need("spec", "기술지원", 0.30), Need("price", "가격", 0.30), Need("leadtime", "납기", 0.25)],
        actions=[Action("tech_support", "기술 세미나·샘플 측정 지원", "spec", 12.0, 4.0),
                 Action("annual_contract", "연간 단가 계약", "price", 8.0, 4.0),
                 Action("lead_guarantee", "납기 보장 옵션", "leadtime", 10.0, 3.0)],
        legacy={
            "B01": ("전반적으로 거래에 만족한다", "2024 고객사 만족도 조사", "annual_contract", QSpec(None, 0.0, 1.0)),
            "B02": ("기술 문의 응답에 만족한다", "2024 고객사 만족도 조사", "tech_support", QSpec("spec", -1.1)),
            "B03": ("제품 사양 자료가 충분하다", "2025 기술 컨설팅 기록", "tech_support", QSpec("spec", -0.7)),
            "B04": ("가격 경쟁력이 있다", "2024 고객사 만족도 조사", "annual_contract", QSpec("price", -1.0)),
            "B05": ("담당자가 전문적이다", "2024 고객사 만족도 조사", "tech_support", QSpec(None, 0.0, 0.6)),
            "B06": ("카탈로그 디자인이 좋다", "2023 전시회 설문", "annual_contract", QSpec(None, 0.0)),
            "B07": ("결제 조건에 만족한다", "2025 기술 컨설팅 기록", "annual_contract", QSpec("price", -0.4)),
            "B08": ("브랜드를 신뢰한다", "2023 전시회 설문", "annual_contract", QSpec(None, 0.0, 0.5)),
        },
        catalog={"납기": QSpec("leadtime", 1.2), "기술지원·사양": QSpec("spec", 1.0),
                 "가격·마진": QSpec("price", 0.9), "서비스·AS": QSpec(None, 0.0), "배송": QSpec(None, 0.0)},
        texts={k: TXT[k] for k in ("spec", "price", "leadtime")},
    )
    C = Segment(
        key="b2c_instax", label="B2C 인스탁스·후지필름몰 회원", n=4000, rr_max=0.22, length_decay=0.03, base_sigma=0.5,
        needs=[Need("filmcost", "필름 가격 부담", 0.35), Need("gift", "선물 수요", 0.25),
               Need("upgrade", "신모델 관심", 0.20), Need("print", "인화 관심", 0.20),
               Need("goods", "포토카드 굿즈 수요", 0.25, start_round=4)],
        actions=[Action("film_sub", "필름 구독 할인", "filmcost", 15.0, 3.0),
                 Action("gift_pack", "선물 패키지 캠페인", "gift", 12.0, 2.5),
                 Action("preorder", "신모델 사전예약 혜택", "upgrade", 18.0, 4.0),
                 Action("print_coupon", "인화 쿠폰", "print", 10.0, 2.0),
                 Action("goods_bundle", "포토카드 굿즈 번들", "goods", 14.0, 3.0)],
        legacy={
            "C01": ("전반적으로 만족한다", "2024 회원 만족도 조사", "film_sub", QSpec(None, 0.0, 1.0)),
            "C02": ("지인에게 추천할 의향이 있다", "2024 회원 만족도 조사", "film_sub", QSpec(None, 0.0, 1.1)),
            "C03": ("제품 디자인이 마음에 든다", "2024 회원 만족도 조사", "preorder", QSpec(None, 0.0, 0.5)),
            "C04": ("사진 화질에 만족한다", "2024 회원 만족도 조사", "print_coupon", QSpec(None, 0.0, 0.5)),
            "C05": ("선물용으로 구매한 적이 있다", "2025 구매 동기 조사", "gift_pack", QSpec("gift", 1.2)),
            "C06": ("기념일에 사진을 선물한다", "2025 구매 동기 조사", "gift_pack", QSpec("gift", 0.7)),
            "C07": ("신모델 소식에 관심이 있다", "2025 구매 동기 조사", "preorder", QSpec("upgrade", 1.1)),
            "C08": ("찍은 사진을 SNS에 공유한다", "2025 구매 동기 조사", "print_coupon", QSpec(None, 0.0)),
            "C09": ("인화 서비스를 이용할 의향이 있다", "2025 구매 동기 조사", "print_coupon", QSpec("print", 1.0)),
            "C10": ("후지필름몰 앱이 편리하다", "2024 회원 만족도 조사", "film_sub", QSpec(None, 0.0)),
            "C11": ("배송에 만족한다", "2024 회원 만족도 조사", "film_sub", QSpec(None, 0.0)),
            "C12": ("오프라인 매장을 자주 방문한다", "2023 매장 방문객 설문", "print_coupon", QSpec(None, 0.0)),
            "C13": ("최근 광고를 본 기억이 있다", "2023 매장 방문객 설문", "preorder", QSpec(None, 0.0)),
            "C14": ("필름을 자주 구매한다", "2023 매장 방문객 설문", "film_sub", QSpec("filmcost", -0.3)),
        },
        catalog={"필름가격": QSpec("filmcost", 1.3), "굿즈·포토카드": QSpec("goods", 1.2), "선물": QSpec("gift", 0.9),
                 "신모델": QSpec("upgrade", 0.9), "인화": QSpec("print", 0.9), "배송": QSpec(None, 0.0),
                 "앱·온라인몰": QSpec(None, 0.0)},
        texts={k: TXT[k] for k in ("filmcost", "gift", "upgrade", "print", "goods")},
    )
    return [A, B, C]


@dataclass
class _SegState:
    spec: Segment
    base: np.ndarray                      # 고객별 매출 규모
    sat: np.ndarray                       # 전반 만족(혼동 요인)
    needs: dict[str, np.ndarray]          # need -> 0/1
    last_sales: np.ndarray
    custom_q: dict[str, QSpec] = field(default_factory=dict)   # 에이전트가 만든 문항 qid -> 진짜 성질


class World:
    """회차(분기) 단위로 진행되는 세계. 니즈는 회차마다 10% 확률로 바뀐다."""

    TURNOVER = 0.10
    NOISE_PCT = 0.12      # 분기 매출의 잡음(규모 대비 표준편차)
    TEXT_RATE = 0.75      # 응답자가 개방형 문항에 자기 니즈를 쓸 확률

    def __init__(self, seed: int, length_decay_override: float | None = None):
        # 정책끼리 '같은 세계'에서 비교되도록 난수를 용도별 스트림으로 나눴다(공통 난수).
        # 설문 길이가 달라도 고객의 니즈·매출 잡음·응답 성향은 정책과 무관하게 같다.
        self.seed = seed
        self.rng = np.random.default_rng([seed, 0])
        self.round = 0
        self.seg: dict[str, _SegState] = {}
        self._segidx = {}
        for si, s in enumerate(build_segments()):
            self._segidx[s.key] = si
            if length_decay_override is not None:
                s.length_decay = length_decay_override
            n = s.n
            base = 100.0 * np.exp(self.rng.normal(0, s.base_sigma, n))
            sat = self.rng.normal(0, 1, n)
            needs = {nd.key: (self.rng.random(n) < nd.prevalence).astype(int) if nd.start_round == 0
                     else np.zeros(n, int) for nd in s.needs}
            st = _SegState(s, base, sat, needs, last_sales=np.zeros(n))
            st.last_sales = self._baseline(st)
            self.seg[s.key] = st

    # ── 관측 가능한 것: 세그먼트 구조(공개 정보), 직전 매출(ERP) ──
    def public_catalog(self):
        """회사가 원래 아는 정보: 세그먼트, 고객 수, 액션 메뉴와 비용, 누적 설문 문항(출처·사람이 붙인 연결)."""
        out = {}
        for k, st in self.seg.items():
            s = st.spec
            out[k] = {
                "label": s.label, "n": s.n,
                "actions": [{"key": a.key, "label": a.label, "cost": a.cost} for a in s.actions],
                "legacy": [{"qid": q, "text": t, "source": src, "linked_action": la,
                            "polarity": "-" if q in POLARITY_NEG else "+"}
                           for q, (t, src, la, _) in s.legacy.items()],
            }
        return out

    def erp_last_sales(self, seg: str) -> np.ndarray:
        return self.seg[seg].last_sales.copy()

    # ── 내부 ──
    def _baseline(self, st: _SegState) -> np.ndarray:
        return st.base * (1 + 0.15 * np.tanh(st.sat))

    def _qspec(self, st: _SegState, qid: str) -> QSpec:
        if qid in st.spec.legacy:
            return st.spec.legacy[qid][3]
        return st.custom_q[qid]

    def register_question(self, seg: str, qid: str, theme: str):
        """에이전트가 새 문항을 주제(theme)와 함께 등록. 세계는 주제로 진짜 성질을 정한다(모르는 주제면 아무것도 안 잼)."""
        st = self.seg[seg]
        st.custom_q[qid] = st.spec.catalog.get(theme, QSpec(None, 0.0))

    def response_rate(self, seg: str, n_questions: int) -> float:
        s = self.seg[seg].spec
        return float(np.clip(s.rr_max * np.exp(-s.length_decay * (n_questions - 8)), 0.02, 0.95))

    # ── 설문 ──
    def survey(self, seg: str, qids: list[str]):
        """응답자 인덱스, 문항별 1~5점 응답, 개방형 응답 문장을 돌려준다."""
        st = self.seg[seg]
        n, si, t = st.spec.n, self._segidx[seg], self.round
        rr = self.response_rate(seg, len(qids) + 1)       # +1: 개방형 문항
        u = np.random.default_rng([self.seed, 2, t, si]).random(n)
        resp = np.flatnonzero(u < rr)                     # 짧은 설문 응답자 ⊇ 긴 설문 응답자(결합)
        answers = {}
        for q in qids:
            sp = self._qspec(st, q)
            eps = np.random.default_rng([self.seed, 3, t, si, zlib.crc32(q.encode())]).normal(0, 1, n)[resp]
            z = 0.6 * sp.sat * st.sat[resp] + eps
            if sp.need is not None:
                z += 0.8 * sp.loading * (2 * st.needs[sp.need][resp] - 1)
            answers[q] = np.clip(np.rint(3 + z), 1, 5).astype(int)
        tr = np.random.default_rng([self.seed, 4, t, si]).random((n, 3))
        texts = []
        for i in resp:
            active = [k for k, v in st.needs.items() if v[i] == 1 and k in st.spec.texts]
            if active and tr[i, 0] < self.TEXT_RATE:
                k = active[int(tr[i, 1] * len(active))]
                pool = st.spec.texts[k]
            else:
                pool = IRRELEVANT
            texts.append(pool[int(tr[i, 2] * len(pool))])
        return resp, answers, texts

    # ── 액션 실행과 매출 ──
    def _uplift(self, st: _SegState, idx: np.ndarray, action_key: str | None) -> np.ndarray:
        if action_key is None:
            return np.zeros(len(idx))
        a = next(a for a in st.spec.actions if a.key == action_key)
        return st.base[idx] * a.effect_pct / 100.0 * st.needs[a.need][idx]

    def apply_actions(self, seg: str, assign: list[str | None]):
        """고객별 액션을 실행하고 이번 분기 매출(관측)을 돌려준다. 정답(실제 순효과)은 self.last_truth 에 남긴다."""
        st = self.seg[seg]
        n = st.spec.n
        idx = np.arange(n)
        up = np.zeros(n)
        cost = np.zeros(n)
        for a in st.spec.actions:
            m = np.array([x == a.key for x in assign])
            up[m] = self._uplift(st, idx[m], a.key)
            cost[m] = a.cost
        noise = np.random.default_rng([self.seed, 5, self.round, self._segidx[seg]]).normal(0, self.NOISE_PCT, n) * st.base
        sales = self._baseline(st) + up + noise
        # 오라클: 진짜 니즈를 알 때 고를 최선의 순효과
        best = np.zeros(n)
        for a in st.spec.actions:
            best = np.maximum(best, self._uplift(st, idx, a.key) - a.cost)
        self.last_truth = {"net": up - cost, "oracle": best}
        st.last_sales = sales
        return sales

    def truth_informative(self, seg: str, qid: str) -> bool:
        st = self.seg[seg]
        sp = self._qspec(st, qid)
        return sp.need is not None and abs(sp.loading) >= 0.5

    def truth_need_of(self, seg: str, qid: str):
        return self._qspec(self.seg[seg], qid).need

    def advance(self):
        """다음 분기로. 니즈 일부 교체 + 새 니즈 등장."""
        self.round += 1
        rng = np.random.default_rng([self.seed, 1, self.round])
        for st in self.seg.values():
            n = st.spec.n
            for nd in st.spec.needs:
                if self.round < nd.start_round:
                    continue
                if self.round == nd.start_round and nd.start_round > 0:
                    st.needs[nd.key] = (rng.random(n) < nd.prevalence).astype(int)
                    continue
                flip = rng.random(n) < self.TURNOVER
                fresh = (rng.random(n) < nd.prevalence).astype(int)
                st.needs[nd.key] = np.where(flip, fresh, st.needs[nd.key])
