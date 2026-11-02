# Q-Loop — 문항이 매출로 검증받고 스스로 날카로워지는 설문 에이전트

LOTTE AX Challenge 2026 **과제 7 · 한국후지필름**(누적 설문·컨설팅 자료 기반 정기 설문 자동 생성·취합 / 응답 분석 기반 B2B·B2C 맞춤 매출 확대 제안·액션플랜) 예선 프로토타입.

설문이 보고서로 끝나 매출 액션과 끊겨 있으면, 어떤 문항이 실제로 매출을 움직였는지 모른 채 문항만 쌓인다.
Q-Loop 는 모든 문항에게 "누구에게 어떤 액션을 줄지 바꾸는가"로 자리를 증명하게 한다.
액션의 실제 매출 결과가 다음 분기 설문을 바꾸는 닫힌 고리다.

```
① 문항 은행 → ② 설문 구성 → (승인) → ③ 취합·코딩 → ④ 분석 → ⑤ 액션플랜(숫자 검증) → (승인) → 실행 → ERP 매출 → ⑥ 진화 → ①
```

## 3분 안에 보는 법

| 무엇 | 어디 |
|---|---|
| 한 장 요약·그림 | `portfolio/Q-Loop_portfolio.pdf` (6쪽, 수치는 `results/` 에서 자동으로 읽음) |
| 비교 전에 고정한 실험 설계와 변경 기록 | `docs/PREREG.md` |
| 실험 결과 원본 | `results/e1_real_survey.json`, `results/e2_summary.json`, `results/e3_verifier.json` |
| 진화 원장 예시(폐기·승격·새 문항과 근거) | `results/e2_bank_seed0.json`, `results/sample_plan_seed0.md` |

## 결과 (실측)

| 실험 | 결과 |
|---|---|
| E2 심은 정답 시뮬레이션 (후지필름형 3세그먼트, 8분기, 시드 20, 사전등록) | 8분기 순이익의 오라클 대비 포착률: 기존 방식(상관 동인) 3.8% · 같은 학습기에 문항 고정 23.5% · **Q-Loop 26.5%**. Q-Loop − 문항 고정 = +3.0%p [2.6, 3.3], 20/20 시드 우위. 응답률 감쇠 0 세계에서도 +2.8%p |
| E1 실제 응답 129,880건 (공개 항공 만족도 설문) | 같은 폐기 규칙으로 출장 고객 14→7문항(AUC 0.925→0.924), 개인 고객 14→4문항(0.919→0.918). 무작위 같은 개수보다 높음. 단, 결과 예측만 보면 단순 상관 상위 선택도 비슷 — Q-Loop 의 차별점은 액션 효과 판별(E2) |
| E3 검증기 | 맞는 문장 390/390 통과, 틀리게 바꾼 문장 1,528/1,528 차단 (생성기·검증기 같은 저자 → 기계적 보증 시험) |

한계: E2 세계는 직접 만든 것이라 "메커니즘이 설계대로 작동한다"는 증거이지 실제 매출 효과의 크기가 아니다.
고객이 적은 B2B 세그먼트는 신호 문항 오폐기가 약 30% (보완: 계층 모델, 순열 귀무분포 임계값).

## 재현

```bash
pip install numpy pandas scikit-learn
python3 -m pytest -q tests                       # 핵심 보증 시험 5개
python3 experiments/e2_sim_loop.py --seeds 20     # 약 30초
python3 experiments/e2_sim_loop.py --seeds 20 --decay 0 --tag _decay0
./fetch_data.sh && python3 -I experiments/e1_real_survey.py   # 공개 데이터 내려받기(SHA1 확인) 후 실행
python3 experiments/e3_verifier.py
python3 portfolio/build.py                        # 포트폴리오 PDF (google-chrome 필요)
# 본선용 OpenAI 배선(키 없이 구성·가드레일 확인)
pip install openai==3.42.1 openai-agents==0.24.0 && python3 qloop/agent_openai.py --build-only
```

## 구조

```
qloop/world.py         심은 정답 세계(하네스 전용). 에이전트 코드는 import 금지 — tests 로 막음
qloop/agent.py         세그먼트 에이전트: 설문 구성·배정·학습·진화(불변 영역 IMMUTABLE)
qloop/analyst.py       액션×문항 상호작용 회귀, 결정 기여도(max|t|), 대조군 IPS
qloop/bank.py          문항 은행: 계보·상태·기여도 이력·폐기 사유·부활 조건·원장·스냅샷
qloop/coder.py         개방형 응답 코딩(오프라인 사전) + 새 문항 템플릿
qloop/report.py        인용 가능한 숫자표와 액션플랜 문장
qloop/verifier.py      문장 숫자·근거 재계산 검증(불변)
qloop/agent_openai.py  OpenAI Agents SDK 0.24.0 배선: 도구, needs_approval, 출력 가드레일
```

## 데이터와 고지

- E1 데이터(Airline Passenger Satisfaction)는 원본 라이선스가 확인되지 않아 저장소에 넣지 않고 내려받기 스크립트만 둔다.
- E2 의 세그먼트·문항·액션은 한국후지필름 공개 사업 정보(사진·인화·키오스크·LTO·프리스케일·후지필름몰)를 참고해 만든 **가상** 설정이며, 실제 고객·매출 데이터가 아니다.
- 이 저장소는 응모자가 목표와 검증 원칙을 정하고, AI 에이전트(Claude)가 분석·구현·실험·문안 초안을 맡아 만들었다.
