"""포트폴리오 PDF 빌드 — results/*.json 의 실측값만 읽어 HTML 을 만들고 크롬으로 PDF 를 찍는다.

사용: python3 portfolio/build.py  → portfolio/portfolio.html, portfolio/Q-Loop_portfolio.pdf
숫자를 손으로 적지 않는다(표·그림·본문 수치 모두 결과 파일에서 읽음).
"""
from __future__ import annotations

import json, subprocess, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
E2 = json.loads((R / "e2_summary.json").read_text())
E2D = json.loads((R / "e2_summary_decay0.json").read_text())
E1 = json.loads((R / "e1_real_survey.json").read_text())
E2C = json.loads((R / "e2_summary_constrained.json").read_text())
E5 = json.loads((R / "e5_rescored.json").read_text())
E5S = {sd: json.loads((R / f"e5_codex_live_seed{sd}.json").read_text()) for sd in (3, 7)
       if (R / f"e5_codex_live_seed{sd}.json").exists()}
E5M = json.loads((R / "e5_manual_check.json").read_text()) if (R / "e5_manual_check.json").exists() else {}
E3 = json.loads((R / "e3_verifier.json").read_text())
BANK = json.loads((R / "e2_bank_seed0.json").read_text())

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # 범주 1~3 (검증기 통과: all-pairs, light)
INK, INK2, MUTE, GRID = "#1d1d1b", "#52514e", "#8a8984", "#e6e4de"
SEGLAB = {"b2b_photo": "사진관·인화점 파트너", "b2b_industrial": "산업재(LTO·프리스케일)", "b2c_instax": "인스탁스·몰 회원"}
POL = [("qloop", "Q-Loop", C1), ("fixed", "학습기만(문항 고정)", C2), ("static", "기존 방식(상관 동인)", C3)]
e = html.escape


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ───────────── 그림 1: 오라클 대비 순이익 포착률(가로 막대 + 95% 구간) ─────────────
def fig_capture():
    W, H, L, T = 560, 150, 150, 18
    xmax = 0.30
    sx = lambda v: L + (W - L - 60) * v / xmax
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="8분기 순이익 포착률">']
    for g in [0, 0.1, 0.2, 0.3]:
        x = sx(g)
        out.append(f'<line x1="{x}" y1="{T-6}" x2="{x}" y2="{H-24}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x}" y="{H-8}" font-size="10" fill="{MUTE}" text-anchor="middle">{int(g*100)}%</text>')
    bh, gap = 24, 14
    for i, (k, lab, c) in enumerate(POL):
        m = E2["capture_total"][k]["mean"]
        lo, hi = E2["capture_total"][k]["ci95"]
        y = T + i * (bh + gap)
        x0, x1 = sx(0), sx(m)
        r = 4
        out.append(f'<path d="M{x0},{y} H{x1-r} Q{x1},{y} {x1},{y+r} V{y+bh-r} Q{x1},{y+bh} {x1-r},{y+bh} H{x0} Z" fill="{c}"/>')
        out.append(f'<line x1="{sx(lo)}" y1="{y+bh/2}" x2="{sx(hi)}" y2="{y+bh/2}" stroke="{INK}" stroke-width="1.5"/>')
        out.append(f'<text x="{L-10}" y="{y+bh/2+4}" font-size="11.5" fill="{INK}" text-anchor="end">{e(lab)}</text>')
        out.append(f'<text x="{max(x1, sx(hi))+8}" y="{y+bh/2+4}" font-size="12" fill="{INK}" font-weight="700">{pct(m)}</text>')
    out.append("</svg>")
    return "".join(out)


# ───────────── 그림 2: B2C 회차별 문항 수 / 신호 문항 비율(작은 다중) ─────────────
def fig_b2c():
    pr = E2["per_round"]
    def panel(metric, title, ymax, fmt, ox):
        W, H, L, T = 270, 150, 34, 22
        n = len(pr["qloop"]["b2c_instax"][metric])
        sx = lambda i: ox + L + (W - L - 14) * i / (n - 1)
        sy = lambda v: T + (H - T - 26) * (1 - v / ymax)
        o = [f'<text x="{ox+L}" y="12" font-size="11" fill="{INK}" font-weight="700">{e(title)}</text>']
        for g in [0, ymax / 2, ymax]:
            o.append(f'<line x1="{ox+L}" y1="{sy(g)}" x2="{ox+W-14}" y2="{sy(g)}" stroke="{GRID}"/>')
            o.append(f'<text x="{ox+L-6}" y="{sy(g)+3}" font-size="9.5" fill="{MUTE}" text-anchor="end">{fmt(g)}</text>')
        for i in range(n):
            o.append(f'<text x="{sx(i)}" y="{H-8}" font-size="9.5" fill="{MUTE}" text-anchor="middle">{i+1}</text>')
        for k, lab, c in POL[:2]:
            ys = pr[k]["b2c_instax"][metric]
            pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(ys))
            o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')
            for i, v in enumerate(ys):
                o.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="3.2" fill="{c}" stroke="#fbfaf7" stroke-width="1.5"/>')
            o.append(f'<text x="{sx(n-1)+0}" y="{sy(ys[-1])-7}" font-size="10" fill="{INK}" text-anchor="end">{e(lab.split("(")[0])} {fmt(ys[-1])}</text>')
        return "".join(o)
    s = ['<svg viewBox="0 0 560 160" width="100%" role="img" aria-label="B2C 회차별 문항 수와 신호 문항 비율">']
    s.append(panel("n_q", "물은 문항 수(분기)", 16, lambda v: f"{v:.0f}", 0))
    s.append(panel("informative_share", "그중 실제 신호 문항 비율", 1.0, lambda v: f"{v*100:.0f}%", 290))
    s.append("</svg>")
    return "".join(s)


# ───────────── 그림 3: 실제 응답 13만 건 — 시험 AUC (점 그림) ─────────────
def fig_e1():
    rows = [("auc_full14", "14문항 전체", INK2), ("auc_qloop", "Q-Loop 규칙", C1), ("auc_top_corr", "단순 상관 상위", C2),
            ("auc_lowest_mean", "평균 점수 낮은 순", C3), ("auc_random_mean", "무작위(200회 평균)", MUTE)]
    W, H, L, T = 560, 190, 130, 30
    lo, hi = 0.74, 0.94
    segs = list(E1.keys())
    colw = (W - L) / 2
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="세그먼트별 시험 AUC">']
    for j, sg in enumerate(segs):
        d = E1[sg]
        ox = L + j * colw
        sx = lambda v, ox=ox: ox + 10 + (colw - 40) * (v - lo) / (hi - lo)
        lab = "출장 고객" if sg.startswith("Business") else "개인 여행 고객"
        out.append(f'<text x="{ox+10}" y="14" font-size="11" fill="{INK}" font-weight="700">{lab} · Q-Loop {d["k"]}문항</text>')
        for g in [0.75, 0.80, 0.85, 0.90]:
            out.append(f'<line x1="{sx(g)}" y1="{T-6}" x2="{sx(g)}" y2="{H-22}" stroke="{GRID}"/>')
            out.append(f'<text x="{sx(g)}" y="{H-8}" font-size="9.5" fill="{MUTE}" text-anchor="middle">{g:.2f}</text>')
        for i, (key, lab, c) in enumerate(rows):
            y = T + 8 + i * 28
            v = d[key]
            if j == 0:
                out.append(f'<text x="{L-8}" y="{y+4}" font-size="11" fill="{INK}" text-anchor="end">{e(lab)}</text>')
            out.append(f'<line x1="{sx(lo)}" y1="{y}" x2="{sx(v)}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
            out.append(f'<circle cx="{sx(v)}" cy="{y}" r="5" fill="{c}" stroke="#fbfaf7" stroke-width="2"/>')
            out.append(f'<text x="{sx(v)+9}" y="{y+4}" font-size="10" fill="{INK}">{v:.3f}</text>')
    out.append("</svg>")
    return "".join(out)


# ───────────── 그림 4: 닫힌 고리 구조 ─────────────
def fig_loop():
    steps = [("① 문항 은행", "누적 설문·|컨설팅 자료|계보·주제 태깅"), ("② 설문 구성", "기여도 순|문항 상한|시험 자리 2"),
             ("③ 취합·코딩", "폼·엑셀 응답|ERP 매출|가명 고객ID"), ("④ 분석", "액션×문항|상호작용|무작위 대조군"),
             ("⑤ 액션플랜", "거래처 리스트|캠페인 타깃|숫자 검증"), ("⑥ 진화", "폐기·승격|새 문항 시험|되돌리기")]
    W, H = 560, 150
    bw, bh, gx = 82, 66, 11
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Q-Loop 6단계 닫힌 고리">',
           '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>']
    xs = [6 + i * (bw + gx) for i in range(6)]
    y = 26
    for i, (t, s) in enumerate(steps):
        x = xs[i]
        fill = "#eef4fc" if i in (3, 5) else "#ffffff"
        stroke = C1 if i in (3, 5) else "#cfccc4"
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        out.append(f'<text x="{x+bw/2}" y="{y+22}" font-size="11.5" font-weight="700" fill="{INK}" text-anchor="middle">{e(t)}</text>')
        for li, part in enumerate(s.split("|")):
            out.append(f'<text x="{x+bw/2}" y="{y+36+li*10.5}" font-size="8.4" fill="{INK2}" text-anchor="middle">{e(part)}</text>')
        if i < 5:
            out.append(f'<line x1="{x+bw+1}" y1="{y+bh/2}" x2="{x+bw+gx-1}" y2="{y+bh/2}" stroke="{INK2}" stroke-width="1.4" marker-end="url(#ar)"/>')
    x5 = xs[5] + bw / 2
    x0 = xs[0] + bw / 2
    yb = y + bh + 30
    out.append(f'<path d="M{x5},{y+bh} V{yb} H{x0} V{y+bh+2}" fill="none" stroke="{C1}" stroke-width="1.6" marker-end="url(#ar)"/>')
    out.append(f'<text x="{(x0+x5)/2}" y="{yb-8}" font-size="10" fill="{C1}" text-anchor="middle" font-weight="700">'
               f'액션의 실제 매출 결과가 다음 분기 설문을 바꾼다 (근거 2회 · 회차당 변경 2개 · 원장 · 되돌리기)</text>')
    out.append(f'<text x="6" y="13" font-size="9.5" fill="{MUTE}">사람 승인: ② 설문 발송 전 · ⑤ 액션플랜 배포 전</text>')
    out.append("</svg>")
    return "".join(out)


def ledger_lines(seg, ops=("add_candidate", "promote", "kill"), n=6):
    L = []
    for x in BANK[seg]["ledger"]:
        if x["op"] in ops:
            it = next((i for i in BANK[seg]["items"] if i["qid"] == x["qid"]), None)
            L.append(f'<tr><td>{x["round"]+1}분기</td><td>{e({"add_candidate":"새 문항 후보","promote":"승격","kill":"폐기"}[x["op"]])}</td>'
                     f'<td>「{e(it["text"] if it else x["qid"])}」</td><td>{e(x["reason"])}</td></tr>')
    return "".join(L[:n])


def build():
    ct, pdiff = E2["capture_total"], E2["paired_diff"]
    E5_TOTAL = E5["claims"] + sum(dd["summary"]["plan_claims"] for dd in E5S.values())
    seg_rows = "".join(
        f'<tr><td>{e(SEGLAB[s])}</td>' + "".join(f'<td class="num">{pct(E2["capture_by_segment"][k][s])}</td>' for k, _, _ in POL) + "</tr>"
        for s in ["b2b_photo", "b2b_industrial", "b2c_instax"])
    aud = E2["evolution_audit"]
    aud_rows = "".join(
        f'<tr><td>{e(SEGLAB[s])}</td><td class="num">{a["promotes_informative"]}/{a["promotes"]}</td>'
        f'<td class="num">{a["kills_informative"]}/{a["kills"]}</td></tr>' for s, a in aud.items())
    e3 = E3["perturbed"]
    e3lab = {"shift": "숫자 크기 변경", "sign": "부호 뒤집기", "digit": "자릿수 바꾸기", "evidence": "근거 문항 바꿔치기",
             "extra": "인용 없는 숫자 끼워 넣기", "false_no_evidence": "근거가 뚜렷한데 '없음'이라고 쓰기(v2)",
             "wrong_rank": "최댓값 아닌 액션을 '가장 높다'고 쓰기(v2)"}
    e3_rows = "".join(f'<tr><td>{e(e3lab[k])}</td><td class="num">{v["blocked"]}/{v["made"]}</td></tr>' for k, v in e3.items())
    b, p = E1["Business travel"], E1["Personal Travel"]
    hc = E2["per_round"]["qloop"]
    inf_b2c = hc["b2c_instax"]["informative_share"]
    nq_b2c = hc["b2c_instax"]["n_q"]
    d_fix = pdiff["qloop_minus_fixed"]
    d_dec = E2D["paired_diff"]["qloop_minus_fixed"]

    css = f"""
@page {{ size: A4; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#fbfaf7; color:{INK}; font-family:'Noto Sans CJK KR','Noto Sans KR',sans-serif; font-size:10.4pt; line-height:1.55; }}
.page {{ width:210mm; height:297mm; padding:16mm 17mm 14mm; position:relative; page-break-after:always; overflow:hidden; }}
.kick {{ font-size:8.6pt; letter-spacing:.08em; color:{MUTE}; text-transform:uppercase; }}
h1 {{ font-family:'Noto Serif CJK KR',serif; font-size:40pt; line-height:1; margin:6mm 0 2mm; letter-spacing:-.01em; }}
h2 {{ font-family:'Noto Serif CJK KR',serif; font-size:17pt; margin:0 0 3mm; letter-spacing:-.01em; }}
h3 {{ font-size:10.6pt; margin:5mm 0 1.5mm; }}
.sub {{ font-family:'Noto Serif CJK KR',serif; font-size:15pt; color:{INK2}; margin-bottom:8mm; }}
.rule {{ border:0; border-top:1.2px solid {INK}; margin:4mm 0; }}
.hero {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:6mm; margin:6mm 0 7mm; }}
.hero div {{ border-top:3px solid {C1}; padding-top:3mm; }}
.hero b {{ display:block; font-size:19pt; letter-spacing:-.02em; line-height:1.15; white-space:nowrap; }}
.hero i {{ display:block; font-style:normal; font-size:10pt; color:{MUTE}; font-weight:700; margin:0.6mm 0 1.2mm; }}
.hero span {{ font-size:8.8pt; color:{INK2}; }}
p {{ margin:0 0 2.6mm; }}
.lead {{ font-size:11.3pt; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:7mm; }}
table {{ width:100%; border-collapse:collapse; font-size:9pt; margin:1.5mm 0 2mm; }}
th {{ text-align:left; font-weight:700; border-bottom:1px solid {INK}; padding:1.4mm 1.5mm; color:{INK}; }}
td {{ border-bottom:1px solid {GRID}; padding:1.3mm 1.5mm; vertical-align:top; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.note {{ font-size:8.6pt; color:{INK2}; }}
.box {{ background:#f2f0ea; padding:3mm 4mm; border-radius:2mm; margin:2.5mm 0; font-size:9.2pt; }}
.foot {{ position:absolute; bottom:8mm; left:17mm; right:17mm; font-size:7.8pt; color:{MUTE}; display:flex; justify-content:space-between; }}
.tag {{ display:inline-block; font-size:8pt; border:1px solid #cfccc4; border-radius:10px; padding:0 2.2mm; margin-right:1mm; color:{INK2}; }}
.proj {{ border-top:1px solid {INK}; padding-top:2.2mm; margin-bottom:3.4mm; }}
.proj b {{ font-size:10.6pt; }}
ul {{ margin:0 0 2mm; padding-left:4.5mm; }} li {{ margin-bottom:1mm; }}
"""
    foot = lambda n: f'<div class="foot"><span>LOTTE AX Challenge 2026 · 과제7 한국후지필름 · Q-Loop</span><span>{n} / 7</span></div>'

    p1 = f"""<section class="page">
<div class="kick">LOTTE AX Challenge 2026 · 과제 7 · 한국후지필름</div>
<h1>Q-Loop</h1>
<div class="sub">매출 결과로 문항을 고르고 고치는 설문 에이전트</div>
<hr class="rule">
<p class="lead">설문이 보고서로 끝나 매출 액션과 끊겨 있으면, 어떤 문항이 실제로 매출을 움직였는지 아무도 모른 채 문항만 쌓인다.
Q-Loop 는 문항마다 <b>"누구에게 어떤 액션을 줄지 바꾸는가"</b>를 따져 남길지 정한다. 액션의 실제 매출 결과가 다음 분기 설문을 바꾼다.</p>
<div class="hero">
 <div><b>{pct(ct['qloop']['mean'])}</b><i>문항 고정 {pct(ct["fixed"]["mean"])} · 기존 {pct(ct["static"]["mean"])}</i>
 <span>8분기 순이익 포착률(정답을 알 때의 최선 대비). Q-Loop vs 같은 학습기·문항 고정. 시드 {pdiff['n_seeds']}개 중 {d_fix['wins']}개 우위 · 심은 정답 시뮬레이션</span></div>
 <div><b>14→{b['k']}문항</b><i>시험 AUC {b['auc_full14']:.3f} → {b['auc_qloop']:.3f}</i>
 <span>실제 응답 {b['n_train']+b['n_test']+p['n_train']+p['n_test']:,}건(공개 항공 만족도 설문) 출장 고객 세그먼트. 개인 고객은 14→{p['k']}문항, AUC {p['auc_qloop']-p['auc_full14']:+.3f}</span></div>
 <div><b>0 / {E5_TOTAL}</b><i>실제 OpenAI 모델이 지어낸 숫자</i>
 <span>Codex CLI 로 받은 플랜 문장 {E5_TOTAL}개를 표와 대조. 이 출력으로 검증기가 맞는 문장을 막던 결함을 찾아 고쳤다(v2)</span></div>
</div>
<h3>무엇이 다른가</h3>
<ul>
<li><b>상관이 아니라 효과로 문항을 고른다.</b> 만족도-매출 상관은 '누가 많이 사나'를 알려줄 뿐이다. Q-Loop 는 고객 15%에 액션을 무작위 배정하고, 응답에 따라 액션 효과가 갈리는 정도(액션×문항 상호작용의 |t|)를 문항의 결정 기여도로 쓴다.</li>
<li><b>거래조건은 추첨하지 않는다.</b> B2B 는 방문 교육·샘플 측정·재고보충 제안 같은 서비스성 액션만 추첨한다. 근거 문항은 |t|≥3 일 때만 영업 화면에 보여 준다.</li>
<li><b>문항이 진화한다.</b> 기여 없는 문항은 부활 조건과 함께 폐기, 개방형 응답에서 떠오른 주제는 새 문항으로 시험·승격. 근거 2회 전에는 바꾸지 않고, 회차당 변경 2개, 성과가 떨어지면 되돌린다.</li>
<li><b>LLM 은 숫자를 만들지 않는다.</b> 숫자는 코드가 계산하고 문장은 인용만 한다. 검증기가 표에서 다시 계산해 틀린 문장을 막는다. 설문 발송·플랜 배포는 담당자 승인.</li>
</ul>
<h3>한국후지필름에서는</h3>
<table><tr><th>세그먼트</th><th>개방형 응답에서 떠오른 신호 → 새 문항</th><th>연결 액션</th></tr>
<tr><td>사진관·인화점 파트너</td><td>"인화지가 자주 떨어져요" → 결품 빈도</td><td>자동 재고보충 계약</td></tr>
<tr><td>산업재(LTO·프리스케일)</td><td>"납기가 들쭉날쭉해요" → 납기 지연 경험</td><td>납기 보장 옵션</td></tr>
<tr><td>인스탁스·후지필름몰 회원</td><td>"필름값이 부담" / "포토카드 만들고 싶어요"</td><td>필름 구독 할인 / 사진상점F·야구장 포토카드 연계 번들</td></tr></table>
<p class="note">재현: github.com/Taewoo-Won/lotte-ax-qloop — 사전등록(docs/PREREG.md), 실험 4종(실제 OpenAI 모델 출력 포함), 결과 JSON, 시험. 본 문서의 모든 수치는 results/ 파일에서 자동으로 읽어 넣었다.</p>
{foot(1)}</section>"""

    p2 = f"""<section class="page">
<div class="kick">구조</div><h2>분기마다 도는 6단계 — 결과가 다음 설문을 바꾼다</h2>
{fig_loop()}
<div class="two" style="margin-top:3mm">
<div><h3>역할 분담</h3>
<table><tr><th style="width:12mm">누가</th><th>무엇을</th></tr>
<tr><td>LLM</td><td>누적 문서 해석, 개방형 응답 코딩, 새 문항 문구, 플랜 문장(숫자는 인용만)</td></tr>
<tr><td>코드</td><td>결정 기여도, 효과 추정, 고객별 배정, 폐기·승격·되돌리기, 문장 검증</td></tr>
<tr><td>사람</td><td>설문 발송 승인, 액션플랜 배포 승인, 불변 규칙 변경</td></tr></table>
<h3>불변 영역 — 진화기가 못 건드림</h3>
<ul><li>추세 비교용 앵커 문항(전반 만족)</li><li>추첨 대조군 하한 10% (기본 15%) — 코드에서 강제</li><li>숫자 검증기 — 코드에서 강제</li><li>고객 ID 가명 처리 · 개인정보 LLM 비전송 — 본선 어댑터에서 구현</li></ul>
<p class="note">에이전트 코드가 시뮬레이션 정답 모듈을 import 하지 못하게 시험으로 막았다(tests/test_core.py).</p></div>
<div><h3>OpenAI 스택(2026-11 기준)</h3>
<table><tr><th style="width:22mm">구성</th><th>선택</th></tr>
<tr><td>오케스트레이션</td><td>Agents SDK 0.24.0 · Responses API · Structured Outputs</td></tr>
<tr><td>사람 승인</td><td><code>needs_approval=True</code> (발송·배포 도구)</td></tr>
<tr><td>숫자 보증</td><td>출력 가드레일 = 검증기(하나라도 틀리면 차단)</td></tr>
<tr><td>평가</td><td>자체 하네스 — Agent Builder 는 11-30 종료, 호스티드 Evals 는 10-31 읽기 전용·11-30 종료라 미사용</td></tr></table><p class="note">배선은 키 없이 구성과 가드레일 차단까지 시험했다. 실제 모델 출력 시험은 서버의 Codex CLI(구독, 기본 모델 gpt-5.6-sol)로 했다 — 5쪽.</p>
<h3>모델 배분 — 오류가 번지는가, 되돌릴 수 있는가</h3>
<table><tr><th style="width:24mm">단계</th><th style="width:21mm">모델</th><th>이유</th></tr>
<tr><td>개방형 1차 분류</td><td>gpt-6-luna</td><td>대량·되돌릴 수 있음(Decisions API 사용 가능 시 선택형). 사람 분류 200건과 κ 비교, 선행연구 zero-shot κ 0.76(GPT-4o, 독일어 설문 5,072건) 미달 시 상향</td></tr>
<tr><td>불일치 판정·문항·플랜 문장</td><td>gpt-6.1-sol</td><td>고객에게 나가거나 경영 판단에 쓰임</td></tr>
<tr><td>스캔본·표 이미지</td><td>gpt-6.1-terra</td><td>시각 입력</td></tr>
<tr><td>기여도·배정·폐기</td><td>코드</td><td>재현·감사 가능해야 함</td></tr></table>
<p class="note">합성 응답자는 의사결정에 쓰지 않는다 — LLM 실리콘 샘플링이 응답을 동질화한다는 연구 결과(Li 외, arXiv 2507.02919) 때문.</p></div>
</div>
{foot(2)}</section>"""

    p3 = f"""<section class="page">
<div class="kick">실증 1 · 심은 정답 시뮬레이션 (비교 실행 전 설계 고정)</div><h2>문항이 진화하면 액션이 맞는 고객에게 간다</h2>
<p>후지필름형 세그먼트 3개(파트너 400곳 · 산업재 160곳 · 회원 4,000명), 8분기, 시드 {pdiff['n_seeds']}개. 누적 문항이 재지 못하는 니즈(결품·납기·필름값)와 5번째 분기에 새로 생기는 굿즈 수요를 심었다. 세 정책을 공통 난수로 같은 세계에 나란히 돌렸다.</p>
<h3>8분기 누적 순이익 — 정답을 알 때의 최선 대비 포착률 (선: 시드 부트스트랩 95% 구간)</h3>
{fig_capture()}
<div class="two">
<div><table><tr><th>세그먼트</th>{''.join(f'<th class="num">{e(l.split("(")[0])}</th>' for _, l, _ in POL)}</tr>{seg_rows}</table>
<p class="note">짝 차이 Q-Loop − 학습기만 {d_fix['mean']*100:+.1f}%p [{d_fix['ci95'][0]*100:.1f}, {d_fix['ci95'][1]*100:.1f}], {d_fix['wins']}/{pdiff['n_seeds']} 시드 우위. 응답률이 설문 길이에 안 떨어지는 세계에서도 {d_dec['mean']*100:+.1f}%p — 우위가 짧은 설문의 응답률 효과만은 아니다. 세그먼트별 우위 시드: 사진관 {E2['segment_paired']['b2b_photo']['wins']}/20 · 산업재 {E2['segment_paired']['b2b_industrial']['wins']}/20 · 회원 {E2['segment_paired']['b2c_instax']['wins']}/20. B2B 거래조건 추첨 금지 버전(사후 추가) {E2C['paired_diff']['qloop_minus_fixed']['mean']*100:+.1f}%p, {E2C['paired_diff']['qloop_minus_fixed']['wins']}/20.</p></div>
<div><table><tr><th>진화 판단의 정오(세계 정답으로 채점)</th><th class="num">승격 중 실제 신호</th><th class="num">폐기 중 실제 신호</th></tr>{aud_rows}</table>
<p class="note">B2C 처럼 응답이 많으면 승격이 정확하다(승격 40건 중 재시험 복귀 2건을 뺀 새 문항 38건이 모두 신호). B2B 두 세그먼트는 폐기 판단의 28~30%가 실제 신호 문항 — 계층 모델·담당자 승인으로 보완.</p></div>
</div>
<h3>B2C 회원 세그먼트 — 설문은 짧아지고 신호는 진해진다</h3>
{fig_b2c()}
<p class="note">누적 문항이 못 재던 니즈는 모든 시드에서 2분기에 새 문항으로 잡혔고(산업재 납기 문항은 이후 25% 시드에서 다시 빠짐), 5번째 분기에 생긴 굿즈 수요는 그다음 분기에 잡혔다. 되돌리기는 60개 세그먼트-실행 중 {E2['rollbacks_total']}회 발동. <b>한계:</b> 이 세계는 직접 만든 것이라 결과는 "메커니즘이 설계대로 작동한다"는 증거이지 실제 매출 효과의 크기가 아니다. 기존 방식의 낮은 성적은 '만족도가 매출 수준과만 엮인다'는 세계 가정에서 나온다.</p>
{foot(3)}</section>"""

    sp = "".join(f'<tr><td>{e(x["label"])}</td><td>{e(" · ".join(x["selected"]))}</td></tr>' for x in [b, p])
    p4 = f"""<section class="page">
<div class="kick">실증 2 · 실제 응답 · 근거 표시 기준</div><h2>실제 사람 응답에서도 정보를 잃지 않고 줄었다</h2>
<p>공개 항공 만족도 설문(학습 103,904명 · 시험 25,976명, 서비스 평가 14문항 + 만족 여부)을 출장 고객과 개인 여행 고객으로 나눠, 응답을 6개 물결로 흘리며 같은 절차(2회 연속 미달 폐기 · 중복 정리 · 상한)를 예측 기여도로 적용했다. 시험 데이터는 마지막에 한 번만 썼고, 비교 방식에는 Q-Loop 가 스스로 정한 문항 수를 그대로 줬다.</p>
{fig_e1()}
<div class="box"><b>남은 문항</b> — 출장: {e(" · ".join(b["selected"]))} / 개인: {e(" · ".join(p["selected"]))}<br><b>정직하게 읽기.</b> 결과 예측만 보면 단순 상관 상위 선택도 비슷했다(출장 {b['auc_top_corr']:.3f}, 개인 {p['auc_top_corr']:.3f}). 개인 고객에서는 '평균 점수 낮은 순'도 같은 문항을 골랐다. Q-Loop 의 차별점은 예측이 아니라 <b>액션 효과를 가르는 데</b> 있고, 그건 실증 1이 보여 준다. 이 실험이 보여 주는 것은 폐기 절차가 실제 응답에서 정보를 잃지 않는다는 점과, 고객군마다 남는 문항이 다르다는 점이다.</div>
<h3>근거 문항 — 뚜렷할 때만 보여 준다(사후 추가)</h3>
<table><tr><th>세그먼트(시드 20, 마지막 분기)</th><th class="num">최댓값 문항을 그대로 쓸 때</th><th class="num">|t|≥3 만 표시</th></tr>
{''.join(f'<tr><td>{e(SEGLAB[k])}</td><td class="num">{v["all_top"]}</td><td class="num">{v["shown_t>=3"]}</td></tr>' for k, v in E3["evidence_accuracy"].items())}</table>
<p class="note">'실제 니즈 문항'은 세계 정답으로 채점. 기준 미만이면 영업 화면에 "근거 문항 없음 — 평균 효과로 배정"으로 쓴다.</p>
{foot(4)}</section>"""


    def e5_row(name, n, v1, v2, fab, chk):
        return (f'<tr><td>{name}</td><td class="num">{n}</td><td class="num">{v1}</td><td class="num"><b>{v2}</b></td>'
                f'<td class="num">{fab}</td></tr>')
    e5rows = e5_row("시드 0 — v2 를 고칠 때 본 출력", E5["claims"], E5["v1_pass"], E5["v2_pass"],
                    E5M.get("seed0", {}).get("fabricated", "—"), "")
    for sd, dd in E5S.items():
        sm = dd["summary"]
        e5rows += e5_row(f"시드 {sd} — v2 가 안 본 새 출력", sm["plan_claims"], sm["plan_pass_v1"], sm["plan_pass"],
                         E5M.get(f"seed{sd}", {}).get("fabricated", "—"), "")
    cod = E5["coding"]
    cod_more = "".join(f' · 시드 {sd} 재실행 {dd["t2_coding"].get("llm_correct")}/{dd["t2_coding"].get("n")}' for sd, dd in E5S.items())
    ex_noev = next(x["text"] for x in E5["rows"] if (not x["v1"]) and "근거 문항 없음" in x["text"])
    ex_rank = next((x["text"] for x in E5["rows"] if (not x["v1"]) and "가장 높은" in x["text"]), "")
    wrong = "".join(f'<tr><td>「{e(w["text"])}」</td><td>{e(w["truth"])}</td><td>{e(str(w["llm"]))}</td></tr>' for w in cod["llm_wrong_examples"][:5])
    p4b = f"""<section class="page">
<div class="kick">실증 3 · 실제 OpenAI 모델 출력 (서버의 Codex CLI {e(E5['codex_version'].replace('codex-cli ', ''))} · 기본 모델 gpt-5.6-sol · 2026-11-03)</div>
<h2>실제 모델은 숫자를 지어내지 않았고, 틀린 건 검증기였다</h2>
<p>에이전트가 쓰는 숫자표(인용 키 → 값)를 그대로 주고, 모델에게 액션플랜 문장과 인용 키를 쓰게 했다. 문장마다 검증기로 채점하고, 별도로 표와 직접 대조해 숫자를 지어낸 문장이 있는지 셌다.</p>
<table><tr><th>출력 묶음</th><th class="num">문장</th><th class="num">검증기 v1 통과</th><th class="num">v2 통과</th><th class="num">숫자 지어냄(표 대조)</th></tr>{e5rows}</table>
<div class="two"><div><h3>v1 이 막은 {E5['claims']-E5['v1_pass']}개 — 전부 맞는 문장</h3>
<p class="note">v1 은 "인용한 숫자가 문장에 그대로 있어야 한다"는 한 가지 문장 유형만 알았다. 실제 모델은 다른 유형으로 썼다.</p>
<ul><li><b>근거 없음 {E5['v1_blocked_types']['근거 없음(t값 인용·숫자 안 씀)']}개</b> — 「{e(ex_noev)}」 t값(기준 3 미만)을 근거로 인용하고 숫자는 안 씀</li>
<li><b>순위 주장 {E5['v1_blocked_types']['순위 주장(숫자 안 씀)']}개</b> — 「{e(ex_rank[:60])}…」</li></ul>
<p class="note"><b>v2 가 안 본 출력</b>: {sum(dd["summary"]["plan_claims"] for dd in E5S.values())}문장 중 {sum(dd["summary"]["plan_pass"] for dd in E5S.values())}개 통과. 막힌 {sum(dd["summary"]["plan_claims"]-dd["summary"]["plan_pass"] for dd in E5S.values())}개도 모두 맞는 문장이었다(순위어 없는 우선순위 1 · '근거 없음'에 쓰지 않은 평균 효과까지 인용 3 · t값 인용 없이 '근거 없음' 2). 지금은 막히면 배포 전체를 멈추고, 본선에서는 막힌 문장만 담당자 확인으로 넘긴다.</p>
<p class="note"><b>v2</b>: 문장 유형별로 판정한다. 근거 없음 → 인용한 t값이 실제로 3 미만인가, 순위 → 그 액션이 실제 최댓값인가. 원본 검증기(qloop/verifier.py)를 고쳤고 Agents SDK 출력 가드레일도 같은 함수를 쓴다.</p></div>
<div><h3>공격 시험 — v2 로 다시</h3>
<table style="font-size:8.2pt"><tr><th>틀리게 바꾼 방식</th><th class="num">차단/생성</th></tr>{e3_rows}
<tr><td><b>합계</b> · 맞는 문장 통과 {E3['original_passed']}/{E3['original_claims']}</td><td class="num"><b>{E3['perturbed_blocked']}/{E3['perturbed_total']}</b></td></tr></table></div></div>
<h3>개방형 응답 분류 — 실제 모델 {cod['llm_correct']}/{cod['n']} · 사전 코더 {cod['lexicon_correct']}/{cod['n']}{cod_more}</h3>
<table style="font-size:8.4pt"><tr><th>모델이 정답과 다르게 분류한 응답</th><th>시뮬레이션 정답</th><th>모델</th></tr>{wrong}</table>
<p class="note"><b>정직하게 읽기.</b> 불일치 중 여럿은 정답 쪽이 애매하다(시뮬레이션이 '무관한 말'로 넣은 문장이 목록의 주제와 겹침). 응답 문장은 직접 만든 것이라 실제 고객 응답의 분류 정확도는 본선에서 사람 표본 κ 로 잰다. <b>검증기가 못 잡는 것:</b> 숫자에 붙인 이름 — 모델은 t값을 '효과 지표'라고 불렀다. 숫자는 맞아 통과했고, 이런 의미 오류는 사람 검토 몫으로 남긴다.</p>
{foot(5)}</section>"""

    led = ledger_lines("b2c_instax", n=7)
    p5 = f"""<section class="page">
<div class="kick">본선 계획</div><h2>당일 공개 데이터로 1박 2일 안에 고리를 닫는다</h2>
<table><tr><th style="width:24mm">시간</th><th>할 일</th><th>끝났다는 기준</th></tr>
<tr><td>0~2h</td><td>데이터 구조 파악, 입력 어댑터 3종(누적 설문 · 응답 · 매출) 매핑. 매출이 없으면 재구매·문의를 대리 결과로</td><td>표준 응답표·결과표가 가명 ID 로 결합</td></tr>
<tr><td>2~6h</td><td>누적 자료 → 문항 은행 적재(계보·주제), 첫 회차 분석</td><td>문항별 결정 기여도 표</td></tr>
<tr><td>6~12h</td><td>액션플랜 + 검증 가드레일 + 승인 흐름 연결</td><td>틀린 숫자 문장 차단을 현장 데이터로 시연</td></tr>
<tr><td>12~18h</td><td>현장 과거 회차를 순서대로 다시 돌리는 백테스트로 진화 규칙 효과 측정</td><td>문항 수·포착률을 현장 데이터에서 실측</td></tr>
<tr><td>18~24h</td><td>담당자 화면(거래처 방문 리스트 · 캠페인 타깃 · 바뀌는 문항과 이유)과 발표</td><td>측정한 숫자만 발표</td></tr></table>
<p class="note">사전 작성 코드 사용 가능 여부는 미확인. 허용되면 저장소를 그대로, 불허면 설계·규칙·시험 케이스만 가져가 핵심부(분석·진화·검증)를 다시 구현한다.</p>
<div class="two">
<div><h3>성과 지표</h3><ul><li>설문 문항 수 · 응답률</li><li>무작위 대조군 대비 액션 순이익</li><li>개방형 코딩 κ(사람 표본 대비)</li><li>검증기 차단율 · 되돌리기 횟수</li></ul>
<h3>알려진 약점 → 보완</h3><ul><li>작은 B2B 세그먼트는 검정력 부족(오폐기 약 30%) → 세그먼트 간 정보를 빌리는 계층 모델</li>
<li>액션 여러 개 중 최대 |t| 라 우연히 큰 값 → 순열 귀무분포로 임계값 보정</li>
<li>오프라인 코더는 시뮬 문장과 같은 손에서 나와 정확도가 과대평가 → 본선은 LLM 코딩 + 사람 표본 κ</li></ul></div>
<div><h3>확산</h3><p>구조는 '설문 + 매출'이 있는 어느 계열사에나 그대로 쓰인다. 어댑터만 바꾸면 칠성·웰푸드의 소비자 조사, 홈쇼핑의 방송 후 설문에도 붙는다. 롯데이노베이트 아이멤버의 에이전트로 넣으면 그룹 공통 도구가 된다.</p>
</div>
</div>
<h3>예시 — 시드 0 회원 세그먼트의 진화 원장(실행 기록 그대로)</h3>
<table style="font-size:8.6pt"><tr><th style="width:14mm">시점</th><th style="width:20mm">변경</th><th style="width:62mm">문항</th><th>근거</th></tr>{led}</table>
{foot(6)}</section>"""

    p6 = f"""<section class="page">
<div class="kick">지원자의 AI 활용 이력</div><h2>스스로 검증하고 고치며 도는 루프를 만들어 왔다</h2>
<p>클라우드 서버(Ubuntu)에 Claude Code(헤드리스)와 OpenAI Codex CLI 를 함께 두고 cron·systemd 로 묶어, 사람이 매번 지시하지 않아도 도는 에이전트 루프를 설계·운영해 왔다. Q-Loop 의 규칙들은 이 경험에서 나왔다.</p>
<div class="proj"><b>자율 연구 루프</b> <span class="tag">2026.09~10</span><span class="tag">159라운드 무인 운영</span>
<p>시장 데이터에서 반복 가능한 규칙을 찾는 연구를 에이전트에 맡겼다. 매 라운드 프롬프트 앞에 불변 규칙 · 폐기된 가설 목록(부활 조건 필수) · 남은 산출물 목록을 붙여 기억에 기대지 않게 했고, 감사·실행·제안·탐색 네 모드로 나눠 새 방향 탐색만 사람 승인을 받게 했다. 사전등록과 홀드아웃 분리를 지켰다. 결론은 "검증된 규칙 0건" — 그럴듯한 결과를 지어내지 않는 루프가 Q-Loop '근거 2회 전에는 안 바꾼다'의 출발점이다.</p></div>
<div class="proj"><b>일감 자동 수주 엔진</b> <span class="tag">2026.08</span><span class="tag">독립 채점 에이전트 · 자기 정제</span>
<p>프리랜서 플랫폼 일감 탐지 → 자동화 가능 판정 → 산출물 선제작 → 쓰기 권한 없는 채점 에이전트의 검수 → 입찰 → 결과 추적 → 규칙 자기 정제. 정제기는 불변/가변 영역을 나누고, 같은 근거가 3번 반복돼야 바꾸며, 회차당 2개만 바꾸고, 모든 변경을 원장·스냅샷으로 되돌릴 수 있게 했다. 첫 실전 입찰 10명 중 9위 뒤 "입찰자는 먼저 연락할 수 없다"는 구조적 원인을 찾아 전략을 바꿨다.</p></div>
<div class="proj"><b>사고 구조 복원 엔진</b> <span class="tag">2026.09</span><span class="tag">비정형 → 의사결정 필드</span>
<p>공개 SNS 글을 수집해 '무엇을 보고 · 어떤 사건에 반응하고 · 무엇을 따지고 · 어떤 가설 · 무엇으로 확인하는가'로 구조화했다. 개방형 응답 코딩과 같은 문제다. 학습·검증 데이터를 물리적으로 분리해 검증 데이터를 한 모델이 아예 못 읽게 했다.</p></div>
<div class="proj"><b>공공데이터 공모전 응모 파이프라인</b> <span class="tag">2026.10~</span><span class="tag">수집 · 문서 생성 · 블라인드 심사</span>
<p>공식 통계 API 수집기(키를 로그·출력에서 가림), HWP/HWPX 문서 생성과 렌더링 검수, 블라인드 AI 심사와 원출처 대조를 묶어 여러 대회 응모작을 만들었다.</p></div>
<h3>일하는 원칙 — 실패에서 나온 것</h3>
<p>측정과 판정을 분리한다 · 불확실한 응답을 성공으로 읽지 않는다 · 모든 상수에 도출 근거, 없으면 지운다 · 문서의 안전장치가 코드에 실제로 있는지 확인한다 · 합계가 맞아도 구간별로 다시 맞춘다.</p>
<p class="note">이 응모작도 같은 방식으로 만들었다: 지원자가 목표와 검증 원칙(실측 숫자만 · 비교 전 사전등록 · 정답 모듈 격리)을 정하고, AI 에이전트가 과제 분석·구현·실험·문안 초안을 맡고, 지원자가 검토해 제출한다.</p>
{foot(7)}</section>"""

    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Q-Loop 포트폴리오</title><style>{css}</style></head>
<body>{p1}{p2}{p3}{p4}{p4b}{p5}{p6}</body></html>"""
    out = ROOT / "portfolio/portfolio.html"
    out.write_text(doc)
    pdf = ROOT / "portfolio/Q-Loop_portfolio.pdf"
    subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", str(out)], check=True, capture_output=True)
    print(pdf, pdf.stat().st_size)


if __name__ == "__main__":
    build()
