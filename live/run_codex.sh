#!/usr/bin/env bash
# VPS 에서 실제 OpenAI 모델(Codex CLI, ChatGPT 구독 로그인)로 E5 를 돌린다. 키 불필요. 약 5~10분.
cd "$(dirname "$0")/.." || exit 1
export PATH="$PATH:/root/.local/bin"
python3 -c "import numpy, pandas" 2>/dev/null || pip3 install -q --break-system-packages numpy pandas
command -v codex >/dev/null || { echo "codex 없음 — /root/.local/bin/codex 확인"; exit 1; }
echo "시작 $(date '+%H:%M:%S')"
python3 -u experiments/e5_codex_live.py
echo "끝 $(date '+%H:%M:%S')"
# 결과를 저장소에 올리기 시도(자격증명이 없으면 실패해도 괜찮다 — 요약만 채팅에 붙이면 된다)
git add results/e5_codex_live.json >/dev/null 2>&1
git -c user.name="Taewoo-Won" -c user.email="Taewoo-Won@users.noreply.github.com" commit -q -m "E5: Codex CLI 실제 모델 출력 기록" >/dev/null 2>&1
timeout 60 git push -q origin main >/dev/null 2>&1 && echo "푸시 성공" || echo "푸시 안 됨(정상) — 위 요약만 붙여줘"
