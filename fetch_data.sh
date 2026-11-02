#!/usr/bin/env bash
# E1 용 공개 설문 데이터(Airline Passenger Satisfaction) 내려받기 + SHA1 확인. 원본 라이선스 미확인이라 저장소에는 넣지 않는다.
set -euo pipefail
mkdir -p data/raw/airline
curl -sS -o data/raw/airline/train.csv https://raw.githubusercontent.com/msmetinsimsek/AirlinePassengerSatisfaction/main/train.csv
curl -sS -o data/raw/airline/test.csv  https://raw.githubusercontent.com/msmetinsimsek/AirlinePassengerSatisfaction/main/test.csv
echo "57dbd93dc76846470a3f3d3c8139282fd63bada6  data/raw/airline/train.csv
6e864267676e31bc8f25a295bc52fd34c4d52dd7  data/raw/airline/test.csv" | sha1sum -c -
