#!/usr/bin/env bash
# 상태 화면과 학습 절차를 띄운다. 이미 돌고 있으면 바로 끝나므로 여러 번 불러도 안전하다.
cd "$(dirname "$0")/.." || exit 1
mkdir -p models
if ! ss -ltn 2>/dev/null | grep -q ':5001 '; then
    setsid nohup ./venv/bin/python training_monitor.py >> models/monitor.log 2>&1 < /dev/null &
fi
[ -e models/STOP ] && exit 0
[ -e models/PIPELINE_DONE ] && exit 0
exec nice -n 5 ./venv-train/bin/python -m training.pipeline >> models/pipeline.log 2>&1
