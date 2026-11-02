#!/usr/bin/env bash
# 학습 제어. 네트워크 없이 보드 터미널에서도 그대로 쓴다.
#   ./training_ctl.sh start    시작 또는 재개
#   ./training_ctl.sh stop     현재 위치를 저장하고 멈춤 (재부팅해도 멈춘 채 유지)
#   ./training_ctl.sh status   상태
#   ./training_ctl.sh log      최근 기록
#   ./training_ctl.sh backup   지금 바로 백업
cd "$(dirname "$0")" || exit 1
case "$1" in
    start)
        rm -f models/STOP
        echo "시작합니다. 터미널이나 네트워크가 끊겨도 계속 돕니다."
        setsid nohup bash training/watchdog.sh > /dev/null 2>&1 < /dev/null &
        ;;
    stop)
        touch models/STOP
        sync
        echo "멈춤 요청을 남겼습니다. 현재 작업이 저장 지점까지 간 뒤 멈춥니다."
        ;;
    status) python3 training_status.py ;;
    log)    tail -n "${2:-30}" models/events.log ;;
    backup) python3 training/backup.py && tail -n 3 models/events.log ;;
    *)      sed -n 2,7p "$0" ;;
esac
