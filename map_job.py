"""
map_job.py
웨이퍼 맵 합성과 결함 검출을 별도 프로세스로 실행한다. 웹 서버는 이 프로세스를 띄우고 진행 줄을 읽는다.

합성은 한 번에 수백 MB 를 쓰고, 같은 프로세스에서 되풀이하면 메모리를 다 돌려주지 않는다.
보드(3.8 GB)에서 웹 서버가 합성 세 번 만에 1.5 GB 를 넘겨 강제 종료된 뒤(2026-11-03) 따로 돌리게 했다.

출력 줄: 'phase <stitch|detect>', 'log <메시지>', 실패하면 'error <메시지>' 와 종료 코드 1.
실행: ./venv/bin/python map_job.py <스캔 폴더> [--flip-y] [--scale-hint=<px/갠트리 mm>]
"""
import os
import sys


def say(kind: str, msg: str = ""):
    print(f"{kind} {msg}", flush=True)


def main() -> int:
    scan_dir = sys.argv[1]
    try:
        import wafer_stitch
        from adc import scan_input
        say("phase", "stitch")
        hint = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--scale-hint=")), None)
        wafer_stitch.build(scan_dir, log=lambda m: say("log", m), flip_y="--flip-y" in sys.argv, scale_hint=hint)
        say("phase", "detect")
        say("log", "결함 검출")
        try:
            scan_input.detect(os.path.basename(scan_dir.rstrip("/")))   # 분석 화면이 바로 쓰도록 미리 계산
        except Exception as e:                                          # 맵은 이미 저장됐다. 분석 화면이 다시 시도한다
            say("log", f"결함 검출 실패 ({e})")
    except Exception as e:
        say("error", f"{type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
