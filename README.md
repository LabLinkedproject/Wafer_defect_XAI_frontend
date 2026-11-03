# Wafer_defect_XAI

Ø270 mm · 다이 32 mm · 44다이 웨이퍼 모형을 3축 갠트리와 USB 카메라로 스캔하고,
결함 검출 → 패턴 판정(ADC) → 판단 근거(XAI) → 공정 피드백까지 잇는 검사 시스템.

## 실행

```bash
./venv/bin/python app.py        # http://<보드 IP>:5000
```

| 화면 | 내용 |
|---|---|
| `/scan` | 카메라 미리보기, 갠트리 이동, 자동 스캔, 웨이퍼 맵 합성 |
| `/analysis` | 검출 → ADC 판정 → XAI 근거 → 능동 추론 → 공정 피드백 |

## 구성

| 파일 | 역할 |
|---|---|
| `app.py` | 웹 서버, 카메라 |
| `scan_service.py`, `scan_routes.py`, `scan_config.py` | 스캔 콘솔 (갠트리 제어는 `~/motor_test/gantry.py`) |
| `wafer_model.py` | 웨이퍼 규격과 다이 배치 (단일 기준) |
| `wafer_stitch.py` | 촬영본 → 한 장의 웨이퍼 맵 |
| `wafer_detect.py` | 기준 다이 비교 결함 검출 (Particle · Scratch · Stain) |
| `wafer_raster.py` | 다이·구역 판정 → 모델 입력 |
| `adc/` | 패턴 판정, XAI, 능동 추론, 공정 피드백. `adc/cnn.py` 가 학습 모델(ONNX) 추론 |
| `training/` | 패턴 분류 모델 학습 절차 (`venv-train`, 런타임에는 쓰지 않음) |
| `models/wafernet.onnx` | 운용 모델과 판정선 (`wafernet.json`) |

## 시험

```bash
./venv/bin/python -m unittest tests.test_adc tests.test_stitch tests.test_detect
```
