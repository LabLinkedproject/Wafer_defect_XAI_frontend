"""
adc/knowledge.py
패턴·공정 도메인 지식. 코드가 아니라 데이터로 두어, 값을 고치거나 항목을 더해도 로직은 그대로 동작한다.

⚠ 아래 사전확률·우도는 공정 지식에 기반한 전문가 추정값이며, 실측 데이터로 학습한 값이 아니다.
"""

# WM-811K 표준 8종
PATTERNS = {
    "Center":    {"ko": "중앙 집중",   "desc": "웨이퍼 중심부 다이에 결함이 몰린 패턴"},
    "Donut":     {"ko": "도넛",        "desc": "중심을 비운 채 중간 반경에 고리 모양으로 분포"},
    "Edge-Loc":  {"ko": "엣지 국부",   "desc": "가장자리 일부 구간에만 결함이 몰린 패턴"},
    "Edge-Ring": {"ko": "엣지 링",     "desc": "가장자리를 따라 전체 둘레에 분포"},
    "Local":     {"ko": "국부 군집",   "desc": "특정 위치의 다이 묶음에 군집"},
    "Random":    {"ko": "무작위",      "desc": "뚜렷한 공간 구조 없이 산발적으로 분포"},
    "Scratch":   {"ko": "스크래치",    "desc": "한 방향의 선을 따라 여러 다이를 관통"},
    "Near-full": {"ko": "전면 불량",   "desc": "웨이퍼 대부분의 다이가 불량"},
}

DEFECT_TYPES = {
    "particle": {"ko": "파티클", "repr": "다이별 박스 + 개수"},
    "scratch":  {"ko": "스크래치", "repr": "다이별 박스 (선형)"},
    "stain":    {"ko": "스테인", "repr": "영역 폴리곤, 색 농도 = 강도"},
}

# 각 특징의 화면 표시 이름
FEATURE_LABELS = {
    "density":      "전체 불량 다이 비율",
    "center_fill":  "중앙 존 불량 점유율",
    "middle_fill":  "중간 존 불량 점유율",
    "edge_fill":    "엣지 존 불량 점유율",
    "center_share": "결함 중 중앙 존 비중",
    "middle_share": "결함 중 중간 존 비중",
    "edge_share":   "결함 중 엣지 존 비중",
    "hole":         "중앙이 빈 고리 구조",
    "angular":      "둘레 방향 균일도",
    "linearity":    "직선성",
    "line_fit":     "한 선 위에 놓인 비율",
    "cluster":      "최대 군집 비중",
    "scatter":      "군집 밖 결함 비중",
    "isolated":     "군집 밖 결함 수",
    "main_group":   "최대 군집 크기",
    "saturation":   "웨이퍼 포화도",
}

# 분류기 가중치 (로그오즈 선형 결합). bias + Σ weight × feature → sigmoid
# 학습 모델로 교체하기 전까지 쓰는 해석 가능한 규칙 기반 모델이다.
CLASS_WEIGHTS = {
    "Center": {
        "bias": -4.6,
        "weights": {"center_fill": 4.2, "center_share": 4.4, "edge_share": -3.0,
                    "linearity": -0.8, "saturation": -6.0},
    },
    "Donut": {
        "bias": -5.6,
        "weights": {"middle_fill": 4.6, "middle_share": 2.4, "hole": 2.6, "angular": 2.2,
                    "center_share": -2.2, "edge_share": -2.0, "linearity": -2.0, "saturation": -6.0},
    },
    "Edge-Loc": {
        "bias": -3.8,
        "weights": {"edge_share": 4.2, "cluster": 2.8, "angular": -4.5,
                    "edge_fill": 1.0, "linearity": -0.4, "saturation": -6.0},
    },
    "Edge-Ring": {
        "bias": -6.2,
        "weights": {"edge_fill": 5.0, "edge_share": 3.0, "angular": 2.6,
                    "center_share": -2.0, "linearity": -1.0, "saturation": -8.0},
    },
    "Local": {
        "bias": -0.9,
        "weights": {"cluster": 5.2, "angular": -3.2, "edge_share": -2.6, "center_share": -2.5, "scatter": -2.0,
                    "density": -3.0, "linearity": -2.5, "hole": -1.5, "saturation": -6.0},
    },
    "Random": {
        "bias": -7.0,
        "weights": {"isolated": 10.0, "scatter": 2.5, "main_group": -6.0, "saturation": -12.0},
    },
    "Scratch": {
        "bias": -7.0,
        "weights": {"line_fit": 5.0, "linearity": 5.0, "angular": -1.2,
                    "density": -1.5, "edge_share": -2.0},
    },
    "Near-full": {
        "bias": -8.0,
        "weights": {"density": 12.0},
    },
}

LABEL_THRESHOLD = 0.5    # 복수 라벨 판정선

# 공정 원인 가설.
#   patterns: 그 원인일 때 해당 패턴이 나타날 상대 가능성 (우도, 0~1)
#   types:    그 원인일 때 해당 결함 유형이 나타날 상대 가능성
#   params:   연결되는 레시피 파라미터
HYPOTHESES = [
    {"id": "gas_flow", "name": "가스 유량 불균일", "process": "Deposition (CVD/PVD)", "prior": 0.10,
     "patterns": {"Donut": 0.90, "Center": 0.30, "Edge-Ring": 0.15},
     "types": {"stain": 0.85, "particle": 0.15, "scratch": 0.02},
     "params": ["Gas flow"], "action": "MFC 유량 로그를 기준값과 비교하고, 샤워헤드 유량 분포를 재측정"},
    {"id": "plasma", "name": "플라즈마 밀도 편차", "process": "Etch / PECVD", "prior": 0.08,
     "patterns": {"Donut": 0.72, "Edge-Ring": 0.35, "Center": 0.25},
     "types": {"stain": 0.70, "particle": 0.25, "scratch": 0.02},
     "params": ["Plasma"], "action": "RF 전력·매칭 로그를 점검하고 챔버 시즈닝 상태 확인"},
    {"id": "chuck_temp", "name": "척 온도 불균일", "process": "Deposition / Etch", "prior": 0.07,
     "patterns": {"Donut": 0.55, "Center": 0.35, "Edge-Loc": 0.20},
     "types": {"stain": 0.75, "particle": 0.15, "scratch": 0.02},
     "params": ["Temperature"], "action": "척 온도 맵을 측정하고 히터 존별 편차 보정"},
    {"id": "rotation", "name": "웨이퍼 회전 불안정", "process": "Deposition / Coat", "prior": 0.05,
     "patterns": {"Donut": 0.40, "Edge-Ring": 0.25, "Edge-Loc": 0.25},
     "types": {"stain": 0.65, "particle": 0.15, "scratch": 0.05},
     "params": ["Rotation"], "action": "회전 속도 로그와 척 편심을 확인"},
    {"id": "ebr", "name": "EBR 불량", "process": "Photo (Edge Bead Removal)", "prior": 0.09,
     "patterns": {"Edge-Ring": 0.90, "Edge-Loc": 0.30},
     "types": {"stain": 0.55, "particle": 0.55, "scratch": 0.03},
     "params": ["Pressure"], "action": "EBR 노즐 위치와 용제 압력을 점검"},
    {"id": "edge_contact", "name": "에지 그립·링 접촉", "process": "Handling", "prior": 0.07,
     "patterns": {"Edge-Loc": 0.80, "Edge-Ring": 0.35, "Local": 0.25},
     "types": {"particle": 0.70, "scratch": 0.30, "stain": 0.10},
     "params": ["Pressure"], "action": "에지 그립 패드와 포커스 링 오염·마모 점검"},
    {"id": "showerhead", "name": "샤워헤드 중앙 막힘", "process": "Deposition", "prior": 0.06,
     "patterns": {"Center": 0.85, "Donut": 0.15},
     "types": {"stain": 0.70, "particle": 0.30, "scratch": 0.02},
     "params": ["Gas flow"], "action": "샤워헤드 중앙부 홀 막힘을 검사하고 세정"},
    {"id": "center_pressure", "name": "중앙부 압력 편차", "process": "Etch / Deposition", "prior": 0.05,
     "patterns": {"Center": 0.60, "Donut": 0.20},
     "types": {"stain": 0.60, "particle": 0.25, "scratch": 0.02},
     "params": ["Pressure"], "action": "챔버 압력 로그와 배기 밸브 상태 확인"},
    {"id": "cmp", "name": "CMP 패드·슬러리 오염", "process": "CMP", "prior": 0.08,
     "patterns": {"Scratch": 0.85, "Random": 0.15},
     "types": {"scratch": 0.90, "particle": 0.25, "stain": 0.05},
     "params": ["Pressure"], "action": "패드 컨디셔닝 이력과 슬러리 필터 교체 주기 확인"},
    {"id": "transport", "name": "이송 암 접촉", "process": "Handling", "prior": 0.08,
     "patterns": {"Scratch": 0.75, "Local": 0.30, "Edge-Loc": 0.20},
     "types": {"scratch": 0.85, "particle": 0.25, "stain": 0.03},
     "params": [], "action": "로봇 암 티칭 위치와 엔드 이펙터 오염 점검"},
    {"id": "etch_polymer", "name": "에치 챔버 폴리머 오염", "process": "Etch", "prior": 0.09,
     "patterns": {"Random": 0.85, "Near-full": 0.25},
     "types": {"particle": 0.90, "stain": 0.10, "scratch": 0.03},
     "params": ["Plasma"], "action": "챔버 세정 주기를 확인하고 PM 수행 여부 판단"},
    {"id": "clean_dry", "name": "세정·건조 불량", "process": "Cleaning", "prior": 0.07,
     "patterns": {"Local": 0.70, "Random": 0.25, "Near-full": 0.15},
     "types": {"stain": 0.80, "particle": 0.25, "scratch": 0.02},
     "params": ["Temperature"], "action": "건조 공정 조건과 약액 잔류 여부 확인"},
    {"id": "chuck_contact", "name": "척 접촉 오염", "process": "Handling", "prior": 0.05,
     "patterns": {"Local": 0.75, "Edge-Loc": 0.25},
     "types": {"particle": 0.70, "stain": 0.25, "scratch": 0.10},
     "params": [], "action": "척 표면 파티클 측정 후 세정"},
    {"id": "process_excursion", "name": "공정 전반 이탈", "process": "Process Excursion", "prior": 0.06,
     "patterns": {"Near-full": 0.95, "Random": 0.15},
     "types": {"stain": 0.50, "particle": 0.50, "scratch": 0.05},
     "params": ["Gas flow", "Temperature", "Plasma"], "action": "로트를 격리하고 직전 공정 레시피 변경 이력 전수 확인"},
]

# 사용자가 현장에서 예/아니오로 답할 수 있는 확인 항목.
#   yes: 해당 가설이 참일 때 '예'가 나올 확률. 없는 가설은 default 사용
CHECKS = [
    {"id": "mfc_log", "ask": "MFC 유량 로그에 기준 대비 편차가 있습니까?", "where": "장비 FDC 로그",
     "default": 0.08, "yes": {"gas_flow": 0.90, "showerhead": 0.35, "center_pressure": 0.25, "process_excursion": 0.45}},
    {"id": "rf_log", "ask": "RF 반사 전력 또는 매칭 값이 흔들렸습니까?", "where": "장비 FDC 로그",
     "default": 0.07, "yes": {"plasma": 0.88, "etch_polymer": 0.25, "process_excursion": 0.40}},
    {"id": "temp_map", "ask": "척 온도 맵에서 존별 편차가 관측됩니까?", "where": "온도 센서 웨이퍼",
     "default": 0.08, "yes": {"chuck_temp": 0.90, "clean_dry": 0.20, "process_excursion": 0.35}},
    {"id": "rotation_log", "ask": "회전 속도 로그에 흔들림이 있습니까?", "where": "장비 FDC 로그",
     "default": 0.06, "yes": {"rotation": 0.90, "process_excursion": 0.25}},
    {"id": "ebr_nozzle", "ask": "EBR 노즐 위치나 용제 압력이 벗어나 있습니까?", "where": "Photo 트랙 점검",
     "default": 0.05, "yes": {"ebr": 0.92}},
    {"id": "edge_wear", "ask": "에지 그립 또는 포커스 링에 마모·오염이 보입니까?", "where": "PM 육안 점검",
     "default": 0.08, "yes": {"edge_contact": 0.88, "ebr": 0.15}},
    {"id": "showerhead_scope", "ask": "샤워헤드 중앙부 홀에 막힘이 보입니까?", "where": "PM 내시경 점검",
     "default": 0.05, "yes": {"showerhead": 0.92, "gas_flow": 0.20}},
    {"id": "pad_life", "ask": "CMP 패드가 수명 한도에 가깝거나 슬러리 필터 교체가 밀렸습니까?", "where": "CMP 소모품 이력",
     "default": 0.10, "yes": {"cmp": 0.85}},
    {"id": "robot_teach", "ask": "로봇 암 티칭 위치가 어긋났거나 최근 변경됐습니까?", "where": "이송 시스템 이력",
     "default": 0.07, "yes": {"transport": 0.85, "edge_contact": 0.25, "chuck_contact": 0.20}},
    {"id": "pm_overdue", "ask": "챔버 PM 주기가 지났습니까?", "where": "PM 이력",
     "default": 0.12, "yes": {"etch_polymer": 0.85, "process_excursion": 0.30, "plasma": 0.30}},
    {"id": "dry_residue", "ask": "세정 후 잔류물 또는 물자국이 관측됩니까?", "where": "세정 후 현미경 검사",
     "default": 0.07, "yes": {"clean_dry": 0.88}},
    {"id": "recipe_change", "ask": "최근 공정 레시피가 변경됐습니까?", "where": "레시피 변경 이력",
     "default": 0.12, "yes": {"process_excursion": 0.80, "center_pressure": 0.30, "gas_flow": 0.25}},
    {"id": "same_position", "ask": "이전 로트에서도 같은 위치에 결함이 반복됐습니까?", "where": "로트 이력 비교",
     "default": 0.20, "yes": {"chuck_contact": 0.80, "transport": 0.65, "edge_contact": 0.60, "showerhead": 0.55}},
]

RECIPE_PARAMS = {
    "Gas flow":    {"affects": "막 두께 분포",   "note": "유량 편차는 반경 방향 두께 프로파일에 바로 영향을 줍니다."},
    "Temperature": {"affects": "막 균일도",      "note": "척 온도 링 편차는 결정성과 두께 균일도에 영향을 줍니다."},
    "Rotation":    {"affects": "둘레 방향 대칭", "note": "회전이 불안정하면 비대칭·환형 패턴이 생깁니다."},
    "Plasma":      {"affects": "식각·증착률",    "note": "플라즈마 밀도 분포가 존별 반응 속도를 결정합니다."},
    "Pressure":    {"affects": "스텝 커버리지",  "note": "공정 압력은 스텝 커버리지와 입자 응집에 영향을 줍니다."},
}

SEVERITY_ADVICE = {
    "Low":      "정상 범위입니다. 정기 모니터링을 유지합니다.",
    "Medium":   "이상 징후가 있습니다. 해당 존의 공정 조건을 점검합니다.",
    "High":     "결함 집중이 확인됐습니다. 즉시 공정 점검을 권고합니다.",
    "Critical": "광범위한 결함입니다. 라인 중단과 원인 조사가 필요합니다.",
}
