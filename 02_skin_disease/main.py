"""
반려묘 피부질환 탐지 — 투트랙 파이프라인 (A4/A6 2클래스)

══════════════════════════════════════════════════════
 실행 순서
══════════════════════════════════════════════════════

  5_convert_to_yolo.py     JSON 어노테이션 → YOLO 포맷 변환 (A4/A6만 포함)
  8_train_yolo.py          YOLO11l-seg 학습 (Stage 1)
  9_build_crops.py         bbox/polygon → crop 이미지 생성 (Stage 2 준비)
  10_train_efficientnet.py EfficientNet-B7 학습 (Stage 2)
  11_pipeline.py           투트랙 추론 / YOLO 평가

══════════════════════════════════════════════════════
 아키텍처
══════════════════════════════════════════════════════

  이미지
    │
    ▼
  [Stage 1: YOLO11l-seg + TTA]
  바이너리 검출 — 병변 유무만 판단 (클래스 구분 X)
    │
    ├─ 병변 없음 → "무증상" 반환
    │
    └─ 병변 있음 → bbox + polygon 마스크
                      │
                      ▼
               Masked Crop (배경 → 회색 128)
                      │
                      ▼
         [Stage 2: EfficientNet-B7]
         A4(농포/여드름) vs A6(결절/종괴) 분류
                      │
                      ▼
           max(prob) ≥ 0.45 → 클래스 + bbox 반환
           max(prob) < 0.45 → 검출 무시 → "무증상"

  ※ A2(인설/비듬)는 데이터 전처리 단계에서 제외.
  ※ 최종 판단은 EfficientNet 확률만 사용 (YOLO 신뢰도는 참고용).

══════════════════════════════════════════════════════
 설치
══════════════════════════════════════════════════════

  pip install ultralytics timm torch torchvision albumentations opencv-python tqdm
"""
