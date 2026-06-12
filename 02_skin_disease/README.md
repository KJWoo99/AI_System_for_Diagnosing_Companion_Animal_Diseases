# 피부 질환 진단 — 2단계 파이프라인

AI Hub 반려동물 질환 데이터셋을 활용한 고양이 피부 질환 검출 + 분류 2단계 파이프라인.

> **상태**: 학습 실행 완료 — 중복 데이터 및 데이터 퀄리티 저하 이슈로 인해 학습 파일(best.pt) 미보존.

---

## 대상 클래스

| ID | 질환 | 코드 |
|----|------|------|
| 0 | 농포/여드름 | A4 |
| 1 | 결절/종괴 | A6 |

> A2(인설/비듬)는 데이터 전처리 단계에서 제외 — 두 모델 모두 학습하지 않음.  
> 정상(무증상)은 별도 클래스 없이 EfficientNet 신뢰도 임계값(0.45)으로 처리.

---

## 아키텍처

```
입력 이미지 (1920×1080)
    │
    ▼
[Stage 1: YOLO11l-seg + TTA]
이진 검출 — "병변 있음 (class 0)" vs "병변 없음 (empty label)"
    │
    ├─ 병변 없음 ────────────────────── "무증상" 반환
    │
    └─ 병변 발견 → bbox + 폴리곤 마스크
                        │
                        ▼
              마스크 크롭
              (병변 외부 → grey=128 / bbox + 15% 패딩)
                        │
                        ▼
           [Stage 2: EfficientNet-B7]
             (600×600 크롭 → A4/A6 확률)
                        │
                        ▼
              max(prob) ≥ 0.45 → 클래스 + bbox 반환
              max(prob) < 0.45 → 검출 무시
              모든 검출 무시 → "무증상"
```

> YOLO 신뢰도는 참고용으로만 출력.  
> 최종 클래스 및 증상 여부 판단은 EfficientNet 확률만 사용.

---

## Stage 1: YOLO11l-seg

이진 병변 검출 — 병변 위치 파악 (유형 분류 X).

**주요 하이퍼파라미터:**

| 파라미터 | 값 | 이유 |
|---------|-----|------|
| `imgsz` | 1280 | 1920×1080 입력 처리, 병변 세부 정보 포착 |
| `batch` | 4 | 16GB VRAM 안전 범위 |
| `nbs` | 64 | Gradient Accumulation (대배치 효과) |
| `epochs` | 200 | 충분한 수렴 보장 |
| `patience` | 30 | 조기 종료 |
| `optimizer` | AdamW | 안정적 수렴 |
| `cos_lr` | True | 5 에폭 웜업 + 코사인 어닐링 |
| `amp` | True | Mixed Precision fp16 |
| `mosaic` | 1.0 | 소형 병변 검출 강화 |
| `copy_paste` | 0.3 | 세그멘테이션 특화 증강 |
| `cls_pw` | 1.5 | 증상/무증상 불균형 처리 |

**출력:** `runs/pet_skin_seg/weights/best.pt`

---

## Stage 2: EfficientNet-B7

마스크 크롭 이미지를 A4(농포) 또는 A6(결절)로 분류.

**2단계 파인튜닝:**

| 단계 | 에폭 | 학습 레이어 | LR |
|------|------|------------|-----|
| Phase 1 (웜업) | 1–10 | 분류기만 | 3e-4 |
| Phase 2 (전체 파인튜닝) | 11–80 | 전체 레이어 | 3e-5 |

**주요 기법:**

| 기법 | 내용 |
|------|------|
| AMP (fp16) | VRAM ~40% 절감, `GradScaler`로 언더플로우 방지 |
| WeightedRandomSampler | A4/A6 클래스 불균형 처리 |
| Label Smoothing | `CrossEntropyLoss(label_smoothing=0.1)` |
| Gradient Clipping | `clip_grad_norm_(max_norm=1.0)` |
| EarlyStopping | patience=15 |

**증강 (Albumentations):**

```python
A.ElasticTransform(alpha=1, sigma=50)   # 의료 이미지 변형
A.GridDistortion()                       # 의료 이미지 왜곡
A.GaussNoise() / A.ISONoise()           # 노이즈 증강
A.MotionBlur() / A.MedianBlur()         # 블러 증강
A.ColorJitter(brightness=0.3, ...)
A.CoarseDropout(max_holes=4)
```

**출력:**
```
runs/stage2_efficientnet/best.pt        # 11_pipeline.py에서 로드
runs/stage2_efficientnet/history.json  # 에폭별 loss/acc 로그
```

---

## 파이프라인 출력

**단일 이미지 추론:**

```bash
python 11_pipeline.py image.jpg
```

**출력 JSON:**

```json
{
  "detections": [
    {
      "final_class": "농포_여드름",
      "yolo_conf": 0.731,
      "effnet_probs": [0.81, 0.12],
      "bbox": [312.4, 201.1, 489.7, 378.3]
    }
  ],
  "is_symptomatic": true
}
```

| 필드 | 설명 |
|------|------|
| `final_class` | EfficientNet 결과 (A4: 농포 / A6: 결절) |
| `yolo_conf` | YOLO 검출 신뢰도 (참고용) |
| `effnet_probs` | EfficientNet softmax 확률 [A4, A6] |
| `bbox` | 병변 위치 [x1, y1, x2, y2] |

---

## 전처리 파이프라인

```
1_eda.py              → 데이터셋 현황 파악 (클래스 분포, 불균형 확인)
2_verify_dataset.py   → 이미지-JSON 1:1 매칭 검사 → verify_issues.txt
3_fix_dataset.py      → 잘못 배치된 파일 자동 이동 (분할 오류 수정)
4_cleanup.py          → 미사용 CSV / 라벨링된 JPG 제거
5_convert_to_yolo.py  → AI Hub JSON → YOLO seg 포맷 변환 (8:1:1 분할, A4/A6만)
6_view_yolo.py        → 변환된 폴리곤 라벨 시각화 (선택)
7_check_dataset.py    → 최종 데이터셋 통계 확인 (선택)
8_train_yolo.py       → Stage 1 학습
9_build_crops.py      → ground-truth JSON bbox로 마스크 크롭 데이터셋 생성
10_train_efficientnet.py → Stage 2 학습
11_pipeline.py        → 2단계 추론 파이프라인
```

---

## 설치

```bash
pip install ultralytics timm torch torchvision albumentations opencv-python tqdm
```

---

## 하드웨어 참고

| 항목 | 사양 |
|------|------|
| VRAM | 16GB 기준 |
| YOLO 배치 | 4 (nbs=64, Gradient Accumulation) |
| EfficientNet 배치 | 16 + AMP fp16 (VRAM ~40% 절감) |
