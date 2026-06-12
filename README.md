# 반려동물 질환 진단 AI

반려동물(고양이/강아지) 안구 질환, 피부 질환, 근골격계 질환을 딥러닝으로 진단하는 시스템.  
의료 이미지 특화 증강과 2단계 파인튜닝 전략을 적용한 3개의 독립 파이프라인으로 구성.

---

## 프로젝트

### 1. 안구 질환 진단 (Eye Disease Diagnosis)

고양이 안구 질환을 질환별 독립 이진 분류 모델로 진단.

- **모델**: ViT-Base-16 (질환별 독립 이진 분류, BCEWithLogitsLoss + pos_weight)
- **고양이 (완료)**: 각막궤양 / 각막부골편 / 결막염 / 비궤양성각막염 / 안검염 (5종)
- **강아지**: 미학습
- **학습**: 2단계 파인튜닝 — 백본 동결 → 전체 파인튜닝 (차등 LR)
- **추론**: CLI 기반 진단 도구 (신뢰도 임계값 0.45, EXIF 자동 보정)
- **평가 결과 (고양이, Validation)**:

| 질환 | Accuracy | AUROC |
|------|----------|-------|
| 각막궤양 | 0.832 | 0.918 |
| 각막부골편 | 0.942 | 0.976 |
| 결막염 | 0.760 | 0.807 |
| 비궤양성각막염 | 0.925 | 0.970 |
| 안검염 | 0.825 | 0.953 |
| **평균** | **0.857** | **0.925** |

### 2. 피부 질환 진단 (Skin Disease Diagnosis)

고양이 피부 병변 검출 + 분류 2단계 파이프라인.

- **Stage 1**: YOLO11l-seg — 이진 병변 검출 + 세그멘테이션 마스크 추출
- **Stage 2**: EfficientNet-B7 — 폴리곤 마스크 적용 크롭 이미지 분류 (A4/A6)
- **증강**: Albumentations (ElasticTransform, GaussNoise, GridDistortion)
- **데이터**: AI Hub 반려동물 질환 데이터셋

> 학습 실행 완료 — 중복 데이터 및 데이터 퀄리티 저하 이슈로 인해 학습 파일(best.pt) 미보존.

### 3. 근골격계 질환 분류 (Skeletal Disease Classification)

X-ray/MRI 이미지에서 반려동물 근골격계 질환 4종을 분류.

- **모델**: EfficientNet-B5 (2단계 파인튜닝)
- **클래스**: Mu03 (갈비뼈골절) / Mu05 (슬개골탈구) / Mu06 (전십자인대파열) / Mu07 (추간판질환)
- **데이터**: AI Hub 반려동물 질환 데이터셋 (JSON 어노테이션)
- **특징**: 2단계 파인튜닝, WeightedRandomSampler, AMP, EarlyStopping, ReduceLROnPlateau

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Framework | PyTorch |
| Detection / Segmentation | YOLO11l-seg |
| Classification | ViT-Base-16, EfficientNet-B5 / B7 |
| Augmentation | torchvision transforms, Albumentations |
| Optimization | AMP (fp16), Gradient Accumulation, WeightedRandomSampler |
| Evaluation | scikit-learn (classification_report, AUROC) |
| Dataset | AI Hub 반려동물 질환 데이터셋 |

---

## 디렉토리 구조

```
.
├── 01_eye_disease/
│   ├── README.md
│   ├── 1_check_structure.py       # 데이터셋 구조 확인
│   ├── 2_cleanup_unused.py        # 미사용 파일 정리
│   ├── 3_train_cat.py             # 고양이 학습 (5종 질환 자동화)
│   ├── 4_train_dog.py             # 강아지 학습 (10종 질환 자동화)
│   ├── 5_inference.py             # CLI 진단 도구
│   ├── 6_evaluate_cat.py          # 고양이 Validation 평가 (Accuracy / AUROC)
│   └── eval_results/              # confusion_matrix.png, performance.png, CSV/TXT
├── 02_skin_disease/
│   ├── README.md
│   ├── 1_eda.py ~ 11_pipeline.py  # 전처리 → 학습 → 추론 파이프라인
│   └── dataset.yaml
└── 03_skeletal_disease/
    ├── README.md
    ├── dataset.py                 # JSON 어노테이션 기반 CustomDataset
    ├── 1_train.py                 # EfficientNet-B5 2-phase 학습
    ├── 2_evaluate.py              # Classification Report + Confusion Matrix
    └── 3_inference.py             # 추론 시각화
```
