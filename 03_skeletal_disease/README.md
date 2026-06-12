# Skeletal Disease Classification

EfficientNet-B5 기반 반려동물 근골격계 질환 4클래스 분류 모델입니다.  
JSON 어노테이션 파일에서 질환 코드를 파싱하는 커스텀 데이터셋을 사용합니다.

## Dataset

| 항목 | 내용 |
|------|------|
| 출처 | AI Hub 반려동물 질환 데이터셋 |
| 클래스 | Mu03 (갈비뼈골절) / Mu05 (슬개골탈구) / Mu06 (전십자인대파열) / Mu07 (추간판질환) |
| 어노테이션 | JSON (Disease-Name 필드 파싱) |
| 구조 | images/ + annotations/ 쌍 → CustomDataset으로 로드 |
| 입력 크기 | 456×456 |

## Model

- **EfficientNet-B5** (`EfficientNet_B5_Weights.IMAGENET1K_V1`)
- Phase 1: 백본 동결, 헤드 학습 (20 epochs, lr=3e-4)
- Phase 2: 전체 파인튜닝 (50 epochs, lr=3e-5)

## Training Strategy

| 기법 | 내용 |
|------|------|
| WeightedRandomSampler | 클래스 불균형 대응 — 클래스별 역빈도 가중치 |
| AMP (fp16) | `torch.cuda.amp.autocast` + `GradScaler` |
| Gradient Accumulation | accumulation_steps=2 |
| Gradient Clipping | `clip_grad_norm_(max_norm=1.0)` |
| Label Smoothing | `CrossEntropyLoss(label_smoothing=0.1)` |
| RAdam Optimizer | 적응형 학습률 + 안정적 수렴 |
| ReduceLROnPlateau | mode='max', factor=0.5, patience=5, min_lr=1e-7 |
| EarlyStopping | Phase 2 patience=10 |

## Usage

```bash
pip install torch torchvision tqdm scikit-learn matplotlib

# 1. 학습 (Phase 1 → Phase 2, best.pt 자동 저장)
python 1_train.py

# 2. 평가 (Classification Report + Confusion Matrix)
python 2_evaluate.py

# 3. 추론 시각화
python 3_inference.py
```
