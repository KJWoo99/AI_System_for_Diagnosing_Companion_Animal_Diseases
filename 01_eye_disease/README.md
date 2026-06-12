# 안구 질환 진단 — 질환별 이진 분류

반려동물(고양이/강아지) 안구 질환을 질환별 독립 이진 분류 모델로 진단.  
서강대 ViT 논문(2024) 아키텍처를 기반으로 각 질환마다 별도 모델을 학습.

> **학습 파일**: 고양이 5종 질환 best.pt 학습 완료 (`runs_cat/` 포함).  
> 강아지 모델은 학습 데이터 미확보로 코드만 완성.

---

## 참고 논문

| | 논문 | 핵심 내용 |
|--|------|----------|
| 논문 1 | 딥러닝 기반 반려동물 안구 질환 진단 시스템 (2024.9) | EfficientNet-B0, Acc=0.92, SGD > Adam |
| 논문 2 | Vision Transformer를 활용한 반려동물 안구 질환 이미지 분류 (서강대, 2024.6) | 질환별 이진 모델, ViT-Base-16, BCE Loss |

---

## 설계 원칙 (논문 2)

> "정상 이미지에 다른 질환이 포함될 수 있으므로, 다중 클래스 분류 대신 질환별 독립 이진 분류기를 설계한다."

- 질환 1개당 모델 1개 (이진: 0 = 정상, 1 = 질환)
- 추론: 모든 모델을 단일 이미지에 적용 → 결과 통합

---

## 데이터셋

### 고양이 — 5종 (이진 분류)

| 질환 | 학습 | 테스트 |
|------|------|--------|
| 각막궤양 | 6,245 | 796 |
| 각막부골편 | 6,234 | 787 |
| 결막염 | 6,239 | 783 |
| 비궤양성 각막염 | 2,391 | 306 |
| 안검염 | 1,910 | 246 |

### 강아지 — 10종 (이진 분류 + 이진화 처리)

| 질환 | 원본 클래스 | 처리 방식 |
|------|------------|----------|
| 결막염, 색소성 각막염, 안검내반증, 안검염, 안검종양, 유루증, 핵경화 | 정상 / 질환 | 직접 사용 |
| 궤양성 각막염, 비궤양성 각막염 | 정상 / 경증 / 중증 | **이진화** (경증 + 중증 → 질환) |
| 백내장 | 정상 / 초기 / 미성숙 / 성숙 | **이진화** (초기 + 미성숙 + 성숙 → 질환) |

> 이진화 근거: 대상 사용자는 일반 반려인 — 스마트폰 사진으로 중증도 구분은 비현실적.

---

## 모델 아키텍처

### ViT-Base-16 (기본, 논문 2)

```
ViT-Base-16 (ImageNet 사전학습)
  └── heads: Dropout(0.5) → Linear(768 → 1)
Loss:      BCEWithLogitsLoss (클래스 불균형 pos_weight 적용)
Optimizer: Adam + StepLR(step=7, gamma=0.1)
```

### EfficientNet-B4 (논문 1)

```
EfficientNet-B4 (ImageNet 사전학습)
  └── classifier: Dropout(0.5) → Linear(→256) → BN → SiLU → Dropout(0.3) → Linear(256→1)
Loss:      BCEWithLogitsLoss (클래스 불균형 pos_weight 적용)
Optimizer: SGD(momentum=0.9) + StepLR(step=7, gamma=0.1)
```

### 2단계 파인튜닝

```
Phase 1 (5 에폭):   백본 동결 → 헤드만 학습
Phase 2 (45 에폭):  전체 파인튜닝, 백본 LR = 헤드 LR × 0.1
EarlyStopping:      patience=10
Gradient Accum.:    accum=2 (유효 배치 × 2)
```

### 증강

```python
transforms.RandomCrop(224)
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation(30)
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
transforms.RandomErasing(p=0.2)
```

---

## 사용법

### 학습

```bash
# 고양이 — 5종 전체 자동 학습
python 3_train_cat.py                  # ViT-Base-16 (기본)
python 3_train_cat.py --model effnet   # EfficientNet-B4

# 강아지 — 10종 전체 자동 학습
python 4_train_dog.py                  # ViT-Base-16 (기본)
python 4_train_dog.py --model effnet   # EfficientNet-B4
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `vit` | `vit` 또는 `effnet` |
| `--epochs` | 50 | 총 에폭 수 |
| `--batch` | 16 | 배치 크기 |
| `--patience` | 10 | EarlyStopping patience |

모델 체크포인트: `runs_cat/{질환명}_vit/best.pt`, `runs_dog/{질환명}_vit/best.pt`

### 추론 (CLI)

```bash
python 5_inference.py
```

```
동물 선택:
  1. 고양이    2. 강아지    q. 종료
> 1

[고양이] 모델 로딩 중...
로드 완료: ['각막궤양', '각막부골편', '결막염', '비궤양성 각막염', '안검염']

이미지 경로: C:/path/to/cat_eye.jpg

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  고양이 안구 질환 진단 결과
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠  의심 질환:
      결막염       87.3%  █████████████████░░░
      안검염       62.1%  ████████████░░░░░░░░

  이 결과는 참고용입니다. 정확한 진단은 수의사와 상담하세요.
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| 항목 | 내용 |
|------|------|
| 모델 우선순위 | vit → effnet (`runs_cat/` / `runs_dog/` 자동 탐색) |
| 임계값 | 0.45 (일반 사용자 대상 — 위음성 최소화) |
| EXIF 보정 | 스마트폰 촬영 이미지 자동 회전 보정 |

---

## 논문 vs 구현 비교

| 논문 설정 | 구현 내용 |
|----------|----------|
| 질환별 이진 모델 (논문 2) | 전체 질환 자동 학습, 질환별 best.pt 저장 |
| BCELoss (논문 2) | `BCEWithLogitsLoss` + `Linear(→1)` |
| ViT-Base-16 (논문 2) | `vit_b_16(IMAGENET1K_V1)` |
| CNN에 SGD 적용 (논문 1) | `--model effnet` 시 SGD 자동 적용 |
| StepLR step=7, γ=0.1 (논문 2) | `StepLR(step_size=7, gamma=0.1)` |
| Dropout=0.5 (논문 2) | 헤드 첫 번째 레이어에 적용 |
| BatchNorm (논문 1) | EfficientNet-B4 분류기에 추가 |
| 회전/플립/노이즈 증강 | RandomRotation(30), Flip, ColorJitter, RandomErasing |
