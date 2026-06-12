"""
inference.py - 반려동물 안구질환 진단
======================================

[실행]
  python inference.py

[흐름]
  동물 선택 (고양이/강아지)
  → 안구 이미지 경로 입력
  → 전체 질환 모델 순차 실행
  → 결과 출력 (정상 or 의심 질환+확률)

[전제]
  학습 완료 후 runs_cat/ 또는 runs_dog/ 폴더에 best.pt 존재해야 함
  python train_cat.py  →  runs_cat/{질환}_vit/best.pt
  python train_dog.py  →  runs_dog/{질환}_vit/best.pt
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import (
    vit_b_16,        ViT_B_16_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(".")

# ─────────────────────────────────────────────────────────────
# 동물별 질환 목록 & 모델 저장 위치
# ─────────────────────────────────────────────────────────────
ANIMALS = {
    "고양이": {
        "runs_dir": BASE / "runs_cat",
        "diseases": ["각막궤양", "각막부골편", "결막염", "비궤양성각막염", "안검염"],
    },
    "강아지": {
        "runs_dir": BASE / "runs_dog",
        "diseases": [
            "결막염", "색소침착성각막염", "안검내반증", "안검염",
            "안검종양", "유루증", "핵경화",
            "궤양성각막질환", "비궤양성각막질환", "백내장",
        ],
    },
}

THRESHOLD = 0.45   # 일반인 대상 → 놓치지 않도록 0.5보다 낮게

# ─────────────────────────────────────────────────────────────
# 전처리 (train_cat/dog eval transform과 동일)
# ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_image(path: str) -> torch.Tensor | None:
    """이미지 로드 + EXIF 회전 보정 + 전처리."""
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)   # 스마트폰 사진 회전 보정
        return TRANSFORM(img).unsqueeze(0)   # [1, 3, 224, 224]
    except Exception as e:
        print(f"  [오류] 이미지 로드 실패: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 모델 빌더 (train_cat/dog와 동일 구조)
# ─────────────────────────────────────────────────────────────
def build_model(model_name: str) -> nn.Module:
    if model_name == "vit":
        m    = vit_b_16(weights=None)
        in_f = m.heads.head.in_features
        m.heads = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_f, 1))
    else:
        m    = efficientnet_b4(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_f, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )
    return m


# ─────────────────────────────────────────────────────────────
# 모델 로드 (질환별 best.pt 자동 탐색)
# ─────────────────────────────────────────────────────────────
def load_models(animal: str, device: torch.device) -> dict:
    """
    runs_dir/{disease}_vit/best.pt  우선
    없으면  runs_dir/{disease}_effnet/best.pt
    반환: {disease: model} (로드 성공한 것만)
    """
    info   = ANIMALS[animal]
    models = {}
    missing = []

    for disease in info["diseases"]:
        loaded = False
        for model_name in ["vit", "effnet"]:
            pt = info["runs_dir"] / f"{disease}_{model_name}" / "best.pt"
            if pt.exists():
                m = build_model(model_name).to(device)
                m.load_state_dict(torch.load(pt, map_location=device,
                                             weights_only=True))
                m.eval()
                models[disease] = m
                loaded = True
                break
        if not loaded:
            missing.append(disease)

    if missing:
        print(f"\n  [경고] 학습된 모델 없음 (진단 제외): {', '.join(missing)}")
    return models


# ─────────────────────────────────────────────────────────────
# 단일 이미지 추론
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(models: dict, tensor: torch.Tensor,
            device: torch.device) -> list[tuple[str, float]]:
    """
    반환: [(질환명, 확률), ...]  — 확률 내림차순, threshold 초과만
    """
    tensor  = tensor.to(device)
    results = []
    for disease, model in models.items():
        logit = model(tensor).squeeze()
        prob  = torch.sigmoid(logit).item()
        if prob >= THRESHOLD:
            results.append((disease, prob))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────
def print_result(animal: str, img_path: str,
                 detected: list[tuple[str, float]]):
    emoji = "🐱" if animal == "고양이" else "🐶"
    print(f"\n  {'━'*46}")
    print(f"  {emoji}  {animal} 안구질환 검사 결과")
    print(f"  {'━'*46}")
    print(f"  이미지: {Path(img_path).name}")
    print()

    if not detected:
        print(f"  ✅  정상입니다.")
    else:
        print(f"  ⚠️   의심 질환이 발견되었습니다:")
        print()
        for disease, prob in detected:
            bar_filled = int(prob * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(f"     {disease:>12}  {prob*100:5.1f}%  {bar}")

    print()
    print(f"  ℹ️   본 결과는 참고용입니다. 정확한 진단은 수의사에게 문의하세요.")
    print(f"  {'━'*46}")


# ─────────────────────────────────────────────────────────────
# 대화형 루프
# ─────────────────────────────────────────────────────────────
def select_animal() -> str | None:
    print("\n" + "=" * 50)
    print("  반려동물 안구질환 진단 시스템")
    print("=" * 50)
    print("\n  동물을 선택하세요:")
    print("    1. 고양이")
    print("    2. 강아지")
    print("    q. 종료")
    print()
    while True:
        choice = input("  선택 > ").strip()
        if choice == "1":
            return "고양이"
        if choice == "2":
            return "강아지"
        if choice.lower() == "q":
            return None
        print("  1, 2, q 중에서 입력하세요.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  device: {device}")

    while True:
        # ── 동물 선택 ─────────────────────────────────────
        animal = select_animal()
        if animal is None:
            print("\n  종료합니다.\n")
            break

        # ── 모델 로드 ─────────────────────────────────────
        print(f"\n  [{animal}] 모델 로드 중...")
        models = load_models(animal, device)

        if not models:
            print(f"  [오류] 사용 가능한 모델이 없습니다.")
            print(f"  먼저 train_cat.py 또는 train_dog.py를 실행하세요.")
            continue

        print(f"  로드 완료: {list(models.keys())}")

        # ── 이미지 입력 루프 ──────────────────────────────
        while True:
            print(f"\n  이미지 경로를 입력하세요.")
            print(f"  (b: 동물 다시 선택 / q: 종료)")
            img_path = input("  경로 > ").strip().strip('"').strip("'")

            if img_path.lower() == "q":
                print("\n  종료합니다.\n")
                return
            if img_path.lower() == "b":
                break
            if not img_path:
                continue

            # 이미지 로드
            tensor = load_image(img_path)
            if tensor is None:
                continue

            # 추론
            detected = predict(models, tensor, device)

            # 결과 출력
            print_result(animal, img_path, detected)


if __name__ == "__main__":
    main()
