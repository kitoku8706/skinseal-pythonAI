"""
모델(.pth)에서 최종 분류 레이어의 out_features(=클래스 수) 추정 도구
- 다양한 체크포인트 포맷(state_dict, model_state_dict, module./model. prefix)을 처리
- EfficientNet( classifier.1.weight ) 또는 일반적 fc.weight 키를 우선 탐색

사용법 (PowerShell):
  python inspect_model.py models/best_efficientnet.pth
  # 경로 인자를 생략하면 기본값(models/best_efficientnet.pth)을 사용합니다.
"""
from __future__ import annotations
import sys
import os
import torch

DEFAULT_PATH = os.path.join("models", "best_efficientnet.pth")


def load_state_dict_any(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        sd = ckpt
    # prefix 정리
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v
    return cleaned


def infer_num_classes(state_dict: dict) -> int | None:
    candidates = []
    for k, v in state_dict.items():
        # weight 텐서이면서 2D이면 (out_features, in_features)
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2 and v.shape[0] >= 2:
            name = k.lower()
            if ("classifier" in name and "weight" in name) or name.endswith("fc.weight") or name.endswith("classifier.1.weight"):
                candidates.append((k, v.shape))
    # 우선순위: classifier.1.weight > classifier.*.weight > fc.weight > 기타
    def score(item):
        name = item[0].lower()
        if name.endswith("classifier.1.weight"):
            return 3
        if ("classifier" in name) and ("weight" in name):
            return 2
        if name.endswith("fc.weight"):
            return 1
        return 0
    if candidates:
        candidates.sort(key=score, reverse=True)
        top_name, shape = candidates[0]
        print(f"[INFO] 선택된 레이어: {top_name} shape={tuple(shape)}")
        return int(shape[0])
    # 백업: 가장 큰 out_features를 가진 2D weight 추정
    fallback = [(k, v.shape) for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
    if fallback:
        fallback.sort(key=lambda x: x[1][0], reverse=True)
        top_name, shape = fallback[0]
        print(f"[WARN] 명확한 분류 레이어 키를 찾지 못해 최대 out_features 후보 사용: {top_name} shape={tuple(shape)}")
        return int(shape[0])
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)
    try:
        sd = load_state_dict_any(path)
        n = infer_num_classes(sd)
        if n is None:
            print("[ERROR] 클래스 수를 추정하지 못했습니다.")
            sys.exit(2)
        print(f"[RESULT] 추정 클래스 수: {n}")
    except Exception as e:
        print(f"[ERROR] 검사 중 예외 발생: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
