"""
모델 폴더(models) 내 모든 .pth 파일에 대해 최종 분류 레이어(out_features)를 추정하여 출력합니다.
사용법: python models_inspect_all.py
"""
import os
import torch

MODELS_DIR = os.path.join(os.getcwd(), "models")

def load_state_dict_any(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        sd = ckpt
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v
    return cleaned


def infer_num_classes(state_dict: dict):
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2 and v.shape[0] >= 2:
            name = k.lower()
            if ("classifier" in name and "weight" in name) or name.endswith("fc.weight") or name.endswith("classifier.1.weight"):
                candidates.append((k, v.shape))
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
        return top_name, int(shape[0])
    fallback = [(k, v.shape) for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
    if fallback:
        fallback.sort(key=lambda x: x[1][0], reverse=True)
        top_name, shape = fallback[0]
        return top_name, int(shape[0])
    return None, None


def main():
    if not os.path.isdir(MODELS_DIR):
        print(f"[ERROR] models 폴더를 찾을 수 없습니다: {MODELS_DIR}")
        return
    files = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    if not files:
        print("[WARN] .pth 파일이 없습니다.")
        return
    for f in files:
        print('\n===', f, '===')
        try:
            sd = load_state_dict_any(f)
            name, n = infer_num_classes(sd)
            if n is None:
                print('[RESULT] 추정 실패')
            else:
                print(f'[RESULT] 레이어: {name}  -> 추정 클래스 수: {n}')
        except Exception as e:
            print(f'[ERROR] 검사 실패: {e}')

if __name__ == '__main__':
    main()
