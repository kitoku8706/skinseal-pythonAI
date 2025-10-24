"""
Acne 전용 판독(추론) 모듈
- EfficientNet-B0 기반 이진 분류(Non-Acne, Acne) 기본 구현
- Config 설정(IMAGE_SIZE, MODEL_DEVICE, TTA_*) 반영
- 다양한 체크포인트 포맷(state_dict, model_state_dict, module.*, model.*) 유연 로딩

예시 사용법
-------------
from acne_inference import AcneInference
from PIL import Image

inf = AcneInference(weights_path="models/best_acne_model.pth")
img = Image.open("sample.jpg").convert("RGB")
result = inf.predict(img)
print(result)
# {'top1': {'class': 'Acne', 'probability': 0.92}, 'probs': {'Non-Acne': 0.08, 'Acne': 0.92}}
"""
from __future__ import annotations

import io
import logging
from typing import Dict, Optional, Union

import torch
import torchvision.transforms as T
from PIL import Image

try:
    from config import Config
except Exception:  # 독립 실행 시 기본값
    class Config:  # type: ignore
        IMAGE_SIZE = (224, 224)
        MODEL_DEVICE = "auto"
        TTA_ENABLED = True
        TTA_HFLIP = True
        TTA_VFLIP = False

logger = logging.getLogger(__name__)

DEFAULT_LABELS = ["Non-Acne", "Acne"]


def _select_device(mode: str = "auto") -> torch.device:
    mode = str(mode or "auto").lower()
    if mode == "cpu":
        logger.info("AcneInference: 강제 CPU 사용")
        return torch.device("cpu")
    if mode == "cuda":
        if torch.cuda.is_available():
            logger.info("AcneInference: 강제 CUDA 사용")
            return torch.device("cuda")
        logger.warning("AcneInference: CUDA 미가용, CPU로 폴백")
        return torch.device("cpu")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"AcneInference: auto 디바이스 선택 → {dev}")
    return dev


def _build_efficientnet_b0(num_classes: int = 2) -> torch.nn.Module:
    import torchvision.models as models
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features, num_classes),
    )
    return m


def _clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v
    return cleaned


class AcneInference:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        image_size: Optional[tuple[int, int]] = None,
        labels: Optional[list[str]] = None,
        tta: Optional[dict] = None,
    ) -> None:
        self.labels = labels or DEFAULT_LABELS
        self.num_classes = len(self.labels)
        self.image_size = image_size or getattr(Config, "IMAGE_SIZE", (224, 224))
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = _select_device(getattr(Config, "MODEL_DEVICE", "auto") if device is None else device)
        self.tta_conf = {
            "enabled": getattr(Config, "TTA_ENABLED", True),
            "hflip": getattr(Config, "TTA_HFLIP", True),
            "vflip": getattr(Config, "TTA_VFLIP", False),
        }
        if isinstance(tta, dict):
            self.tta_conf.update(tta)

        # 모델 구성 및 가중치 로딩
        self.model = _build_efficientnet_b0(self.num_classes).to(self.device)
        self.model.eval()

        wp = (
            weights_path
            or getattr(Config, "MODEL_PATH_ACNE", None)
            or "models/best_acne_model.pth"
        )
        self._try_load_weights(wp)

        # 전처리 파이프라인
        self.transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _try_load_weights(self, path: Optional[str]) -> None:
        if not path:
            logger.warning("AcneInference: 가중치 경로 미지정, 랜덤 초기화로 진행")
            return
        try:
            ckpt = torch.load(path, map_location=self.device)
            if isinstance(ckpt, dict):
                sd = (
                    ckpt.get("state_dict")
                    or ckpt.get("model_state_dict")
                    or ckpt
                )
            else:
                sd = ckpt
            sd = _clean_state_dict(sd)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"AcneInference: missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"AcneInference: unexpected keys: {len(unexpected)}")
            logger.info(f"AcneInference: 가중치 로딩 완료 → {path}")
        except Exception as e:
            logger.error(f"AcneInference: 가중치 로딩 실패: {e}")

    def _ensure_pil(self, img: Union[Image.Image, bytes, bytearray, io.BytesIO, str]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, (bytes, bytearray)):
            return Image.open(io.BytesIO(img)).convert("RGB")
        if isinstance(img, io.BytesIO):
            return Image.open(img).convert("RGB")
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        raise TypeError("지원하지 않는 이미지 입력 형식입니다.")

    def _tta_views(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        views = [tensor]
        if self.tta_conf.get("enabled", True):
            if self.tta_conf.get("hflip", True):
                views.append(torch.flip(tensor, dims=[3]))
            if self.tta_conf.get("vflip", False):
                views.append(torch.flip(tensor, dims=[2]))
        return views

    @torch.no_grad()
    def predict(self, img: Union[Image.Image, bytes, bytearray, io.BytesIO, str]) -> Dict:
        pil = self._ensure_pil(img).convert("RGB")
        x = self.transform(pil).unsqueeze(0).to(self.device)

        # TTA 평균
        views = self._tta_views(x)
        prob_sum = None
        for v in views:
            logits = self.model(v)
            prob = torch.softmax(logits, dim=1)
            prob_sum = prob if prob_sum is None else (prob_sum + prob)
        probs = (prob_sum / len(views))[0].detach().cpu().tolist()

        # 결과 조립
        probs_map = {self.labels[i]: float(p) for i, p in enumerate(probs)}
        top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        result = {
            "top1": {
                "class": self.labels[top_idx],
                "probability": float(probs[top_idx]),
            },
            "probs": probs_map,
        }
        return result


__all__ = ["AcneInference", "DEFAULT_LABELS"]
