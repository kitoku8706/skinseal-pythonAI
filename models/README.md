# 모델 디렉토리
이 디렉토리에 PyTorch 모델 파일을 저장하세요.

## 필요한 모델 파일
- `20251006_212412_best_efficientnet.pth`: 질병진단을 위한 EfficientNet 모델

## 모델 배치 방법
1. 훈련된 PyTorch 모델 파일(.pth)을 이 디렉토리에 복사
2. `config.py`에서 MODEL_PATH 설정 확인
3. 서버 재시작 후 모델이 자동으로 로드됨

## 주의사항
- 모델 파일은 용량이 클 수 있으므로 Git에 커밋하지 마세요
- `.gitignore`에 `*.pth` 패턴이 추가되어 있습니다