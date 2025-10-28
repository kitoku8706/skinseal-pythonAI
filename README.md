# 질병진단 AI 서버

이미지 기반 질병 진단을 위한 Python AI 서버입니다. Flask 웹 프레임워크와 PyTorch 딥러닝 모델을 사용하여 구축되었습니다.

## 🚀 주요 기능

- **이미지 업로드**: 다양한 이미지 형식 지원 (PNG, JPG, JPEG, GIF, BMP)
- **AI 진단**: PyTorch 모델을 사용한 자동 질병 진단
- **GradCAM 시각화**: 
  - 원본 이미지, 히트맵, 오버레이 이미지 제공
  - AI 판단 근거의 시각적 설명
  - Base64 인코딩으로 웹 친화적 전송
- **REST API**: JSON 형태의 응답을 제공하는 RESTful API
- **실시간 처리**: 이미지 업로드 즉시 진단 결과 제공
- **오류 처리**: 포괄적인 예외 처리 및 사용자 친화적 오류 메시지

## 📁 프로젝트 구조

```
skinseal-pythonAI/
├── app.py                 # 메인 Flask 애플리케이션 (완전 기능)
├── app_simple.py          # 간단한 Flask 애플리케이션 (테스트용)
├── config.py              # 설정 관리
├── utils.py               # 유틸리티 함수들
├── test_client.py         # API 테스트 클라이언트
├── test_gradcam_visual.py # GradCAM 시각화 테스트 클라이언트
├── requirements.txt       # Python 의존성 패키지
├── .env.example           # 환경 변수 예시 파일
├── .gitignore            # Git 제외 파일 목록
├── models/               # PyTorch 모델 파일 저장소
│   └── README.md
├── uploads/              # 업로드된 이미지 임시 저장소
│   └── README.md
└── .github/
    └── copilot-instructions.md
```

## 🔧 설치 및 설정

### 1. 필수 요구사항

- Python 3.8 이상
- 필요한 Python 패키지들 (requirements.txt 참조)

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정 (선택사항)

```bash
cp .env.example .env
# .env 파일을 편집하여 설정 값들을 조정하세요
```

### 4. 모델 파일 배치

PyTorch 모델 파일(`20251006_212412_best_efficientnet.pth`)을 `models/` 디렉토리에 배치하세요.

## 🚀 서버 실행

### 방법 1: 완전한 AI 서버 실행

```bash
python app.py
```

### 방법 2: 간단한 테스트 서버 실행

```bash
python app_simple.py
```

서버가 시작되면 다음 주소에서 접근할 수 있습니다:
- 로컬: http://localhost:5000
- 네트워크: http://0.0.0.0:5000

## 📡 API 엔드포인트

### 1. 서버 상태 확인

```http
GET /
```

**응답 예시:**
```json
{
  "status": "healthy",
  "message": "질병진단 AI 서버가 정상 작동 중입니다.",
  "model_loaded": true
}
```

### 2. 질병 진단

```http
POST /diagnose
Content-Type: multipart/form-data
```

**요청 파라미터:**
- `image`: 진단할 이미지 파일

**응답 예시:**
```json
{
  "results": [
    {
      "class": "정상",
      "probability": "95.20%"
    },
    {
      "class": "피부염",
      "probability": "3.15%"
    }
  ]
}
```

### 3. GradCAM과 함께 진단 (시각화 포함)

```http
POST /api/diagnosis/{model_name}
```

**요청 파라미터:**
- `file`: 진단할 이미지 파일
- `userId`: 사용자 ID
- `gradcam`: "true"로 설정하면 GradCAM 결과 포함
- `classIndex`: (선택사항) 특정 클래스에 대한 GradCAM 생성

**응답 예시:**
```json
{
  "results": [
    {
      "class": "피부염",
      "probability": "87.30%"
    }
  ],
  "gradcam": {
    "targetIndex": 1,
    "score": 0.8730,
    "original_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "heatmap_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "overlay_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  }
}
```

### 4. 전용 GradCAM API

```http
POST /api/diagnosis/{model_name}/gradcam
```

**요청 파라미터:**
- `file`: 진단할 이미지 파일
- `userId`: 사용자 ID
- `classIndex`: (선택사항) 특정 클래스에 대한 GradCAM 생성

**응답**: 위와 동일한 형식으로 항상 GradCAM 결과 포함

## 📊 GradCAM 시각화 이해하기

GradCAM(Gradient-weighted Class Activation Mapping)은 AI 모델이 이미지의 어느 부분을 보고 판단을 내렸는지 시각적으로 보여주는 기술입니다.

### 응답 이미지 설명

1. **original_base64**: 업로드한 원본 이미지
   - 분석 대상이 된 원래 이미지
   
2. **heatmap_base64**: 열화상 맵 (히트맵)
   - 빨간색 영역: AI가 중요하게 본 부분 (높은 관심도)
   - 파란색 영역: AI가 덜 중요하게 본 부분 (낮은 관심도)
   
3. **overlay_base64**: 오버레이 이미지
   - 원본 이미지 위에 히트맵을 투명하게 겹친 결과
   - 실제 병변 위치와 AI 판단 영역 비교 가능

### 클라이언트 구현 예제

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# GradCAM API 호출
response = requests.post('http://localhost:5000/api/diagnosis/efficientnet/gradcam', 
                        files={'file': ('image.jpg', open('image.jpg', 'rb'))},
                        data={'userId': 'user123'})

if response.status_code == 200:
    result = response.json()
    gradcam = result['gradcam']
    
    # Base64 이미지를 파일로 저장
    for img_type in ['original', 'heatmap', 'overlay']:
        b64_data = gradcam[f'{img_type}_base64']
        img_data = base64.b64decode(b64_data)
        
        with open(f'{img_type}.png', 'wb') as f:
            f.write(img_data)
```

### 5. 모델 정보 조회

```http
GET /model/info
```

**응답 예시:**
```json
{
  "model_loaded": true,
  "device": "cpu",
  "model_path": "models/20251006_212412_best_efficientnet.pth"
}
```

## 🧪 테스트

### 기본 API 테스트
```bash
python test_client.py
```

### GradCAM 시각화 테스트
```bash
python test_gradcam_visual.py
```

이 테스트는 다음과 같은 결과를 생성합니다:
- `gradcam_results/diagnosis_original.png`: 원본 이미지
- `gradcam_results/diagnosis_heatmap.png`: GradCAM 히트맵
- `gradcam_results/diagnosis_overlay.png`: 원본 + 히트맵 오버레이

### Acne 전용 모델 테스트
```bash
python test_acne_client.py
```

## 🔧 설정 옵션

`config.py` 파일에서 다음 설정들을 조정할 수 있습니다:

- `MAX_CONTENT_LENGTH`: 업로드 파일 최대 크기 (기본: 16MB)
- `MODEL_PATH`: PyTorch 모델 파일 경로
- `ALLOWED_EXTENSIONS`: 허용된 이미지 파일 확장자
- `IMAGE_SIZE`: 모델 입력 이미지 크기 (기본: 224x224)

## 📋 지원되는 진단 클래스

현재 모델은 다음과 같은 피부 질환을 분류합니다:

- **정상** (Class ID: 0)
- **피부염** (Class ID: 1)  
- **습진** (Class ID: 2)
- **건선** (Class ID: 3)
- **기타 피부질환** (Class ID: 4)

## ⚠️ 주의사항

1. **의료용 주의**: 이 시스템은 의료 전문가의 진단을 대체하지 않습니다.
2. **개발 서버**: Flask 내장 서버는 개발용입니다. 프로덕션 환경에서는 Gunicorn, uWSGI 등을 사용하세요.
3. **모델 파일**: 용량이 큰 모델 파일은 Git에 커밋하지 마세요.
4. **보안**: 프로덕션 환경에서는 적절한 인증 및 보안 설정을 추가하세요.

## 🔧 개발 환경

VS Code에서 개발할 때:

1. Python 확장 프로그램 설치
2. Pylance 확장 프로그램 설치  
3. 터미널에서 `python app.py` 실행 또는 VS Code 작업 사용

## 📝 라이선스

이 프로젝트는 의료 AI 연구 및 개발 목적으로 제작되었습니다.

## 🤝 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

---

**개발자**: SkinSeal AI Team  
**버전**: 1.0.0  
**마지막 업데이트**: 2025년 10월 6일