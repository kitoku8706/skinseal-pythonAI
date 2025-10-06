# 질병진단 AI 서버

이미지 기반 질병 진단을 위한 Python AI 서버입니다. Flask 웹 프레임워크와 PyTorch 딥러닝 모델을 사용하여 구축되었습니다.

## 🚀 주요 기능

- **이미지 업로드**: 다양한 이미지 형식 지원 (PNG, JPG, JPEG, GIF, BMP)
- **AI 진단**: PyTorch 모델을 사용한 자동 질병 진단
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
  "success": true,
  "filename": "skin_sample.jpg",
  "diagnosis": "정상",
  "confidence": 0.952,
  "class_id": 0
}
```

### 3. 모델 정보 조회

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

테스트 클라이언트를 사용하여 API를 테스트할 수 있습니다:

```bash
python test_client.py
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