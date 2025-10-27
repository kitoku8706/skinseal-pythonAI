# SkinSeal Python AI — 실행 및 진단 가이드

간단 문서: 서버 재시작, 로그 확인, 모델 레이블 파일 위치 및 문제 해결 요약

## 생성된 파일(요약)
- run_server.ps1 — 서버(Flask) 재시작 및 로그 수집 도우미 스크립트
- models/class_labels_efficientnet.json — EfficientNet용 클래스 라벨(22개)
- models/class_labels_skin_model.json — Skin Model용 클래스 라벨(22개)
- logs/app.log — 서버 실행 로그(실행 시 생성/갱신)

## 모델 파일(현재 위치)
- c:\2st-pro\skinseal-pythonAI\models\best_efficientnet.pth  (추정 클래스 수: 8)
- c:\2st-pro\skinseal-pythonAI\models\skin_model.pth         (추정 클래스 수: 22)
- 레이블 JSON: models\class_labels_efficientnet.json (22 labels)
- 레이블 JSON: models\class_labels_skin_model.json (22 labels)

> 주의: 현재 `best_efficientnet.pth`는 체크포인트 내부 정보 기준으로 8클래스입니다. 레이블 JSON은 22개로 되어 있어 불일치 경고가 발생합니다. 파일명을 분리(예: `best_efficientnet_8cls.pth`, `best_efficientnet_22cls.pth`)하거나 Config의 `MODEL_PATH_EFFICIENTNET`를 원하는 파일로 지정하세요.

## 서버 재시작(권장)
PowerShell에서 작업 디렉터리로 이동 후 run_server.ps1을 사용하세요.

- 포그라운드(콘솔 출력 + 로그 저장):
  powershell -ExecutionPolicy Bypass -File "C:\2st-pro\run_server.ps1"

- 포그라운드(특정 모델 경로 지정):
  powershell -ExecutionPolicy Bypass -File "C:\2st-pro\run_server.ps1" -EfficientPath 'skinseal-pythonAI\models\best_efficientnet.pth' -SkinPath 'skinseal-pythonAI\models\skin_model.pth' -Port 5000

- 백그라운드(로그만 기록):
  powershell -ExecutionPolicy Bypass -File "C:\2st-pro\run_server.ps1" -Background

서버 중지: 실행 터미널에서 Ctrl+C 또는 작업 관리자/Stop-Process.

## 로그 확인
- 로그 파일: `c:\2st-pro\skinseal-pythonAI\logs\app.log`
- 실시간 보기(다른 PowerShell 창):
  Get-Content 'C:\2st-pro\skinseal-pythonAI\logs\app.log' -Wait -Tail 200

로그에서 확인할 키워드:
- `state_dict keys(sample 30)` — 체크포인트 키 샘플
- `candidate classifier keys` — 분류기 관련 키 후보
- `missing keys` / `unexpected keys` — load_state_dict 결과
- `Model config info` — 설정 경로 및 체크포인트로 추정한 클래스 수

## 헬스 체크
- 직접 Flask:
  Invoke-RestMethod -Uri http://127.0.0.1:5000/health
- Vite 프록시 사용시:
  Invoke-RestMethod -Uri http://127.0.0.1:5173/api/health

응답의 `models_loaded` 항목에 로드된 모델 리스트가 표시됩니다.

## 문제 발견 시 권장 조치
1. `app.log`를 열어 `missing/unexpected keys`와 `Model config info` 확인.
2. EfficientNet의 경우 체크포인트 내부 추정 클래스 수(`inferred_classes`)와 레이블 JSON 길이가 일치하도록:
   - 체크포인트 이름을 명확히(예: `best_efficientnet_8cls.pth`, `best_efficientnet_22cls.pth`)하고 `config.py` 또는 환경변수로 지정.
3. state_dict 키가 `module.` 또는 `model.` 프리픽스가 있으면 `app.py`가 이미 정리(clean)하지만, 필요한 경우 `cleaned_state` 매핑 보강.
4. 그래도 문제면 `logs/app.log`의 관련 블록(모델 로드 직후)을 복사해 공유하면 구체적 패치 제안.

## 빠른 체크 명령
- models 폴더 파일 목록
  Get-ChildItem -Path .\skinseal-pythonAI\models -File | Select-Object Name,Length,LastWriteTime

- 체크포인트에서 추정 클래스 수(이미 제공된 스크립트 사용):
  & python .\skinseal-pythonAI\models_inspect_all.py

---
문서 생성 완료. 더 추가할 항목(예: 파일명 변경 권장 표, 마이그레이션 스크립트 등)이 있으면 알려주세요.
