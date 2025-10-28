# 질병진단 AI 서버 프로젝트

이 프로젝트는 이미지 기반 질병 진단을 위한 완전한 Python AI 서버입니다.

## 프로젝트 완료 상태
✅ Flask 웹 서버 구성 완료
✅ PyTorch 딥러닝 모델 통합 완료  
✅ 이미지 처리 및 분석 기능 완료
✅ REST API 엔드포인트 구현 완료
✅ **GradCAM 시각화 기능 완료** (NEW!)
✅ 원본/히트맵/오버레이 이미지 Base64 인코딩 완료
✅ VS Code 개발 환경 설정 완료
✅ 테스트 클라이언트 구현 완료
✅ 시각화 테스트 클라이언트 구현 완료

## 개발 가이드라인
- Python 3.8+ 사용
- Flask 웹 프레임워크
- PyTorch 모델 통합
- 의료 이미지 처리 파이프라인
- **GradCAM 시각화 통합** (NEW!)
- Base64 이미지 인코딩
- JSON API 응답 형식
- 포괄적인 오류 처리
- 모듈화된 코드 구조

## 실행 방법
1. `python app.py` - 완전한 AI 서버 (모델 포함)
2. `python app_simple.py` - 간단한 테스트 서버
3. `python test_client.py` - API 테스트 실행

## 주요 파일
- `app.py`: 메인 Flask AI 서버
- `utils.py`: 이미지 처리 및 모델 관리 유틸리티
- `config.py`: 환경 설정 관리
- `test_client.py`: API 테스트 클라이언트