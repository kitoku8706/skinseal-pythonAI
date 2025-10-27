"""
설정 파일
환경변수 및 애플리케이션 설정 관리
"""

import os
import secrets
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """애플리케이션 설정 클래스"""
    
    # Flask 설정
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'  # 기본값을 False로 변경
    
    # 서버 설정
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # 파일 업로드 설정
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    
    # 모델 설정
    # 현재 저장소의 모델 파일명 예시:
    # - 22클래스: models/best_efficientnet22.pth
    # - 8클래스(예시): models/best_efficientnet_8cls.pth
    # 필요시 환경변수로 덮어쓰세요.
    # 모델 설정: 실제 존재하는 파일로 기본값을 맞춥니다.
    # 기존에 혼선이 있었으므로 기본값을 명시적으로 models/best_efficientnet.pth 로 설정합니다.
    MODEL_PATH_EFFICIENTNET = os.environ.get('MODEL_PATH_EFFICIENTNET', 'models/best_efficientnet.pth')

    MODEL_PATH_ACNE = os.environ.get('MODEL_PATH_ACNE', 'models/best_acne_model.pth')
    MODEL_PATH_SKIN_MODEL = os.environ.get('MODEL_PATH_SKIN_MODEL', 'models/skin_model.pth')
    MODEL_DEVICE = os.environ.get('MODEL_DEVICE', 'auto')  # auto, cpu, cuda
    
    # 허용된 파일 확장자
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # 로깅 설정
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # 이미지 처리 설정
    IMAGE_SIZE = (224, 224)  # 모델 입력 이미지 크기 (학습 시 입력 크기와 일치시켜야 함)
    
    # 모델별 클래스 수
    NUM_CLASSES_EFFICIENTNET = 22
    NUM_CLASSES_SKIN_MODEL = 22
    NUM_CLASSES_ACNE = 2
    
    # Spring Boot Backend API
    SPRING_BOOT_API_URL = "http://localhost:8090/api/diagnosis"
    
    # TTA(Test-Time Augmentation) 옵션
    TTA_ENABLED = os.environ.get('TTA_ENABLED', 'true').lower() == 'true'
    TTA_HFLIP = os.environ.get('TTA_HFLIP', 'true').lower() == 'true'   # 좌우반전 평균
    TTA_VFLIP = os.environ.get('TTA_VFLIP', 'false').lower() == 'true'  # 상하반전 평균
    # 필요 시 회전 등 추가 가능

    @classmethod
    def init_app(cls, app):
        """Flask 앱 초기화"""
        app.config.from_object(cls)
        
        # 업로드 폴더 생성
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(cls.MODEL_PATH_EFFICIENTNET), exist_ok=True)
        os.makedirs(os.path.dirname(cls.MODEL_PATH_SKIN_MODEL), exist_ok=True)
        os.makedirs(os.path.dirname(cls.MODEL_PATH_ACNE), exist_ok=True)