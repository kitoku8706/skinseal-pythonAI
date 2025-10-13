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
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/20251006_212412_best_efficientnet.pth')
    MODEL_DEVICE = os.environ.get('MODEL_DEVICE', 'auto')  # auto, cpu, cuda
    
    # 허용된 파일 확장자
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # 로깅 설정
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # 이미지 처리 설정
    IMAGE_SIZE = (224, 224)  # 모델 입력 이미지 크기
    
    @classmethod
    def init_app(cls, app):
        """Flask 앱 초기화"""
        app.config.from_object(cls)
        
        # 업로드 폴더 생성
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)