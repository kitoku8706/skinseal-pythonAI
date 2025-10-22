from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import io
import logging
from werkzeug.utils import secure_filename
from config import Config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 허용된 파일 확장자는 Config에서 가져옴

# 모델 글로벌 변수
model = None
device = None

# 클래스 이름 리스트 (훈련 데이터 디렉토리 순서에 맞춰야 함)
CLASS_NAMES = [
    'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis', 
    'DrugEruption', 'Eczema', 'Infestations_Bites', 'Lichen', 'Lupus', 
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 
    'Sun_Sunlight_Damage', 'Tinea', 'Unknown_Normal', 'Vascular_Tumors', 
    'Vasculitis', 'Vitiligo', 'Warts'
]

def allowed_file(filename):
    """파일 확장자 검증"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def create_efficientnet_model(num_classes=22):
    """EfficientNet 모델 아키텍처 생성"""
    try:
        # torchvision의 EfficientNet 사용 (사전 학습된 가중치는 사용 안 함)
        import torchvision.models as models
        model = models.efficientnet_b0(weights=None)
        
        # 모델의 분류기 부분을 프로젝트에 맞게 수정
        # EfficientNet-B0의 분류기는 1280개의 입력 피처를 가짐
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(num_ftrs, num_classes)
        )
        
        return model
    except Exception as e:
        logger.error(f"모델 아키텍처 생성 오류: {str(e)}")
        return None

def load_model():
    """PyTorch 모델 로드"""
    global model, device
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join('models', 'best_efficientnet.pth')

        if os.path.exists(model_path):
            logger.info(f"모델 파일 경로: {os.path.abspath(model_path)}")
            
            # 1. 모델 아키텍처부터 생성
            model = create_efficientnet_model(num_classes=22)
            if model is None:
                logger.error("모델 아키텍처 생성 실패")
                return

            # 2. 저장된 가중치(state_dict)를 모델에 로드
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model = model.to(device)
                model.eval() # 추론 모드로 설정
                logger.info("모델이 성공적으로 로드되었습니다.")
            except Exception as e:
                logger.error(f"저장된 가중치(state_dict) 로드 중 오류 발생: {str(e)}")
                # 오류 발생 시 모델을 None으로 설정하여 예측 시도를 막음
                model = None
        else:
            logger.warning(f"모델 파일을 찾을 수 없습니다: {os.path.abspath(model_path)}")
            model = None
    except Exception as e:
        logger.error(f"모델 로드 과정에서 예기치 않은 오류 발생: {str(e)}")
        model = None

def preprocess_image(image_file):
    """이미지 전처리"""
    try:
        # PIL 이미지로 변환
        image = Image.open(io.BytesIO(image_file.read()))
        
        # RGB로 변환 (RGBA 등 다른 채널 처리)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 이미지 전처리 파이프라인
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 텐서로 변환 및 배치 차원 추가
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        logger.error(f"이미지 전처리 중 오류: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'message': '질병진단 AI 서버가 정상 작동 중입니다.',
        'model_loaded': model is not None
    })

@app.route('/diagnose', methods=['POST'])
def diagnose():
    """이미지 기반 질병 진단 엔드포인트"""
    try:
        # 파일 업로드 확인
        if 'image' not in request.files:
            return jsonify({'error': '이미지 파일이 제공되지 않았습니다.'}), 400
        
        file = request.files['image']
        
        # 빈 파일 체크
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        # 파일 형식 확인
        if not allowed_file(file.filename):
            return jsonify({'error': '지원되지 않는 파일 형식입니다.'}), 400
        
        # 이미지 전처리
        image_tensor = preprocess_image(file)
        if image_tensor is None:
            return jsonify({'error': '이미지 처리 중 오류가 발생했습니다.'}), 400
        
        # 모델 예측
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction_idx = predicted.item()

        # 예측된 인덱스를 클래스 이름으로 변환
        if 0 <= prediction_idx < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[prediction_idx]
        else:
            predicted_class_name = "Unknown"
            logger.warning(f"예측된 인덱스 {prediction_idx}가 클래스 이름 범위 밖에 있습니다.")

        # 결과 반환
        logger.info(f"진단 결과: {predicted_class_name}")
        return jsonify({'diagnosis': predicted_class_name})

    except Exception as e:
        logger.error(f"진단 중 오류 발생: {str(e)}")
        return jsonify({
            'success': False,
            'error': '서버 내부 오류가 발생했습니다.'
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """모델 정보 반환"""
    return jsonify({
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown',
        'model_path': 'models/20251006_212412_best_efficientnet.pth'
    })

@app.errorhandler(413)
def too_large(e):
    """파일 크기 초과 오류 처리"""
    return jsonify({'error': '파일 크기가 너무 큽니다. 16MB 이하의 파일을 업로드하세요.'}), 413

@app.errorhandler(404)
def not_found(e):
    """404 오류 처리"""
    return jsonify({'error': '요청한 엔드포인트를 찾을 수 없습니다.'}), 404

@app.errorhandler(500)
def server_error(e):
    """500 오류 처리"""
    return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    # 설정 적용
    Config.init_app(app)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    
    # 모델 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    # 모델 로드 (실패해도 서버는 계속 시작)
    load_model()
    
    # 서버 시작
    logger.info("질병진단 AI 서버를 시작합니다...")
    logger.info(f"모델 로드 상태: {'성공' if model is not None else '실패 (모델 없이 실행)'}")
    
    # 네트워크 드라이브 문제 해결을 위해 디버그 모드 강제 비활성화
    app.run(host=Config.HOST, port=Config.PORT, debug=False)