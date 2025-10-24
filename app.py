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

# 모델 및 장치 전역 변수
models = {}
device = None

# 모델별 클래스 레이블 정의
CLASS_LABELS = {
    "efficientnet": [
        'Tinea', 'Seborrh Keratoses', 'Rosacea', 'Psoriasis', 'Moles', 
        'Lupus', 'Lichen', 'Eczema'
    ],
    "skin_model": [
        'Acne', 'Actinic Keratosis', 'Benign tumors', 'Bullous', 'Candidiasis',
        'DrugEruption', 'Eczema', 'Infestations Bites', 'Lichen', 'Lupus',
        'Moles', 'Psoriasis', 'Rosacea', 'Seborrh Keratoses', 'SkinCancer',
        'Sun Sunlight Damage', 'Tinea', 'Unknown Normal', 'Vascular Tumors',
        'Vasculitis', 'Vitiligo', 'Warts'
    ]
}

def allowed_file(filename):
    """파일 확장자 검증"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def create_efficientnet_model(num_classes=22):
    """EfficientNet 모델 아키텍처 생성"""
    try:
        # torchvision의 EfficientNet 사용
        import torchvision.models as models
        model = models.efficientnet_b0(weights=None)
        
        # 분류기 레이어 수정
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
        return model
    except Exception as e:
        logger.error(f"모델 아키텍처 생성 오류: {str(e)}")
        return None

def load_models():
    """PyTorch 모델들을 로드하여 딕셔너리에 저장"""
    global models, device
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 장치: {device}")

        model_configs = {
            "efficientnet": {
                "path": Config.MODEL_PATH_EFFICIENTNET,
                "num_classes": Config.NUM_CLASSES_EFFICIENTNET
            },
            "skin_model": {
                "path": Config.MODEL_PATH_SKIN_MODEL,
                "num_classes": Config.NUM_CLASSES_SKIN_MODEL
            }
        }

        for name, config in model_configs.items():
            model_path = os.path.join('models', os.path.basename(config["path"]))
            if os.path.exists(model_path):
                logger.info(f"'{name}' 모델 로드 중... 경로: {model_path}")
                # EfficientNet 아키텍처를 재사용 (필요시 모델별 아키텍처 함수 분리)
                model = create_efficientnet_model(num_classes=config["num_classes"])
                if model:
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()
                    models[name] = model
                    logger.info(f"'{name}' 모델이 성공적으로 로드되었습니다.")
                else:
                    logger.error(f"'{name}' 모델 아키텍처 생성 실패")
            else:
                logger.warning(f"'{name}' 모델 파일을 찾을 수 없습니다: {model_path}")

    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")

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
        'models_loaded': list(models.keys())
    })

@app.route('/api/diagnosis/<model_name>', methods=['POST'])
def diagnosis(model_name):
    """AI 진단 API"""
    if model_name not in models:
        return jsonify({'error': f"'{model_name}' 모델을 찾을 수 없습니다."}), 404

    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    if file and allowed_file(file.filename):
        try:
            # 이미지 전처리
            image_tensor = preprocess_image(file)
            if image_tensor is None:
                return jsonify({'error': '이미지 처리 중 오류가 발생했습니다.'}), 500
            
            image_tensor = image_tensor.to(device)

            # 선택된 모델로 예측
            selected_model = models[model_name]
            with torch.no_grad():
                outputs = selected_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            # 상위 3개 예측 결과 추출
            top_k = 3
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # 클래스 레이블 가져오기
            labels = CLASS_LABELS.get(model_name, [])
            
            results = []
            for i in range(top_k):
                prob = top_probs[i].item()
                class_index = top_indices[i].item()
                class_name = labels[class_index] if class_index < len(labels) else "알 수 없음"
                results.append({'class': class_name, 'probability': f"{prob:.2%}"})

            return jsonify(results)

        except Exception as e:
            logger.error(f"진단 중 오류 발생: {str(e)}")
            return jsonify({'error': '진단 중 서버 오류가 발생했습니다.'}), 500
    else:
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

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
    load_models()
    
    # 서버 시작
    logger.info("질병진단 AI 서버를 시작합니다...")
    logger.info(f"로드된 모델: {list(models.keys())}")
    
    # 네트워크 드라이브 문제 해결을 위해 디버그 모드 강제 비활성화
    app.run(host=Config.HOST, port=Config.PORT, debug=False)