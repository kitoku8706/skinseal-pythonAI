from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import io
import logging
from werkzeug.utils import secure_filename
import requests
from datetime import datetime
from config import Config
from torchvision import transforms as T
import json
from acne_inference import AcneInference

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 모델 및 장치 전역 변수
models = {}
device = None

# Acne 전용 추론기 (지연 생성)
acne_infer = None

# Config.MODEL_DEVICE를 반영한 디바이스 선택 유틸
def select_device_from_config():
    mode = str(getattr(Config, 'MODEL_DEVICE', 'auto')).lower()
    if mode == 'cpu':
        logger.info('디바이스 설정: 강제 CPU')
        return torch.device('cpu')
    if mode == 'cuda':
        if torch.cuda.is_available():
            logger.info('디바이스 설정: 강제 CUDA 사용')
            return torch.device('cuda')
        else:
            logger.warning('CUDA가 가용하지 않습니다. CPU로 폴백합니다.')
            return torch.device('cpu')
    # auto
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"디바이스 설정(auto): {dev}")
    return dev

# 모델별 클래스 레이블 정의(초기값). efficientnet은 부팅 시 JSON/폴더에서 보강 로드
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

def _try_load_labels_for(model_name: str) -> list[str]:
    """레이블 JSON 또는 데이터셋 폴더에서 레이블을 로드."""
    try:
        models_dir = os.path.join(os.getcwd(), 'models')
        if model_name == 'efficientnet':
            # 1) JSON 우선
            json_path = os.path.join(models_dir, 'class_labels_efficientnet.json')
            if os.path.isfile(json_path):
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                labels = data.get('class_names') or data.get('CLASS_LABELS') or data.get('labels')
                if isinstance(labels, list) and len(labels) >= 2:
                    logger.info(f"efficientnet 레이블 JSON 로드: {json_path} (n={len(labels)})")
                    return labels
            # 2) 폴더명 기반 추정(ImageFolder 규칙) - 알파벳 정렬
            # 워크스페이스 상 상대 경로 케어
            repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
            train_dir = os.path.join(repo_root, 'skinseal-model', 'train')
            if os.path.isdir(train_dir):
                class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
                labels = sorted(class_dirs)
                if len(labels) >= 2:
                    logger.warning(f"efficientnet 레이블을 train 폴더에서 추정(n={len(labels)}) → JSON을 생성해두는 것을 권장")
                    return labels
    except Exception as e:
        logger.error(f"레이블 로드 실패({model_name}): {e}")
    return CLASS_LABELS.get(model_name, [])

def _try_load_labels_from_disk():
    """models 디렉토리에서 class_labels_{model}.json을 찾아 동적으로 레이블 로드"""
    try:
        models_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.isdir(models_dir):
            return
        for name in ["efficientnet", "skin_model"]:
            json_path = os.path.join(models_dir, f"class_labels_{name}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # 노트북에서 내보낸 스키마: { class_names: [...], num_classes: N, ... }
                    labels = data.get('class_names') or data.get('CLASS_LABELS')
                    if isinstance(labels, list) and labels:
                        CLASS_LABELS[name] = list(labels)
                        logger.info(f"레이블 로드 완료: {name} -> {len(labels)} classes (from {json_path})")
                except Exception as e:
                    logger.warning(f"레이블 파일 로드 실패({json_path}): {e}")
    except Exception as e:
        logger.warning(f"레이블 자동 로드 중 예외: {e}")

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

def _infer_num_classes_from_ckpt(ckpt_path: str) -> int | None:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict):
            sd = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
        else:
            sd = ckpt
        cleaned = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[len('module.') :]
            if nk.startswith('model.'):
                nk = nk[len('model.') :]
            cleaned[nk] = v
        # 후보 탐색
        candidates = []
        for k, v in cleaned.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] >= 2:
                name = k.lower()
                if ("classifier" in name and "weight" in name) or name.endswith("fc.weight") or name.endswith("classifier.1.weight"):
                    candidates.append((k, v.shape))
        def score(item):
            n = item[0].lower()
            if n.endswith('classifier.1.weight'):
                return 3
            if ('classifier' in n) and ('weight' in n):
                return 2
            if n.endswith('fc.weight'):
                return 1
            return 0
        if candidates:
            candidates.sort(key=score, reverse=True)
            top_name, shape = candidates[0]
            logger.info(f"클래스 수 자동 추정: {top_name} -> out_features={shape[0]}")
            return int(shape[0])
        # 백업: 가장 큰 out_features
        fallback = [(k, v.shape) for k, v in cleaned.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
        if fallback:
            fallback.sort(key=lambda x: x[1][0], reverse=True)
            top_name, shape = fallback[0]
            logger.warning(f"명확한 분류 레이어를 찾지 못해 최대 out_features 후보 사용: {top_name} -> {shape[0]}")
            return int(shape[0])
        return None
    except Exception as e:
        logger.error(f"클래스 수 추정 중 오류: {e}")
        return None

def load_models():
    """PyTorch 모델들을 로드하여 딕셔너리에 저장"""
    global models, device
    try:
        device = select_device_from_config()
        logger.info(f"사용 장치: {device}")

        # 레이블을 디스크에서 우선 로드하여 모델 클래스 수와 정합성 점검 가능
        _try_load_labels_from_disk()

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
            model_path = config["path"]
            # efficientnet의 경우 구파일명 폴백 지원
            if name == "efficientnet" and not os.path.exists(model_path):
                legacy = os.path.join(os.path.dirname(model_path) or '.', 'best_efficientnet.pth')
                if os.path.exists(legacy):
                    logger.warning(f"설정 파일 경로 미존재: {model_path} → 기존 파일명으로 폴백: {legacy}")
                    model_path = legacy
            if os.path.exists(model_path):
                logger.info(f"'{name}' 모델 로드 중... 경로: {model_path}")

                # 클래스 수 자동 추정 및 불일치 처리
                inferred = _infer_num_classes_from_ckpt(model_path)
                effective_num = config["num_classes"]
                if inferred is not None and inferred != effective_num:
                    logger.warning(f"클래스 수 불일치 감지(name={name}): config={effective_num}, ckpt={inferred}. ckpt 기준({inferred})으로 모델 생성.")
                    effective_num = inferred

                model = create_efficientnet_model(num_classes=effective_num)
                if not model:
                    logger.error(f"'{name}' 모델 아키텍처 생성 실패")
                    continue

                try:
                    ckpt = torch.load(model_path, map_location=device)
                    state_dict = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt if isinstance(ckpt, dict) else ckpt
                    cleaned_state = {}
                    for k, v in state_dict.items():
                        nk = k
                        if nk.startswith('module.'):
                            nk = nk[len('module.') :]
                        if nk.startswith('model.'):
                            nk = nk[len('model.') :]
                        cleaned_state[nk] = v

                    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
                    if missing:
                        logger.warning(f"'{name}' 로드 경고 - missing keys: {len(missing)}")
                    if unexpected:
                        logger.warning(f"'{name}' 로드 경고 - unexpected keys: {len(unexpected)}")

                    model = model.to(device)
                    model.eval()
                    models[name] = model
                    logger.info(f"'{name}' 모델이 성공적으로 로드되었습니다. (classes={effective_num})")

                    # 레이블 길이 경고 및 보강 로드
                    if name == 'efficientnet':
                        loaded = _try_load_labels_for('efficientnet')
                        if loaded:
                            CLASS_LABELS['efficientnet'] = loaded
                    labels = CLASS_LABELS.get(name, [])
                    if labels and len(labels) != effective_num:
                        logger.warning(f"'{name}' 레이블 수({len(labels)})와 모델 클래스 수({effective_num})가 일치하지 않습니다.")

                except Exception as e:
                    logger.error(f"'{name}' 모델 로드 중 예외 발생: {e}")
            else:
                logger.warning(f"'{name}' 모델 파일을 찾을 수 없습니다: {model_path}")

        logger.info(f"로드 완료 모델: {list(models.keys())}")

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
        
        # Config.IMAGE_SIZE 적용
        target_size = getattr(Config, 'IMAGE_SIZE', (224, 224))
        transform = transforms.Compose([
            transforms.Resize(target_size),
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

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_ok():
    """서버 상태 확인(별칭 경로)"""
    return jsonify({
        'status': 'healthy',
        'message': '질병진단 AI 서버가 정상 작동 중입니다.',
        'models_loaded': list(models.keys())
    })

# 기존 루트 헬스체크는 유지
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

    # image/file 키 모두 허용
    upload_key = 'image' if 'image' in request.files else ('file' if 'file' in request.files else None)
    if not upload_key:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    file = request.files[upload_key]
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    # 추가: userId 받기
    user_id = request.form.get('userId')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    if file and allowed_file(file.filename):
        try:
            # 이미지 전처리
            image_tensor = preprocess_image(file)
            if image_tensor is None:
                return jsonify({'error': '이미지 처리 중 오류가 발생했습니다.'}), 500
            
            # 전처리된 텐서를 선택된 디바이스로 이동 (Config 반영)
            global device
            if device is None:
                device = select_device_from_config()
            image_tensor = image_tensor.to(device)

            # 선택된 모델로 예측
            selected_model = models[model_name]
            selected_model = selected_model.to(device)

            # --- TTA 적용: 여러 뷰의 평균 확률 ---
            def softmax_logits(t):
                return torch.nn.functional.softmax(t, dim=1)

            views = [image_tensor]
            if getattr(Config, 'TTA_ENABLED', True):
                if getattr(Config, 'TTA_HFLIP', True):
                    views.append(torch.flip(image_tensor, dims=[3]))  # 좌우
                if getattr(Config, 'TTA_VFLIP', False):
                    views.append(torch.flip(image_tensor, dims=[2]))  # 상하

            probs_sum = None
            with torch.no_grad():
                for v in views:
                    out = selected_model(v)
                    prob = softmax_logits(out)
                    probs_sum = prob if probs_sum is None else (probs_sum + prob)
                probabilities = probs_sum / len(views)
                probabilities = probabilities[0]
            
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

            # Spring Boot 백엔드로 보낼 데이터 준비
            diagnosis_data = {
                'userId': user_id,
                'result': results,
                'modelName': model_name
            }

            # Spring Boot 백엔드 API 호출
            try:
                response = requests.post(app.config['SPRING_BOOT_API_URL'], json=diagnosis_data, timeout=5)
                response.raise_for_status()
                app.logger.info(f"Successfully sent diagnosis to Spring Boot: {response.json()}")
            except requests.exceptions.RequestException as e:
                app.logger.error(f"Failed to send diagnosis to Spring Boot: {e}")

            return jsonify(results)

        except Exception as e:
            logger.error(f"진단 중 오류 발생: {str(e)}")
            return jsonify({'error': '진단 중 서버 오류가 발생했습니다.'}), 500
    else:
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

# Acne 전용 진단 엔드포인트
@app.route('/api/diagnosis/acne', methods=['POST'])
def diagnosis_acne():
    """Acne 전용 이진 분류 API"""
    # 파일 키 허용
    upload_key = 'image' if 'image' in request.files else ('file' if 'file' in request.files else None)
    if not upload_key:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400
    file = request.files[upload_key]
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    user_id = request.form.get('userId')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

    try:
        # PIL 이미지 확보
        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        # 추론기 준비(가중치 경로는 내부 기본 또는 Config 사용)
        infer = AcneInference()
        result = infer.predict(img)
        probs = result.get('probs', {})
        # 리스트 형태로 정렬 반환
        pairs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        results = [{ 'class': k, 'probability': f"{v:.2%}" } for k, v in pairs]

        # Spring Boot 저장 호출
        diagnosis_data = {
            'userId': user_id,
            'result': results,
            'modelName': 'acne'
        }
        try:
            response = requests.post(app.config['SPRING_BOOT_API_URL'], json=diagnosis_data)
            response.raise_for_status()
            app.logger.info(f"Successfully sent acne diagnosis to Spring Boot: {response.json()}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Failed to send acne diagnosis to Spring Boot: {e}")

        return jsonify(results)
    except Exception as e:
        logger.error(f"Acne 진단 중 오류: {e}")
        return jsonify({'error': '진단 중 서버 오류가 발생했습니다.'}), 500

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

    # 레이블 자동 로드(존재 시)
    _try_load_labels_from_disk()
    
    # 모델 로드 (실패해도 서버는 계속 시작)
    load_models()
    if not models:
        logger.warning('주의: 어떤 모델도 로드되지 않았습니다. 모델 경로와 클래스 수 설정을 확인하세요.')
    
    # 서버 시작
    logger.info("질병진단 AI 서버를 시작합니다...")
    logger.info(f"로드된 모델: {list(models.keys())}")
    
    # 네트워크 드라이브 문제 해결을 위해 디버그 모드 강제 비활성화
    app.run(host=Config.HOST, port=Config.PORT, debug=False)