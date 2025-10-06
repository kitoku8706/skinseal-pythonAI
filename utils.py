"""
유틸리티 함수들을 모아놓은 모듈
이미지 처리, 파일 검증 등의 공통 기능 제공
"""

import os
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """이미지 처리를 위한 클래스"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def process_image(self, image_file):
        """이미지 파일을 텐서로 변환"""
        try:
            # 파일 포인터를 처음으로 되돌림
            image_file.seek(0)
            
            # PIL 이미지로 변환
            image = Image.open(image_file)
            
            # RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 텐서로 변환
            image_tensor = self.transform(image).unsqueeze(0)
            
            return image_tensor
        except Exception as e:
            logger.error(f"이미지 처리 오류: {str(e)}")
            return None

class ModelManager:
    """모델 관리를 위한 클래스"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.class_labels = {
            0: '정상',
            1: '피부염',
            2: '습진', 
            3: '건선',
            4: '기타 피부질환'
        }
    
    def load_model(self, model_path):
        """모델 로드"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                logger.info(f"모델 로드 성공: {model_path}")
                return True
            else:
                logger.warning(f"모델 파일 없음: {model_path}")
                return False
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            return False
    
    def predict(self, image_tensor):
        """예측 수행"""
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                
                # 확률 계산
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                return {
                    'class_id': predicted_class,
                    'diagnosis': self.class_labels.get(predicted_class, '알 수 없음'),
                    'confidence': round(confidence_score, 3)
                }
        except Exception as e:
            logger.error(f"예측 오류: {str(e)}")
            return None

def validate_file(file, allowed_extensions):
    """파일 검증"""
    if not file or not file.filename:
        return False, "파일이 선택되지 않았습니다."
    
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return False, "지원되지 않는 파일 형식입니다."
    
    return True, "파일이 유효합니다."