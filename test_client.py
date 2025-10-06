"""
테스트 클라이언트
질병진단 AI 서버 API 테스트용 스크립트
"""

import requests
import json
import os
from pathlib import Path

class DiagnosisClient:
    """진단 API 클라이언트"""
    
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
    
    def health_check(self):
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def diagnose_image(self, image_path):
        """이미지 진단"""
        try:
            if not os.path.exists(image_path):
                return {'error': '이미지 파일을 찾을 수 없습니다.'}
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{self.base_url}/diagnose", files=files)
                return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def model_info(self):
        """모델 정보 조회"""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            return response.json()
        except Exception as e:
            return {'error': str(e)}

def main():
    """테스트 실행"""
    client = DiagnosisClient()
    
    print("=== 질병진단 AI 서버 테스트 ===")
    
    # 1. 서버 상태 확인
    print("\n1. 서버 상태 확인")
    health = client.health_check()
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    # 2. 모델 정보 확인
    print("\n2. 모델 정보 확인")
    model_info = client.model_info()
    print(json.dumps(model_info, indent=2, ensure_ascii=False))
    
    # 3. 샘플 이미지 테스트 (이미지가 있는 경우)
    sample_images = ['test_image.jpg', 'sample.png', 'test.jpeg']
    
    for image_name in sample_images:
        if os.path.exists(image_name):
            print(f"\n3. 이미지 진단 테스트: {image_name}")
            result = client.diagnose_image(image_name)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            break
    else:
        print("\n3. 테스트 이미지가 없습니다.")
        print("테스트용 이미지를 추가하여 진단 기능을 테스트하세요.")

if __name__ == '__main__':
    main()