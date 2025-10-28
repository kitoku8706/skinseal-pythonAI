"""
GradCAM 시각화 테스트 클라이언트
원본 이미지, 히트맵, 오버레이 이미지를 저장하여 시각적으로 확인
"""

import requests
import json
import os
import base64
from PIL import Image
from io import BytesIO

# 서버 설정
BASE_URL = "http://localhost:5000/api"
DIAGNOSIS_URL = f"{BASE_URL}/diagnosis/efficientnet"
GRADCAM_URL = f"{BASE_URL}/diagnosis/efficientnet/gradcam"

# 테스트 이미지 경로
TEST_IMAGE_PATH = "c:/2st-pro/skinseal-pythonAI/uploads/test_sample.jpg"
USER_ID = "test_user"

# 결과 저장 디렉토리
RESULT_DIR = "c:/2st-pro/skinseal-pythonAI/gradcam_results"

def create_test_image_if_not_exists():
    """테스트 이미지가 없으면 생성"""
    if not os.path.exists(TEST_IMAGE_PATH):
        os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)
        print(f"테스트 이미지를 생성합니다: {TEST_IMAGE_PATH}")
        # 간단한 그라데이션 이미지 생성
        img = Image.new('RGB', (224, 224))
        pixels = img.load()
        for i in range(224):
            for j in range(224):
                pixels[i, j] = (i, j, (i+j) % 256)
        img.save(TEST_IMAGE_PATH, format='JPEG', quality=90)
        print("테스트 이미지가 생성되었습니다.")

def save_base64_image(base64_str, filename):
    """Base64 문자열을 이미지 파일로 저장"""
    try:
        # Base64 디코딩
        image_data = base64.b64decode(base64_str)
        # PIL 이미지로 변환
        image = Image.open(BytesIO(image_data))
        # 파일로 저장
        image.save(filename)
        print(f"이미지 저장됨: {filename}")
        return True
    except Exception as e:
        print(f"이미지 저장 실패 {filename}: {e}")
        return False

def test_gradcam_with_visualization():
    """GradCAM 테스트 및 이미지 저장"""
    print("=== GradCAM 시각화 테스트 ===")
    
    # 테스트 이미지 확인/생성
    create_test_image_if_not_exists()
    
    # 결과 저장 디렉토리 생성
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 1. 일반 diagnosis API에서 GradCAM 옵션 사용
    print("\n1. 일반 diagnosis API + GradCAM 옵션 테스트")
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {
                'userId': USER_ID,
                'gradcam': 'true'
            }
            response = requests.post(DIAGNOSIS_URL, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("진단 결과:")
            for r in result.get('results', []):
                print(f"  - {r['class']}: {r['probability']}")
            
            if 'gradcam' in result:
                gradcam = result['gradcam']
                print(f"\nGradCAM 정보:")
                print(f"  - Target Index: {gradcam.get('targetIndex')}")
                print(f"  - Score: {gradcam.get('score', 0):.4f}")
                
                # 이미지들 저장
                timestamp = "diagnosis"
                if gradcam.get('original_base64'):
                    save_base64_image(gradcam['original_base64'], 
                                    os.path.join(RESULT_DIR, f"{timestamp}_original.png"))
                if gradcam.get('heatmap_base64'):
                    save_base64_image(gradcam['heatmap_base64'], 
                                    os.path.join(RESULT_DIR, f"{timestamp}_heatmap.png"))
                if gradcam.get('overlay_base64'):
                    save_base64_image(gradcam['overlay_base64'], 
                                    os.path.join(RESULT_DIR, f"{timestamp}_overlay.png"))
            else:
                print("GradCAM 결과가 없습니다.")
        else:
            print(f"API 호출 실패: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"테스트 실패: {e}")
      # 2. 전용 GradCAM API 테스트
    print("\n2. 전용 GradCAM API 테스트")
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {'userId': USER_ID}
            response = requests.post(GRADCAM_URL, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("진단 결과:")
            for r in result.get('results', []):
                print(f"  - {r['class']}: {r['probability']}")
            
            print(f"\n전체 응답 구조:")
            print(f"- 키들: {list(result.keys())}")
            
            if 'gradcam' in result:
                gradcam = result['gradcam']
                print(f"\nGradCAM 정보:")
                print(f"  - Target Index: {gradcam.get('targetIndex')}")
                print(f"  - Score: {gradcam.get('score', 0):.4f}")
                print(f"  - GradCAM 키들: {list(gradcam.keys())}")
                print(f"  - original_base64 존재: {'original_base64' in gradcam}")
                print(f"  - heatmap_base64 존재: {'heatmap_base64' in gradcam}")
                print(f"  - overlay_base64 존재: {'overlay_base64' in gradcam}")
                
                # 이미지들 저장
                timestamp = "gradcam_api"
                if gradcam.get('original_base64'):
                    save_base64_image(gradcam['original_base64'], 
                                    os.path.join(RESULT_DIR, f"{timestamp}_original.png"))
                if gradcam.get('heatmap_base64'):
                    save_base64_image(gradcam['heatmap_base64'], 
                                    os.path.join(RESULT_DIR, f"{timestamp}_heatmap.png"))
                if gradcam.get('overlay_base64'):
                    save_base64_image(gradcam['overlay_base64'], 
                                    os.path.join(RESULT_DIR, f"{timestamp}_overlay.png"))
            else:
                print("GradCAM 결과가 없습니다.")
        else:
            print(f"API 호출 실패: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"테스트 실패: {e}")
    
    print(f"\n결과 이미지들이 저장되었습니다: {RESULT_DIR}")
    print("다음 파일들을 확인하세요:")
    if os.path.exists(RESULT_DIR):
        for file in os.listdir(RESULT_DIR):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - {file}")

def test_server_connection():
    """서버 연결 테스트"""
    print("서버 연결 확인 중...")
    try:
        health_url = f"{BASE_URL}/health"
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print("✓ 서버가 정상적으로 응답하고 있습니다.")
            return True
        else:
            print(f"✗ 서버 응답 오류: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ 서버 연결 실패: {e}")
        print("서버가 실행 중인지 확인하세요: python app.py")
        return False

if __name__ == "__main__":
    if test_server_connection():
        test_gradcam_with_visualization()
    else:
        print("\n서버를 먼저 실행하세요:")
        print("  python app.py")
