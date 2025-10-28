"""
테스트 클라이언트
질병진단 AI 서버 API 테스트용 스크립트
"""

import requests
import json
import os
from pathlib import Path
from PIL import Image

# 서버 URL 및 테스트 설정
BASE_URL = "http://localhost:5173/api"  # Vite 개발 서버 주소
HEALTH_CHECK_URL = f"{BASE_URL}/health"
DIAGNOSIS_URL = f"{BASE_URL}/diagnosis/efficientnet"  # 'efficientnet' 모델로 테스트
IMAGE_PATH = "C:/2st-pro/skinseal-model/test/Acne/0001.jpg"
# 대체 이미지 경로 (없으면 생성)
FALLBACK_IMAGE_PATH = "c:/2st-pro/skinseal-pythonAI/uploads/test_sample.jpg"
USER_ID = "1"  # 테스트용 사용자 ID

def print_json(data):
    """JSON 데이터를 보기 좋게 출력"""
    print(json.dumps(data, indent=2, ensure_ascii=False))

def check_server_status():
    """서버 상태 확인"""
    print("\n1. 서버 상태 확인")
    try:
        response = requests.get(HEALTH_CHECK_URL)
        print(f"Status Code: {response.status_code}")
        print_json(response.json())
    except requests.exceptions.RequestException as e:
        print(f"서버 상태 확인 실패: {e}")

def test_diagnosis():
    """AI 진단 기능 테스트"""
    print("\n2. AI 진단 기능 테스트")
    
    image_to_use = IMAGE_PATH
    if not os.path.exists(image_to_use):
        print(f"기본 테스트 이미지를 찾을 수 없습니다: {IMAGE_PATH}")
        # 대체 이미지가 없으면 생성 (224x224 RGB JPEG)
        os.makedirs(os.path.dirname(FALLBACK_IMAGE_PATH), exist_ok=True)
        if not os.path.exists(FALLBACK_IMAGE_PATH):
            print(f"대체 테스트 이미지를 생성합니다: {FALLBACK_IMAGE_PATH}")
            img = Image.new('RGB', (224, 224), color=(200, 160, 120))
            img.save(FALLBACK_IMAGE_PATH, format='JPEG', quality=90)
        image_to_use = FALLBACK_IMAGE_PATH
        print(f"대체 이미지를 사용합니다: {image_to_use}")

    try:
        with open(image_to_use, 'rb') as f:
            files = {'file': (os.path.basename(image_to_use), f, 'image/jpeg')}
            data = {'userId': USER_ID}
            response = requests.post(DIAGNOSIS_URL, files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        try:
            print_json(response.json())
        except Exception:
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"진단 요청 실패: {e}")

def test_diagnosis_with_gradcam():
    """GradCAM이 포함된 AI 진단 기능 테스트"""
    print("\n3. GradCAM 포함 AI 진단 기능 테스트")
    
    image_to_use = IMAGE_PATH
    if not os.path.exists(image_to_use):
        print(f"기본 테스트 이미지를 찾을 수 없습니다: {IMAGE_PATH}")
        os.makedirs(os.path.dirname(FALLBACK_IMAGE_PATH), exist_ok=True)
        if not os.path.exists(FALLBACK_IMAGE_PATH):
            print(f"대체 테스트 이미지를 생성합니다: {FALLBACK_IMAGE_PATH}")
            img = Image.new('RGB', (224, 224), color=(200, 160, 120))
            img.save(FALLBACK_IMAGE_PATH, format='JPEG', quality=90)
        image_to_use = FALLBACK_IMAGE_PATH
        print(f"대체 이미지를 사용합니다: {image_to_use}")

    try:
        with open(image_to_use, 'rb') as f:
            files = {'file': (os.path.basename(image_to_use), f, 'image/jpeg')}
            data = {
                'userId': USER_ID,
                'gradcam': 'true'  # GradCAM 활성화
            }
            response = requests.post(DIAGNOSIS_URL, files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        try:
            result = response.json()
            print("진단 결과:")
            print_json(result.get('results', []))
            
            if 'gradcam' in result:
                gradcam = result['gradcam']
                print(f"\nGradCAM 정보:")
                print(f"- Target Index: {gradcam.get('targetIndex')}")
                print(f"- Score: {gradcam.get('score', 0):.4f}")
                print(f"- 원본 이미지 Base64 길이: {len(gradcam.get('original_base64', ''))}")
                print(f"- 히트맵 Base64 길이: {len(gradcam.get('heatmap_base64', ''))}")
                print(f"- 오버레이 Base64 길이: {len(gradcam.get('overlay_base64', ''))}")
            else:
                print("\nGradCAM 결과를 찾을 수 없습니다.")
        except Exception as e:
            print(f"JSON 파싱 오류: {e}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"GradCAM 진단 요청 실패: {e}")

def test_gradcam_endpoint():
    """전용 GradCAM 엔드포인트 테스트"""
    print("\n4. 전용 GradCAM 엔드포인트 테스트")
    
    gradcam_url = f"{BASE_URL}/diagnosis/efficientnet/gradcam"
    
    image_to_use = IMAGE_PATH
    if not os.path.exists(image_to_use):
        image_to_use = FALLBACK_IMAGE_PATH

    try:
        with open(image_to_use, 'rb') as f:
            files = {'file': (os.path.basename(image_to_use), f, 'image/jpeg')}
            data = {'userId': USER_ID}
            response = requests.post(gradcam_url, files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        try:
            result = response.json()
            print("진단 결과:")
            print_json(result.get('results', []))
            
            if 'gradcam' in result:
                gradcam = result['gradcam']
                print(f"\nGradCAM 정보:")
                print(f"- Target Index: {gradcam.get('targetIndex')}")
                print(f"- Score: {gradcam.get('score', 0):.4f}")
                print(f"- 원본 이미지 Base64 길이: {len(gradcam.get('original_base64', ''))}")
                print(f"- 히트맵 Base64 길이: {len(gradcam.get('heatmap_base64', ''))}")
                print(f"- 오버레이 Base64 길이: {len(gradcam.get('overlay_base64', ''))}")
        except Exception as e:
            print(f"JSON 파싱 오류: {e}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"전용 GradCAM 요청 실패: {e}")

if __name__ == "__main__":
    print("=== 질병진단 AI 서버 테스트 ===")
    check_server_status()
    test_diagnosis()
    test_diagnosis_with_gradcam()
    test_gradcam_endpoint()