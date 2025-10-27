"""
Acne 이진 모델 엔드포인트 테스트 클라이언트
"""
import requests
import os
from PIL import Image

BASE_URL = "http://localhost:5173/api"  # 프론트 프록시 경로 사용 시
# 직접 Flask를 노출했다면 아래 사용 (예: 5000)
# BASE_URL = "http://localhost:5000"

URL = f"{BASE_URL}/predict-acne"
IMAGE_PATH = "C:/2st-pro/skinseal-model/test/Acne/0001.jpg"
FALLBACK = "C:/2st-pro/skinseal-pythonAI/uploads/test_sample.jpg"
USER_ID = "1"


def main():
    img_path = IMAGE_PATH
    if not os.path.exists(img_path):
        os.makedirs(os.path.dirname(FALLBACK), exist_ok=True)
        if not os.path.exists(FALLBACK):
            Image.new('RGB', (224, 224), color=(180, 150, 120)).save(FALLBACK, format='JPEG')
        img_path = FALLBACK

    with open(img_path, 'rb') as f:
        files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
        data = {'userId': USER_ID}
        resp = requests.post(URL, files=files, data=data)
        print(resp.status_code)
        try:
            print(resp.json())
        except Exception:
            print(resp.text)


if __name__ == "__main__":
    main()
