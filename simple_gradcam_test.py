"""
간단한 GradCAM 테스트
"""
import requests
import json

def test_gradcam_endpoint():
    url = "http://localhost:5000/api/diagnosis/efficientnet/gradcam"
    
    try:
        with open("uploads/test_sample.jpg", 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {'userId': 'test_user'}
            response = requests.post(url, files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("응답 키들:", list(result.keys()))
            
            if 'gradcam' in result:
                gradcam = result['gradcam']
                print("GradCAM 키들:", list(gradcam.keys()))
                print("original_base64 존재:", 'original_base64' in gradcam)
                if 'original_base64' in gradcam:
                    print("original_base64 길이:", len(gradcam['original_base64']))
                else:
                    print("original_base64가 응답에 없습니다!")
        else:
            print("오류 응답:", response.text)
            
    except Exception as e:
        print(f"테스트 실패: {e}")

if __name__ == "__main__":
    test_gradcam_endpoint()
