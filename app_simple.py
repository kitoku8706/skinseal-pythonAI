"""
간단한 Flask 앱 (고급 기능 없이)
의존성 설치가 완료되기 전까지 사용할 수 있는 기본 버전
"""

from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'message': '질병진단 AI 서버가 정상 작동 중입니다.',
        'version': '1.0.0'
    })

@app.route('/diagnose', methods=['POST'])
def diagnose_simple():
    """간단한 진단 엔드포인트 (모델 없이)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '이미지 파일이 제공되지 않았습니다.'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        # 파일 형식 간단 체크
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': '지원되지 않는 파일 형식입니다.'}), 400
        
        # 예시 결과 반환 (실제 모델 대신)
        import random
        diagnoses = ['정상', '피부염', '습진', '건선', '기타']
        
        result = {
            'success': True,
            'filename': file.filename,
            'diagnosis': random.choice(diagnoses),
            'confidence': round(random.uniform(0.7, 0.95), 3),
            'note': '이것은 테스트 결과입니다. 실제 모델을 로드하면 정확한 진단이 가능합니다.'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'처리 중 오류 발생: {str(e)}'
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info_simple():
    """모델 정보 (간단 버전)"""
    return jsonify({
        'model_loaded': False,
        'status': '모델 파일이 필요합니다',
        'model_path': 'models/20251006_212412_best_efficientnet.pth',
        'note': 'PyTorch 패키지 설치 후 실제 모델을 사용할 수 있습니다.'
    })

if __name__ == '__main__':
    print("질병진단 AI 서버 (간단 버전)을 시작합니다...")
    print("모든 의존성 설치 후 app.py를 사용하세요.")
    app.run(host='0.0.0.0', port=5000, debug=True)