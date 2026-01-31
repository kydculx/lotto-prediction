from flask import Flask, render_template, jsonify
import numpy as np
from src.data_loader import LottoDataLoader
from src.ensemble_predictor import EnsemblePredictor
from pathlib import Path
import json

app = Flask(__name__)

# 데이터 로더 초기화
loader = LottoDataLoader()
loader.load()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/predict')
def predict():
    """예측 결과 API"""
    try:
        matrix = loader.get_numbers_matrix()
        predictor = EnsemblePredictor(matrix)
        report = predictor.get_detailed_report()
        
        # JSON 직렬화 가능하도록 변환
        serialized_report = {
            'latest_round': int(loader.get_latest_round()),
            'next_round': int(loader.get_latest_round() + 1),
            'hot_cold': report['hot_cold'],
            'engine_predictions': {k: [int(n) for n in v] for k, v in report['engine_predictions'].items()},
            'predicted_sets': [
                {'numbers': [int(n) for n in s[0]], 'confidence': float(s[1])}
                for s in report['predicted_sets']
            ],
            'sum_range': report['sum_range']
        }
        
        return jsonify(serialized_report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    """추가 통계 데이터 API (필요 시 확장)"""
    return jsonify({
        'total_draws': len(loader.df),
        'latest_draw': [int(n) for n in loader.get_numbers_matrix()[-1]]
    })

@app.route('/api/frequencies')
def frequencies():
    """번호별 출현 빈도 데이터 API"""
    try:
        all_numbers = loader.get_all_numbers_flat()
        unique, counts = np.unique(all_numbers, return_counts=True)
        
        # 1-45 전체 번호에 대한 빈도수 보장
        freq_dict = {int(i): 0 for i in range(1, 46)}
        for num, count in zip(unique, counts):
            freq_dict[int(num)] = int(count)
            
        return jsonify(freq_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Mac에서 5000번 포트는 AirPlay와 충돌할 수 있어 5001번을 권장합니다.
    app.run(host='0.0.0.0', port=8002, debug=True)
