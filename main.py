#!/usr/bin/env python3
"""
🎱 로또 당첨번호 예측 시스템 (Refactored)
Multi-Engine Ensemble Predictor
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LottoDataLoader
from src.ensemble_predictor import EnsemblePredictor
from src.utils.formatter import LottoFormatter


def run_backtest(loader, last_n: int = 100):
    """과거 데이터로 백테스트"""
    print(f"\n🔬 백테스팅 (최근 {last_n}회차)")
    print("-" * 60)
    
    df = loader.df
    total_draws = len(df)
    hit_counts = {i: 0 for i in range(7)}
    total_hits = 0
    
    for i in range(last_n):
        test_idx = total_draws - last_n + i
        train_matrix = loader.get_numbers_matrix()[:test_idx]
        
        if len(train_matrix) < 100: continue
        
        predictor = EnsemblePredictor(train_matrix, use_ml=False, use_validator=False)
        
        # 실제 정답 번호 가져오기
        actual = set(loader.get_draw_by_round(int(df.iloc[test_idx]['round'])))

        # 5개 세트 예측
        predicted_sets = predictor.predict_multiple_sets(5)
        
        # 5개 중 가장 잘 맞은 것 기준 (사용자 입장에서의 당첨 여부)
        best_hit = 0
        best_set = None
        
        for pred, _ in predicted_sets:
            hits = len(set(pred) & actual)
            if hits > best_hit:
                best_hit = hits
                best_set = pred
        
        # 5세트 중 하나라도 없으면 첫 번째 세트로 설정 (출력용)
        if best_set is None:
            best_set = predicted_sets[0][0]
            
        hit_counts[best_hit] += 1
        total_hits += best_hit
        
        # 실시간 로그 출력 (최고 성적 기준)
        print(f"[{test_idx+1}회차] 최고 적중: {best_hit}개 | 예측: {sorted(best_set)} | 정답: {sorted(list(actual))}")
    
    LottoFormatter.print_backtest_report(hit_counts, total_hits / last_n if last_n > 0 else 0)


def main():
    parser = argparse.ArgumentParser(description='로또 당첨번호 예측 시스템')
    parser.add_argument('--sets', type=int, default=5, help='예측 세트 수')
    parser.add_argument('--backtest', action='store_true', help='백테스팅 실행')
    parser.add_argument('--last', type=int, default=100, help='백테스팅 회차 수')
    parser.add_argument('--simple', action='store_true', help='간단 출력 모드')
    
    args = parser.parse_args()
    
    print("\n⏳ 데이터 로딩 및 분석 엔진 초기화 중...")
    loader = LottoDataLoader()
    # 최신 데이터 확인 및 동기화 추가
    loader.check_for_updates()
    matrix = loader.get_numbers_matrix()
    
    if args.backtest:
        run_backtest(loader, args.last)
        return
    
    predictor = EnsemblePredictor(matrix)
    predicted_sets = predictor.predict_multiple_sets(args.sets)
    
    LottoFormatter.print_header(loader.get_latest_round() + 1)
    
    if not args.simple:
        LottoFormatter.print_hot_cold(predictor.get_hot_cold_analysis())
        LottoFormatter.print_engine_predictions(predictor.engine_predictions)
    
    LottoFormatter.print_final_predictions(predicted_sets)
    LottoFormatter.print_footer()


if __name__ == "__main__":
    main()
