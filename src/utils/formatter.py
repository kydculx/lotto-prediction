"""
로또 결과 출력 포맷터
"""

from typing import Dict, List, Tuple


class LottoFormatter:
    """로또 분석 결과 및 예측 결과를 예쁘게 출력"""
    
    @staticmethod
    def print_header(next_round: int):
        print()
        print("=" * 60)
        print(f"🎱 로또 {next_round}회차 예측 시스템")
        print("   Multi-Engine Ensemble Predictor v3.0 (Refactored)")
        print("=" * 60)

    @staticmethod
    def print_hot_cold(hot_cold: dict):
        print("\n🔥 핫 넘버 (최근 50회 기준):")
        hot_nums = [f"{num}" for num, freq in hot_cold.get('hot', [])[:6]]
        print(f"   {', '.join(hot_nums)}")
        
        print("\n❄️  콜드 넘버 (최근 50회 기준):")
        cold_nums = [f"{num}" for num, freq in hot_cold.get('cold', [])[:6]]
        print(f"   {', '.join(cold_nums)}")
        
        if hot_cold.get('overdue'):
            print("\n⏰ 과도 지연 번호 (평균 출현 주기 대비):")
            overdue = hot_cold['overdue'][:6]
            overdue_str = [f"{num}(x{ratio:.1f})" for num, ratio in overdue]
            print(f"   {', '.join(overdue_str)}")

    @staticmethod
    def print_engine_predictions(predictions: dict):
        print("\n" + "-" * 60)
        print("📊 엔진별 추천 번호:")
        print("-" * 60)
        
        engine_names = {
            'statistical': '통계 엔진   ',
            'pattern': '패턴 엔진   ',
            'timeseries': '시계열 엔진 ',
            'lstm': 'LSTM 엔진   ',
            'graph': '그래프 엔진 ',
            'numerology': '숫자론 엔진 ',
            'advanced_pattern': '고급패턴   ',
            'sequence_correlation': '연속상관   ',
            'ml': 'ML 엔진    ',
            'gap': 'Gap 엔진   ',
        }
        
        for key, name in engine_names.items():
            if key in predictions and predictions[key]:
                nums = predictions[key]
                nums_str = ', '.join(f"{n:2d}" for n in nums)
                print(f"   {name}: [{nums_str}]")

    @staticmethod
    def print_final_predictions(predicted_sets: list):
        print("\n" + "=" * 60)
        print("🎯 앙상블 최종 예측")
        print("=" * 60)
        
        for i, (nums, confidence) in enumerate(predicted_sets, 1):
            nums_str = ', '.join(f"{n:2d}" for n in nums)
            stars = "⭐" * min(int(confidence / 20) + 1, 5)
            
            print(f"\n   ✨ 추천 SET {i}: [{nums_str}]")
            
            combo_sum = sum(nums)
            odd_count = sum(1 for n in nums if n % 2 == 1)
            
            # AC값 계산
            sorted_nums = sorted(nums)
            differences = {sorted_nums[k] - sorted_nums[j] for j in range(6) for k in range(j+1, 6)}
            ac_value = len(differences) - 5
            
            print(f"      신뢰도: {confidence:.1f}% {stars}")
            print(f"      합계: {combo_sum} | 홀짝: {odd_count}:{6-odd_count} | AC: {ac_value}")

    @staticmethod
    def print_footer():
        print("\n" + "=" * 60)
        print("⚠️  주의: 로또는 완전 무작위 추첨입니다.")
        print("   이 예측은 통계적 분석에 기반한 참고용이며,")
        print("   실제 당첨을 보장하지 않습니다.")
        print("=" * 60)
        print()

    @staticmethod
    def print_backtest_report(hit_counts: dict, avg_hits: float):
        print("\n   적중 분포:")
        total_tests = sum(hit_counts.values())
        for hits, count in sorted(hit_counts.items(), reverse=True):
            if count > 0:
                pct = count / total_tests * 100 if total_tests > 0 else 0
                bar = "█" * int(pct / 5)
                print(f"   {hits}개 적중: {count:3d}회 ({pct:5.1f}%) {bar}")
        print(f"\n   평균 적중 개수: {avg_hits:.2f}개")
