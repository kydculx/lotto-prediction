#!/usr/bin/env python3
"""
🎓 1~1000회차 데이터로만 학습 (가중치 최적화)
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LottoDataLoader
from src.ensemble_predictor import EnsemblePredictor
import numpy as np
import multiprocessing as mp
from functools import partial


def run_backtest(matrix, weights, test_rounds=100, label=""):
    """백테스팅 실행"""
    n_draws = len(matrix)
    hit_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    for i in range(test_rounds + 1):
        if not label and i % 10 == 0:
            progress = i / test_rounds
            bar_len = 20
            filled_len = int(bar_len * progress)
            bar = "█" * filled_len + "░" * (bar_len - filled_len)
            print(f"\r|{bar}| {i:3d}/{test_rounds:3d}", end="", flush=True)
            
        if i == test_rounds:
            break
            
        test_idx = n_draws - test_rounds + i
        train_matrix = matrix[:test_idx]
        
        if len(train_matrix) < 100:
            continue
        
        # 예측
        predictor = EnsemblePredictor(train_matrix, weights=weights, use_ml=False, use_validator=False)
        predicted, _ = predictor.predict_single_set()
        
        # 실제 번호
        actual = set(matrix[test_idx])
        
        hits = len(set(predicted) & actual)
        hit_counts[hits] += 1
    
    total = sum(hit_counts.values())
    avg_hits = sum(h * c for h, c in hit_counts.items()) / total if total > 0 else 0
    
    return avg_hits, hit_counts


def mutate_weights(weights, mutation_rate=0.1):
    """가중치 돌연변이"""
    new_weights = weights.copy()
    
    engines = list(new_weights.keys())
    
    for _ in range(2):
        eng1, eng2 = np.random.choice(engines, size=2, replace=False)
        
        delta = np.random.uniform(0.01, mutation_rate)
        
        if new_weights[eng1] > delta:
            new_weights[eng1] -= delta
            new_weights[eng2] += delta
    
    # 정규화
    total = sum(new_weights.values())
    return {k: v / total for k, v in new_weights.items()}


def genetic_optimize(matrix, generations=10, population_size=10, test_rounds=100):
    """유전 알고리즘 기반 최적화"""
    
    # 초기 가중치 설정
    result_path = Path(__file__).parent / "trained_weights_1000.json"
    if result_path.exists():
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            base_weights = data['weights']
            best_score = data.get('best_score', 0)
            print(f"🔄 기존 학습 결과 로드 완료 (역대 최고 점수: {best_score:.4f})")
        except Exception as e:
            print(f"⚠️ 기존 학습 파일 로드 실패, 기본값으로 시작합니다: {e}")
            base_weights = {
                'statistical': 0.1600,
                'lstm': 0.1200,
                'sequence_correlation': 0.1200,
                'timeseries': 0.1000,
                'advanced_pattern': 0.1000,
                'pattern': 0.0800,
                'gap': 0.0800,
                'graph': 0.0800,
                'poisson': 0.0800,    # 신규 분석기 추가
                'fourier': 0.0800,    # 신규 분석기 추가
            }
            best_score = 0
    else:
        print("💡 신규 학습을 시작합니다 (기본 가중치 사용)")
        base_weights = {
            'statistical': 0.1600,
            'ml': 0.1000,
            'lstm': 0.1000,
            'sequence_correlation': 0.1000,
            'timeseries': 0.0900,
            'advanced_pattern': 0.0900,
            'pattern': 0.0800,
            'gap': 0.0800,
            'graph': 0.0800,
            'poisson': 0.0600,
            'fourier': 0.0500,
            'numerology': 0.0100,
        }
        best_score = 0
    
    # 초기 개체군 생성
    population = [base_weights.copy()]
    for _ in range(population_size - 1):
        population.append(mutate_weights(base_weights, 0.15))
    
    best_weights = base_weights.copy()
    
    print("=" * 60)
    print("🎓 1~1000회차 데이터로 학습 시작")
    print("=" * 60)
    print(f"   세대 수: {generations}")
    print(f"   개체군 크기: {population_size}")
    print(f"   검증 회차: {test_rounds}")
    print()
    # 가용 코어의 50%만 사용하여 시스템 안정성 확보
    num_cores = max(1, mp.cpu_count() // 2)
    
    for gen in range(generations):
        print(f"\n{'='*60}")
        print(f"🧬 세대 {gen+1}/{generations} 평가 중 (CPU 코어 {num_cores}개 활용)")
        print(f"{'='*60}")
        
        # 병렬 평가를 위한 함수 래퍼 (데이터는 고정, 가중치만 변경)
        eval_func = partial(run_backtest, matrix, test_rounds=test_rounds, label="parallel")
        
        # 프로세스 풀 생성 및 실행
        pool = mp.Pool(processes=num_cores)
        try:
            results = []
            # imap을 사용하여 순차적으로 결과를 받으며 진행률 표시
            for i, res in enumerate(pool.imap(eval_func, population)):
                score, _ = res
                results.append((score, population[i]))
                print(f"\r  🏃 개체 평가 진행률: [{i+1}/{population_size}] 점수: {score:.4f}", end="", flush=True)
            
            pool.close()
            pool.join()
            fitness = results
        except KeyboardInterrupt:
            print("\n⚠️ 사용자에 의해 학습이 중단되었습니다. 하위 프로세스를 정리합니다...")
            pool.terminate()
            pool.join()
            raise # 상위로 전달하여 프로그램 종료
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            pool.terminate()
            pool.join()
            raise
            
        print() 
        
        # 정렬
        fitness.sort(key=lambda x: x[0], reverse=True)
        
        # 최고 기록 갱신
        if fitness[0][0] > best_score:
            best_score = fitness[0][0]
            best_weights = fitness[0][1].copy()
            print(f"\n🎯 새로운 최고 점수! {best_score:.4f} (이전: {fitness[0][0]:.4f})")
        else:
            print(f"\n   현재 세대 최고: {fitness[0][0]:.4f} | 역대 최고: {best_score:.4f}")
        
        # 상위 50% 선택
        survivors = [w for _, w in fitness[:population_size // 2]]
        
        # 새 개체군 생성
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent = survivors[np.random.randint(len(survivors))]
            child = mutate_weights(parent, 0.08)
            new_population.append(child)
        
        population = new_population
    
    print()
    print("=" * 60)
    print(f"✅ 학습 완료!")
    print(f"   최고 점수: {best_score:.4f}")
    print("=" * 60)
    
    return best_weights, best_score


def main():
    # 데이터 로드
    print("\n⏳ 데이터 로딩...")
    loader = LottoDataLoader()
    full_matrix = loader.get_numbers_matrix()
    
    # 1~1000회차만 사용
    train_matrix = full_matrix[:1000]
    
    print(f"✅ 학습 데이터: 1~1000회차 (총 {len(train_matrix)}개)")
    print(f"📌 1001회차 이후는 실전 테스트용으로 보존\n")
    
    # 유전 알고리즘 최적화
    best_weights, best_score = genetic_optimize(
        train_matrix,
        generations=20,    # 원래대로 유지
        population_size=12,
        test_rounds=200    # 원래대로 유지 (사용자 요청)
    )
    
    # 결과 출력
    print("\n📊 최적화된 가중치:")
    for name, weight in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 50)
        print(f"  {name:22s}: {weight:.4f} {bar}")
    
    # 결과 저장
    result = {
        'training_rounds': '1-1000',
        'best_score': best_score,
        'weights': best_weights
    }
    
    result_path = Path(__file__).parent / "trained_weights_1000.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📁 학습 결과가 {result_path}에 저장되었습니다.")
    print("\n💡 이제 이 가중치로 1001회차 이후를 예측할 수 있습니다!")


if __name__ == "__main__":
    main()
