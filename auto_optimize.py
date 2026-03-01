#!/usr/bin/env python3
"""
🔄 자동 최적화 루프
적중률을 반복적으로 개선하는 자동화 스크립트
"""

import sys
import os

# 병렬 처리 성능 최적화: 라이브러리 내부 스레딩 비활성화 (프로세스 병렬화 집중)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LottoDataLoader
from src.ensemble_predictor import EnsemblePredictor
from src.optimization_cache import OptimizationCache
import numpy as np
import multiprocessing as mp
from functools import partial
import threading



def worker_eval_cached(args):
    """캐시된 데이터를 이용한 초고속 평가"""
    weights, cached_data = args
    avg_hits, _ = OptimizationCache.evaluate_weights(cached_data, weights)
    return avg_hits, weights


def worker_eval(args):
    """(구) 병렬 처리를 위한 작업자 함수 wrapper - 더 이상 사용 안 함"""
    weights, matrix, test_rounds = args
    avg_hits, _ = run_backtest(matrix, weights, test_rounds, label="parallel")
    return avg_hits, weights


def run_backtest(matrix, weights, test_rounds=50, label=""):
    """백테스팅 실행"""
    n_draws = len(matrix)
    
    # 데이터보다 테스트 회차가 많으면 최대치로 조정 (최소 100회 학습 데이터 남김)
    if test_rounds >= n_draws - 100:
        test_rounds = n_draws - 100
        print(f"⚠️ 테스트 회차가 전체 데이터보다 많아 {test_rounds}회로 조정되었습니다.")

    hit_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    for i in range(test_rounds):
        test_idx = n_draws - test_rounds + i
        train_matrix = matrix[:test_idx]
        
        if len(train_matrix) < 100:
            continue
        
        # 예측
        predictor = EnsemblePredictor(train_matrix, weights=weights, use_ml=True, use_validator=True)
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
    
    # 랜덤 엔진 선택하여 가중치 조정
    engines = list(new_weights.keys())
    
    for _ in range(2):  # 2개 엔진 조정
        eng1, eng2 = np.random.choice(engines, size=2, replace=False)
        
        # 가중치 이동
        delta = np.random.uniform(0.01, mutation_rate)
        
        if new_weights[eng1] > delta:
            new_weights[eng1] -= delta
            new_weights[eng2] += delta
    
    # 정규화
    total = sum(new_weights.values())
    return {k: v / total for k, v in new_weights.items()}


def genetic_optimize(matrix, generations=10, population_size=10, test_rounds=30):
    """유전 알고리즘 기반 최적화"""
    
    # 초기 가중치
    base_weights = {
        'statistical': 0.18,
        'advanced_pattern': 0.12,
        'sequence_correlation': 0.12,
        'ml': 0.10,           # 머신러닝 엔진 추가
        'lstm': 0.08,
        'timeseries': 0.08,
        'gap': 0.08,
        'poisson': 0.08,
        'fourier': 0.08,
        'graph': 0.04,
        'pattern': 0.03,
        'numerology': 0.01,    # 수비학 엔진 추가
    }
    
    # 초기 개체군 생성
    population = [base_weights.copy()]
    for _ in range(population_size - 1):
        population.append(mutate_weights(base_weights, 0.15))
    
    best_weights = base_weights.copy()
    best_score = 0
    
    print("=" * 60)
    print("🧬 유전 알고리즘 최적화 시작")
    print("=" * 60)
    print(f"   세대 수: {generations}")
    print(f"   개체군 크기: {population_size}")
    print(f"   테스트 회차: {test_rounds}")
    print()
    
    # 코어 설정 (시스템 여유분 1개 확보)
    num_cores = max(1, mp.cpu_count() - 1)
    
    # ⚡️ 최적화 캐시 생성
    cache = OptimizationCache()
    cached_data = cache.precalculate(matrix, test_rounds)

    for gen in range(generations):
        print(f"🧬 세대 {gen+1}/{generations} 평가 중 (CPU 코어 {num_cores}개 활용)")
        
        # 병렬 평가를 위한 인자 준비 (캐시 데이터)
        task_args = [(w, cached_data) for w in population]
        
        # 상태 공유 변수
        completed_count = 0
        current_best_in_gen = 0.0
        lock = threading.Lock()
        
        def _progress_monitor():
            start_time = time.time()
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            idx = 0
            
            while completed_count < population_size:
                elapsed = time.time() - start_time
                percent = completed_count / population_size * 100
                bar_len = 30
                filled = int(bar_len * completed_count / population_size)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                spin = spinner[idx % len(spinner)]
                idx += 1
                
                # 메인 스레드와 값 충돌 방지
                curr_score = current_best_in_gen
                
                status = f"\r  {spin} [시간: {elapsed:3.0f}s] |{bar}| {percent:5.1f}% ({completed_count}/{population_size}) - 최고점수: {curr_score:.4f}"
                sys.stdout.write(status)
                sys.stdout.flush()
                time.sleep(0.1)
                
            # 완료 후 최종 출력
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r  ✅ [시간: {elapsed:3.0f}s] |{'█'*30}| 100.0% ({population_size}/{population_size}) - 최고점수: {current_best_in_gen:.4f}\n")
            sys.stdout.flush()

        # 프로세스 풀 생성 및 실행
        pool = mp.Pool(processes=num_cores)
        
        # 모니터링 스레드 시작
        monitor_thread = threading.Thread(target=_progress_monitor)
        monitor_thread.start()
        
        try:
            fitness = []
            # imap_unordered 사용
            for i, res in enumerate(pool.imap_unordered(worker_eval_cached, task_args)):
                score, weights = res
                fitness.append((score, weights))
                
                with lock:
                    completed_count += 1
                    if score > current_best_in_gen:
                        current_best_in_gen = score
            
            monitor_thread.join()
            pool.close()
            pool.join()
            
        except KeyboardInterrupt:
            print("\n⚠️ 사용자에 의해 학습이 중단되었습니다.")
            pool.terminate()
            raise
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            pool.terminate()
            raise
        
        # 정렬
        fitness.sort(key=lambda x: x[0], reverse=True)
        
        # 최고 기록 갱신
        if fitness[0][0] > best_score:
            best_score = fitness[0][0]
            best_weights = fitness[0][1].copy()
            print(f"🎯 세대 {gen+1} 결과: 새로운 최고 점수 = {best_score:.4f}")
        else:
            print(f"   세대 {gen+1} 결과: 현재 최고 = {fitness[0][0]:.4f}")
        
        # 상위 50% 선택
        survivors = [w for _, w in fitness[:population_size // 2]]
        
        # 새 개체군 생성 (돌연변이)
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent = survivors[np.random.randint(len(survivors))]
            child = mutate_weights(parent, 0.08)
            new_population.append(child)
        
        population = new_population
    
    print()
    print("=" * 60)
    print(f"✅ 최적화 완료!")
    print(f"   최고 점수: {best_score:.4f}")
    print("=" * 60)
    
    return best_weights, best_score


def apply_weights_to_predictor(weights):
    """최적화된 가중치를 ensemble_predictor.py에 적용"""
    predictor_path = Path(__file__).parent / "src" / "ensemble_predictor.py"
    
    with open(predictor_path, 'r') as f:
        content = f.read()
    
    # 가중치 문자열 생성
    weights_str = "    # 최적화된 엔진 가중치 (자동 최적화)\n    DEFAULT_WEIGHTS = {\n"
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        weights_str += f"        '{name}': {weight:.4f},\n"
    weights_str += "    }"
    
    # 기존 가중치 교체
    import re
    pattern = r"    # 최적화된 엔진 가중치.*?DEFAULT_WEIGHTS = \{[^}]+\}"
    content = re.sub(pattern, weights_str, content, flags=re.DOTALL)
    
    with open(predictor_path, 'w') as f:
        f.write(content)
    
    print(f"✅ 가중치가 {predictor_path}에 저장되었습니다.")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='자동 최적화 루프')
    parser.add_argument('--generations', type=int, default=100000, help='세대 수')
    parser.add_argument('--population', type=int, default=12, help='개체군 크기')
    parser.add_argument('--test-rounds', type=int, default=200, help='테스트 회차')
    parser.add_argument('--apply', action='store_true', help='최적화 결과 적용')
    
    args = parser.parse_args()
    
    # 데이터 로드
    print("\n⏳ 데이터 로딩...")
    loader = LottoDataLoader()
    loader.load()
    matrix = loader.get_numbers_matrix()
    print(f"✅ {len(matrix)}회차 로드 완료")
    
    # 유전 알고리즘 최적화
    best_weights, best_score = genetic_optimize(
        matrix,
        generations=args.generations,
        population_size=args.population,
        test_rounds=args.test_rounds
    )
    
    # 결과 출력
    print("\n📊 최적화된 가중치:")
    for name, weight in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 50)
        print(f"  {name:22s}: {weight:.4f} {bar}")
    
    # 최종 백테스팅
    print("\n🔬 최종 백테스팅 (50회차)...")
    final_score, hit_counts = run_backtest(matrix, best_weights, 50)
    
    print("\n   적중 분포:")
    for hits, count in sorted(hit_counts.items(), reverse=True):
        if count > 0:
            pct = count / 50 * 100
            bar = "█" * int(pct / 5)
            print(f"   {hits}개 적중: {count:3d}회 ({pct:5.1f}%) {bar}")
    
    print(f"\n   평균 적중: {final_score:.4f}")
    
    # 가중치 적용
    if args.apply:
        apply_weights_to_predictor(best_weights)
    else:
        print("\n💡 --apply 옵션으로 가중치를 자동 적용할 수 있습니다.")
    
    # 결과 저장
    result = {
        'best_score': best_score,
        'weights': best_weights,
        'hit_counts': hit_counts
    }
    
    result_path = Path(__file__).parent / "optimization_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📁 결과가 {result_path}에 저장되었습니다.")


if __name__ == "__main__":
    main()
