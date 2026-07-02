#!/usr/bin/env python3
"""
🎓 1~1000회차 데이터로만 학습 (가중치 최적화)
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LottoDataLoader
from src.ensemble_predictor import EnsemblePredictor
from src.optimization_cache import OptimizationCache
import numpy as np

import multiprocessing as mp
from functools import partial
import threading
import time


def worker_eval_cached(args):
    """캐시된 데이터를 이용한 초고속 평가"""
    weights, cached_data = args
    avg_hits, _ = OptimizationCache.evaluate_weights(cached_data, weights)
    return avg_hits, weights


def worker_eval(args):
    """(구) 병렬 처리를 위한 작업자 함수 wrapper - 더 이상 사용 안 함"""
    weights, matrix, test_rounds = args
    avg_hits, _ = run_backtest(matrix, weights, test_rounds=test_rounds, label="parallel")
    return avg_hits, weights


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
    # 초기 가중치 설정 (최신 12대 엔진 라인업)
    default_base_weights = {
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

    if result_path.exists():
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            base_weights = data['weights']
            best_score = data.get('best_score', 0)
            print(f"🔄 기존 학습 결과 로드 완료 (역대 최고 점수: {best_score:.4f})")
            
            # 혹시 기존 파일에 누락된 엔진이 있다면 기본값으로 채워줌
            for key in default_base_weights:
                if key not in base_weights:
                    base_weights[key] = default_base_weights[key]
        except Exception as e:
            print(f"⚠️ 기존 학습 파일 로드 실패, 기본값으로 시작합니다: {e}")
            base_weights = default_base_weights.copy()
            best_score = 0
    else:
        print("💡 신규 학습을 시작합니다 (기본 가중치 사용)")
        base_weights = default_base_weights.copy()
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
    
    # ⚡️ 최적화 캐시 생성 (여기서 한 번만 무거운 연산 수행)
    cache = OptimizationCache()
    cached_data = cache.precalculate(matrix, test_rounds)
    
    # 가용 코어 전체 사용 (시스템 여유분 1개 확보)
    num_cores = max(1, mp.cpu_count() - 1)
    
    for gen in range(generations):
        print(f"\n{'='*60}")
        print(f"🧬 세대 {gen+1}/{generations} 평가 중 (CPU 코어 {num_cores}개 활용)")
        print(f"{'='*60}")
        
        # 병렬 평가를 위한 인자 준비 (캐시 데이터 전달)
        # cached_data는 읽기 전용이므로 여러 프로세스에서 공유 가능
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
                
                # 메인 스레드와 값 충돌 방지 (읽기만 하므로 lock 없어도 안전하지만 명시적으로)
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
            results = []
            # imap_unordered 사용
            for i, res in enumerate(pool.imap_unordered(worker_eval_cached, task_args)):
                score, weights = res
                results.append((score, weights))
                
                with lock:
                    completed_count += 1
                    if score > current_best_in_gen:
                        current_best_in_gen = score
            
            monitor_thread.join() # 스레드 종료 대기
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
        
        generation_best_score = fitness[0][0]
        generation_best_weights = fitness[0][1]
        
        # 최고 기록 갱신
        if generation_best_score > best_score:
            best_score = generation_best_score
            best_weights = generation_best_weights
            print(f"\n🎯 새로운 최고 점수! {best_score:.4f}")
            
            # 💾 즉시 자동 저장
            try:
                save_data = {
                    "best_score": best_score,
                    "weights": best_weights,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                print(f"   💾 가중치가 자동 저장되었습니다: {result_path.name}")
            except Exception as e:
                print(f"   ⚠️ 자동 저장 실패: {e}")
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
        generations=20000,   # 무한에 가까운 반복 (사용자가 중단할 때까지)
        population_size=50,
        test_rounds=500
    )
    
    print("\n" + "="*60)
    print("✅ 학습이 중단되거나 완료되었습니다.")
    
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
