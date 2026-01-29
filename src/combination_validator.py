"""
번호 조합 검증기
AC값, 연속번호, 끝수 다양성, 홀짝/고저 비율 등 검증
"""

from typing import List, Tuple, Dict, Set
from itertools import combinations
import numpy as np


class CombinationValidator:
    """번호 조합 검증기"""
    
    # 최적 범위 설정
    OPTIMAL_AC_RANGE = (7, 10)
    OPTIMAL_SUM_RANGE = (100, 175)
    OPTIMAL_ODD_RANGE = (2, 4)  # 홀수 2~4개
    OPTIMAL_HIGH_RANGE = (2, 4)  # 고번호(23-45) 2~4개
    MIN_ENDING_DIVERSITY = 4  # 최소 4가지 다른 끝수
    MAX_CONSECUTIVE_PAIRS = 2  # 연속번호 쌍 최대 2개
    
    @staticmethod
    def calculate_ac(numbers: List[int]) -> int:
        """AC값 계산 (Arithmetic Complexity)"""
        sorted_nums = sorted(numbers)
        differences = set()
        for i in range(len(sorted_nums)):
            for j in range(i+1, len(sorted_nums)):
                differences.add(sorted_nums[j] - sorted_nums[i])
        return len(differences) - 5
    
    @staticmethod
    def count_consecutive_pairs(numbers: List[int]) -> int:
        """연속번호 쌍 개수"""
        sorted_nums = sorted(numbers)
        count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                count += 1
        return count
    
    @staticmethod
    def count_odd_numbers(numbers: List[int]) -> int:
        """홀수 개수"""
        return sum(1 for n in numbers if n % 2 == 1)
    
    @staticmethod
    def count_high_numbers(numbers: List[int]) -> int:
        """고번호(23-45) 개수"""
        return sum(1 for n in numbers if n >= 23)
    
    @staticmethod
    def count_ending_diversity(numbers: List[int]) -> int:
        """끝수 다양성 (서로 다른 1의 자리 개수)"""
        endings = set(n % 10 for n in numbers)
        return len(endings)
    
    @staticmethod
    def get_section_distribution(numbers: List[int]) -> Dict[int, int]:
        """구간별 분포"""
        sections = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for n in numbers:
            if n <= 10:
                sections[1] += 1
            elif n <= 20:
                sections[2] += 1
            elif n <= 30:
                sections[3] += 1
            elif n <= 40:
                sections[4] += 1
            else:
                sections[5] += 1
        return sections
    
    def validate(self, numbers: List[int]) -> Tuple[bool, Dict[str, any]]:
        """
        조합 유효성 검증
        
        Returns:
            (유효 여부, 상세 결과)
        """
        results = {
            'sum': sum(numbers),
            'ac': self.calculate_ac(numbers),
            'consecutive_pairs': self.count_consecutive_pairs(numbers),
            'odd_count': self.count_odd_numbers(numbers),
            'high_count': self.count_high_numbers(numbers),
            'ending_diversity': self.count_ending_diversity(numbers),
            'sections': self.get_section_distribution(numbers)
        }
        
        # 검증
        is_valid = True
        violations = []
        
        # AC값 검증
        if not (self.OPTIMAL_AC_RANGE[0] <= results['ac'] <= self.OPTIMAL_AC_RANGE[1]):
            violations.append(f"AC값 {results['ac']} (권장: {self.OPTIMAL_AC_RANGE})")
            is_valid = False
        
        # 합계 검증
        if not (self.OPTIMAL_SUM_RANGE[0] <= results['sum'] <= self.OPTIMAL_SUM_RANGE[1]):
            violations.append(f"합계 {results['sum']} (권장: {self.OPTIMAL_SUM_RANGE})")
            is_valid = False
        
        # 홀수 비율 검증
        if not (self.OPTIMAL_ODD_RANGE[0] <= results['odd_count'] <= self.OPTIMAL_ODD_RANGE[1]):
            violations.append(f"홀수 {results['odd_count']}개 (권장: {self.OPTIMAL_ODD_RANGE})")
            is_valid = False
        
        # 끝수 다양성 검증
        if results['ending_diversity'] < self.MIN_ENDING_DIVERSITY:
            violations.append(f"끝수 다양성 {results['ending_diversity']} (최소: {self.MIN_ENDING_DIVERSITY})")
            is_valid = False
        
        results['is_valid'] = is_valid
        results['violations'] = violations
        
        return is_valid, results
    
    def score(self, numbers: List[int]) -> float:
        """
        조합 품질 점수 (0~1)
        높을수록 좋은 조합
        """
        is_valid, results = self.validate(numbers)
        
        score = 0.0
        
        # AC값 점수 (0.3)
        ac = results['ac']
        if self.OPTIMAL_AC_RANGE[0] <= ac <= self.OPTIMAL_AC_RANGE[1]:
            score += 0.3
        elif 6 <= ac <= 10:
            score += 0.2
        else:
            score += 0.1
        
        # 합계 점수 (0.25)
        total = results['sum']
        if self.OPTIMAL_SUM_RANGE[0] <= total <= self.OPTIMAL_SUM_RANGE[1]:
            score += 0.25
        elif 90 <= total <= 185:
            score += 0.15
        else:
            score += 0.05
        
        # 홀짝 점수 (0.2)
        odds = results['odd_count']
        if 2 <= odds <= 4:
            score += 0.2
        elif 1 <= odds <= 5:
            score += 0.1
        
        # 끝수 다양성 점수 (0.15)
        diversity = results['ending_diversity']
        score += min(diversity / 6, 1) * 0.15
        
        # 구간 분포 점수 (0.1)
        sections = results['sections']
        covered = sum(1 for v in sections.values() if v > 0)
        score += (covered / 5) * 0.1
        
        return score


class CombinationOptimizer:
    """최적 조합 생성기"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        self.numbers_matrix = numbers_matrix
        self.validator = CombinationValidator()
        self._analyze_historical_patterns()
    
    def _analyze_historical_patterns(self):
        """과거 당첨 조합 패턴 분석"""
        ac_values = []
        sums = []
        odd_counts = []
        consecutive_counts = []
        
        for row in self.numbers_matrix:
            row_list = list(row)
            ac_values.append(self.validator.calculate_ac(row_list))
            sums.append(sum(row_list))
            odd_counts.append(self.validator.count_odd_numbers(row_list))
            consecutive_counts.append(self.validator.count_consecutive_pairs(row_list))
        
        self.historical_patterns = {
            'ac_mean': np.mean(ac_values),
            'ac_std': np.std(ac_values),
            'sum_mean': np.mean(sums),
            'sum_std': np.std(sums),
            'odd_mean': np.mean(odd_counts),
            'consecutive_mean': np.mean(consecutive_counts)
        }
    
    def find_optimal_combinations(self, 
                                   candidates: List[int], 
                                   n_numbers: int = 6,
                                   n_results: int = 10) -> List[Tuple[List[int], float]]:
        """
        후보 번호에서 최적 조합 찾기
        
        Args:
            candidates: 후보 번호 리스트
            n_numbers: 선택할 번호 수
            n_results: 반환할 조합 수
            
        Returns:
            [(조합, 점수), ...] - 점수 높은 순
        """
        if len(candidates) < n_numbers:
            return [(candidates + list(range(1, n_numbers - len(candidates) + 1)), 0.0)]
        
        # 모든 조합 생성 및 점수 계산
        results = []
        
        # 후보가 너무 많으면 상위 15개로 제한
        search_space = candidates[:min(15, len(candidates))]
        
        for combo in combinations(search_space, n_numbers):
            combo_list = list(combo)
            score = self.validator.score(combo_list)
            results.append((combo_list, score))
        
        # 점수 순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:n_results]
    
    def optimize_combination(self, 
                            initial_combo: List[int],
                            all_scores: Dict[int, float]) -> List[int]:
        """
        초기 조합을 최적화 (제약 조건 만족하도록)
        """
        current = list(initial_combo)
        is_valid, _ = self.validator.validate(current)
        
        if is_valid:
            return sorted(current)
        
        # 유효하지 않으면 개선 시도
        best_combo = current
        best_score = self.validator.score(current)
        
        # 각 번호를 교체해보며 최적화
        for i in range(6):
            for replacement in range(1, 46):
                if replacement in current:
                    continue
                
                test_combo = current.copy()
                test_combo[i] = replacement
                
                score = self.validator.score(test_combo)
                if score > best_score:
                    best_score = score
                    best_combo = test_combo.copy()
        
        return sorted(best_combo)


# 테스트
if __name__ == "__main__":
    validator = CombinationValidator()
    
    # 테스트 조합
    test_combos = [
        [1, 2, 3, 4, 5, 6],  # 연속 - 나쁜 조합
        [7, 14, 21, 28, 35, 42],  # 7의 배수 - 괜찮음
        [3, 12, 24, 33, 38, 45],  # 분산 - 좋은 조합
    ]
    
    for combo in test_combos:
        is_valid, results = validator.validate(combo)
        score = validator.score(combo)
        print(f"\n조합: {combo}")
        print(f"  유효: {is_valid}, 점수: {score:.2f}")
        print(f"  AC: {results['ac']}, 합계: {results['sum']}")
        if results['violations']:
            print(f"  위반: {results['violations']}")
