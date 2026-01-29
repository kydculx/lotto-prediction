"""
패턴 분석 엔진
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from .base import BaseEngine


class PatternEngine(BaseEngine):
    """패턴 분석 엔진"""
    
    def analyze_consecutive(self) -> Dict:
        """연속번호 패턴 분석"""
        consecutive_counts = []
        for row in self.numbers_matrix:
            sorted_row = sorted(row)
            count = sum(1 for i in range(len(sorted_row)-1) if sorted_row[i+1] - sorted_row[i] == 1)
            consecutive_counts.append(count)
        
        counter = Counter(consecutive_counts)
        return {
            'consecutive_count': dict(counter),
            'avg_consecutive': np.mean(consecutive_counts),
            'probability': sum(1 for c in consecutive_counts if c > 0) / len(consecutive_counts)
        }
    
    def analyze_odd_even(self) -> Dict:
        """홀짝 비율 분석"""
        distributions = []
        for row in self.numbers_matrix:
            odd_count = sum(1 for n in row if n % 2 == 1)
            distributions.append((odd_count, 6 - odd_count))
        
        counter = Counter(distributions)
        optimal = counter.most_common(1)[0][0]
        recent_odd = [sum(1 for n in row if n % 2 == 1) for row in self.numbers_matrix[-50:]]
        
        return {
            'distribution': dict(counter),
            'optimal_ratio': optimal,
            'recent_trend': np.mean(recent_odd)
        }
    
    def analyze_high_low(self) -> Dict:
        """고저 비율 분석"""
        distributions = []
        for row in self.numbers_matrix:
            low_count = sum(1 for n in row if n <= 22)
            distributions.append((low_count, 6 - low_count))
        
        counter = Counter(distributions)
        optimal = counter.most_common(1)[0][0]
        recent_low = [sum(1 for n in row if n <= 22) for row in self.numbers_matrix[-50:]]
        
        return {
            'distribution': dict(counter),
            'optimal_ratio': optimal,
            'recent_trend': np.mean(recent_low)
        }
    
    def analyze_ending_digit(self) -> Dict[int, float]:
        """끝수 분포 분석"""
        all_numbers = self.numbers_matrix.flatten()
        endings = [n % 10 for n in all_numbers]
        counter = Counter(endings)
        total = len(endings)
        return {i: counter.get(i, 0) / total for i in range(10)}
    
    def analyze_sections(self) -> Dict:
        """구간별 분포 분석"""
        section_counts = {1: [], 2: [], 3: [], 4: [], 5: []}
        for row in self.numbers_matrix:
            counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for n in row:
                if n <= 10: counts[1] += 1
                elif n <= 20: counts[2] += 1
                elif n <= 30: counts[3] += 1
                elif n <= 40: counts[4] += 1
                else: counts[5] += 1
            for sec, cnt in counts.items():
                section_counts[sec].append(cnt)
        
        return {
            'avg_per_section': {sec: np.mean(counts) for sec, counts in section_counts.items()},
            'optimal_distribution': {sec: round(np.mean(counts)) for sec, counts in section_counts.items()}
        }
    
    def analyze_sum_range(self) -> Dict:
        """합계 범위 분석"""
        sums = [sum(row) for row in self.numbers_matrix]
        return {
            'min': min(sums),
            'max': max(sums),
            'mean': np.mean(sums),
            'std': np.std(sums),
            'optimal_range': (int(np.mean(sums) - np.std(sums)), int(np.mean(sums) + np.std(sums)))
        }
    
    def get_scores(self) -> Dict[int, float]:
        """패턴 기반 점수 계산"""
        scores = {i: 0.0 for i in range(1, 45 + 1)}
        
        ending_dist = self.analyze_ending_digit()
        for num in range(1, 46):
            scores[num] += ending_dist[num % 10] * 0.3
            
        section_dist = self.analyze_sections()['avg_per_section']
        for num in range(1, 46):
            if num <= 10: sec_score = section_dist[1] / 6
            elif num <= 20: sec_score = section_dist[2] / 6
            elif num <= 30: sec_score = section_dist[3] / 6
            elif num <= 40: sec_score = section_dist[4] / 6
            else: sec_score = section_dist[5] / 6
            scores[num] += sec_score * 0.4
            
        odd_even = self.analyze_odd_even()
        optimal_odd = odd_even['optimal_ratio'][0]
        for num in range(1, 46):
            if optimal_odd >= 3: scores[num] += 0.3 if num % 2 == 1 else 0.15
            else: scores[num] += 0.15 if num % 2 == 1 else 0.3
            
        max_score = max(scores.values())
        return {k: v / max_score for k, v in scores.items()} if max_score > 0 else scores
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        """패턴 기반 예측"""
        scores = self.get_scores()
        section_opt = self.analyze_sections()['optimal_distribution']
        selected = []
        for sec, target_count in section_opt.items():
            nums = range((sec-1)*10+1, min(sec*10+1, 46))
            sec_scores = sorted([(n, scores[n]) for n in nums], key=lambda x: x[1], reverse=True)
            selected.extend([n for n, _ in sec_scores[:target_count]])
        
        if len(selected) < n_numbers:
            remaining = sorted([n for n in range(1, 46) if n not in selected], key=lambda n: scores[n], reverse=True)
            selected.extend(remaining[:n_numbers - len(selected)])
        return sorted(selected[:n_numbers])
