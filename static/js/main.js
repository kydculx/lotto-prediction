let cachedData = null;
const state = {
    allPredictions: [],
    allResults: [], // 역대 당첨 번호 저장
    currentRound: null,
    latestPrediction: null, // 최신(예상) 데이터 별도 저장
    isHistorical: false,
    winningNumbers: null,
    chart: null, // Frequency Chart
    matchChart: null, // Match Analysis Chart
    searchTimeout: null // Real-time search debounce
};

// DOM Utility: Get element with error check
const getEl = (id) => document.getElementById(id);

/**
 * 🎚️ 슬라이더 조작 시 실시간 인터랙션 (전역 스코프 보장)
 */
window.handleSliderInput = function(value) {
    const display = getEl('round-display');
    if (display) display.innerText = `${value}회`;
    
    const slider = getEl('round-selector');
    if (slider) {
        const min = parseInt(slider.min) || 1;
        const max = parseInt(slider.max) || 1200;
        const percent = ((value - min) / (max - min)) * 100;
        slider.style.setProperty('--p', `${percent}%`);
    }
};

document.addEventListener('DOMContentLoaded', () => {
    // Prevent auto-scroll and lock to top
    if (window.location.hash) {
        history.replaceState(null, null, window.location.pathname);
    }
    window.scrollTo(0, 0);

    initApp();
});

async function initApp() {
    try {
        const [predictionData, resultsData] = await Promise.all([
            fetchPredictionData(),
            fetchLottoResults(),
            fetchStats()
        ]);

        if (resultsData) state.allResults = resultsData;
        if (predictionData) {
            state.latestPrediction = predictionData; // 최신 데이터 저장
            // Simulated AI processing time for premium feel
            setTimeout(async () => {
                renderDashboard(predictionData);

                // 하단 섹션에 최신 과거 회차 데이터 기본 로드
                const selector = getEl('round-selector');
                if (selector) {
                    const maxRound = state.allResults.length > 0 ? state.allResults[state.allResults.length - 1].round : 1221;
                    selector.max = maxRound;
                    selector.value = maxRound;
                    if (window.handleSliderInput) window.handleSliderInput(maxRound);
                    await window.searchRound(false); // 초기 로드 시에는 스크롤하지 않음
                }

                finalizeAppLoad();
            }, 1500);
        } else {
            finalizeAppLoad(); // Hide loader even if prediction data fails
        }
    } catch (error) {
        console.error('App initialization failed:', error);
        finalizeAppLoad(); // Ensure loader is hidden on app init failure
    }
}

async function fetchPredictionData(targetRound = null) {
    try {
        const timestamp = new Date().getTime();
        const path = targetRound
            ? `./data/history/prediction_${targetRound}.json?t=${timestamp}`
            : `./data/prediction.json?t=${timestamp}`;

        const response = await fetch(path);
        if (!response.ok) {
            // 과거 데이터 조회 시 파일이 없어도 콘솔 경고 없이 null 반환 (handleMissingData에서 처리)
            return null;
        }
        return await response.json();
    } catch (error) {
        if (targetRound) return null; // 과거 데이터 조회 시 에러는 null 반환
        handleFetchError(error, '분석 데이터');
        return null;
    }
}

async function fetchLottoResults() {
    try {
        const response = await fetch('./data/lotto_results.json');
        return await response.json();
    } catch (e) {
        console.error('Failed to load lotto results:', e);
        return [];
    }
}

async function fetchStats() {
    try {
        const timestamp = new Date().getTime();
        const response = await fetch(`./data/stats.json?t=${timestamp}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        const totalDrawsEl = getEl('total-draws');
        if (totalDrawsEl) {
            totalDrawsEl.innerText = `${data.total_draws.toLocaleString()}회`;
        }

        // Populate Round Slider
        const slider = getEl('round-selector');
        if (slider) {
            slider.min = 2;
            slider.max = data.total_draws;
            slider.value = data.total_draws;
            
            const maxLabel = getEl('max-round-label');
            if (maxLabel) maxLabel.innerText = `${data.total_draws}회`;
            
            // 초기 슬라이더 상태 업데이트
            if (window.handleSliderInput) {
                window.handleSliderInput(data.total_draws);
            }
        }

        renderBallRow('latest-draw-container', data.latest_draw);

        // Fetch frequency data and init chart
        const freqResponse = await fetch(`./data/frequencies.json?t=${timestamp}`);
        if (freqResponse.ok) {
            const freqData = await freqResponse.json();
            initFrequencyChart(freqData);
        }
    } catch (error) {
        console.error('Stats fetch error:', error);
    }
}

function finalizeAppLoad() {
    window.scrollTo({ top: 0, behavior: 'instant' });
    const loader = getEl('loader');
    if (loader) {
        loader.style.opacity = '0';
        setTimeout(() => { loader.style.display = 'none'; }, 500);
    }
}

function handleFetchError(error, contextMessage) {
    console.error(error);
    alert(`${contextMessage}를 불러오는 중 오류가 발생했습니다: ${error.message}`);
    const loader = getEl('loader');
    if (loader) {
        loader.style.opacity = '0';
        setTimeout(() => { loader.style.display = 'none'; }, 500);
    }
}

function initFrequencyChart(freqData) {
    const ctx = getEl('frequencyChart');
    if (!ctx) return;

    // 기존 차트 파괴 (중복 생성 방지)
    if (state.chart) {
        state.chart.destroy();
    }

    const labels = Object.keys(freqData).sort((a, b) => a - b);
    const counts = labels.map(label => freqData[label]);

    const backgroundColors = labels.map(num => getChartColor(parseInt(num), 0.6));
    const borderColors = labels.map(num => getChartColor(parseInt(num), 1));

    state.chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: '출현 빈도',
                data: counts,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        title: (items) => `${items[0].label}번`,
                        label: (item) => `총 ${item.raw}회 출현`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#94a3b8',
                        font: { size: window.innerWidth < 400 ? 7 : 9 },
                        autoSkip: true,
                        maxTicksLimit: window.innerWidth < 600 ? 8 : 20,
                        maxRotation: 0,
                        minRotation: 0,
                        callback: function (value, index, values) {
                            if (window.innerWidth < 600) {
                                return (value + 1) % 5 === 0 ? (value + 1) : '';
                            }
                            return value + 1;
                        }
                    }
                }
            }
        }
    });
}

/**
 * 📊 차트 데이터 업데이트 함수
 */
function updateChart(freqData) {
    if (!state.chart || !freqData) return;

    const labels = Object.keys(freqData).sort((a, b) => a - b);
    const counts = labels.map(label => freqData[label]);

    state.chart.data.labels = labels;
    state.chart.data.datasets[0].data = counts;

    // 색상도 업데이트 (데이터 범위가 달라질 수 있음)
    state.chart.data.datasets[0].backgroundColor = labels.map(num => getChartColor(parseInt(num), 0.6));
    state.chart.data.datasets[0].borderColor = labels.map(num => getChartColor(parseInt(num), 1));

    state.chart.update('none'); // 애니메이션 없이 업데이트
}

function getChartColor(n, alpha) {
    if (n <= 10) return `rgba(234, 179, 8, ${alpha})`;
    if (n <= 20) return `rgba(59, 130, 246, ${alpha})`;
    if (n <= 30) return `rgba(239, 68, 68, ${alpha})`;
    if (n <= 40) return `rgba(107, 114, 128, ${alpha})`;
    return `rgba(34, 197, 94, ${alpha})`;
}

// Global update handler
window.updateSetCount = function (count) {
    if (state.allPredictions.length > 0) {
        renderPredictionSets(state.allPredictions, parseInt(count));
    }
}

/**
 * Main dashboard rendering entry point
 */
function renderDashboard(data) {
    if (!data) return;

    // 1. 상단 예상 섹션 렌더링 (항상 최신 데이터 사용)
    const predData = state.latestPrediction || data;
    renderPredictionSection(predData);

    // 2. 하단 분석 섹션 렌더링 (검색된 데이터 사용)
    renderHistorySection(data);
}

/**
 * 🔮 상단 예상 섹션 전용 렌더링
 */
function renderPredictionSection(data) {
    const roundHeader = getEl('round-header');
    if (roundHeader) {
        roundHeader.innerText = `${data.next_round}회차 예측 분석`;
    }

    // 핫/콜드/오버듀 넘버
    renderBallRow('hot-numbers', data.hot_cold.hot.slice(0, 6).map(n => n[0]));
    renderBallRow('cold-numbers', data.hot_cold.cold.slice(0, 6).map(n => n[0]));
    renderBallRow('overdue-numbers', data.hot_cold.overdue.slice(0, 5).map(n => n[0]));

    // 차트 업데이트
    updateChart(data.hot_cold.frequencies);

    // 앙상블 추천 조합
    state.allPredictions = data.predicted_sets;
    const selector = getEl('set-count');
    const setCount = selector ? parseInt(selector.value) : 10;
    renderPredictionSets(data.predicted_sets, setCount);

    // 최근 추첨 결과 요약 (시스템 인사이트)
    const latestDrawContainer = getEl('latest-draw-container');
    if (latestDrawContainer && state.allResults.length > 0) {
        const last = state.allResults[0];
        latestDrawContainer.innerHTML = `
            <div style="color: var(--accent-color); margin-bottom: 0.5rem; font-size: 0.8rem; font-weight: 700;">
                ${last.round}회 당첨
            </div>
            <div class="numbers-row" style="justify-content: center; gap: 0.35rem !important;">
                ${last.numbers.map(n => createBall(n)).join('')}
                <div class="number-ball ball-bonus" style="margin-left: 2px;">${last.bonus}</div>
            </div>
        `;
    }

}

/**
 * 📜 하단 과거 분석 섹션 전용 렌더링
 */
function renderHistorySection(data) {

    // 당첨번호 조회 로직
    if (data.actual_winning_numbers) {
        state.winningNumbers = data.actual_winning_numbers;
    } else {
        const historicalResult = state.allResults.find(r => r.round === data.next_round);
        state.winningNumbers = historicalResult ? historicalResult.numbers : null;
        state.bonusNumber = historicalResult ? historicalResult.bonus : null;
    }

    const winningDisplay = getEl('winning-numbers-display');
    if (state.winningNumbers) {
        if (winningDisplay) {
            winningDisplay.innerHTML = `
                <div style="font-weight: 700; margin-bottom: 1rem; font-size: 1.1rem; color: var(--accent-color);">
                    🎯 ${data.next_round}회차 실제 당첨 번호
                </div>
                <div class="numbers-row" style="justify-content: center;">
                    ${state.winningNumbers.map(n => createBall(n)).join('')}
                </div>
            `;
            winningDisplay.style.display = 'block';
        }
    } else {
        if (winningDisplay) {
            winningDisplay.innerHTML = `<div style="color: var(--text-secondary); padding: 1rem;">해당 회차의 당첨 결과가 데이터베이스에 없습니다.</div>`;
        }
    }

    // 적중 대조 테이블 렌더링
    state.historyPredictions = data.predicted_sets;
    
    // 차트 영역 복구
    const chartContainer = getEl('match-summary-container');
    if (chartContainer) chartContainer.style.display = 'block';
    
    renderMatchAnalysis(data.predicted_sets);
}

function renderDynamicWeights(weights, boosts) {
    const container = getEl('dynamic-weights-container');
    if (!container || !weights) return;

    container.innerHTML = '';

    // 가중치 높은 순으로 정렬
    const sortedEngines = Object.keys(weights).sort((a, b) => weights[b] - weights[a]);

    sortedEngines.forEach((name, idx) => {
        const weight = (weights[name] * 100).toFixed(1);
        const boost = boosts[name] || 1.0;
        const boostPct = ((boost - 1.0) * 100).toFixed(0);

        const card = document.createElement('div');
        card.style.cssText = `
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            padding: 0.75rem;
            border-radius: 0.75rem;
            text-align: center;
            animation: fadeIn 0.5s ease-out ${0.1 * idx}s both;
        `;

        card.innerHTML = `
            <div style="font-size: 0.7rem; color: #94a3b8; margin-bottom: 0.25rem;">${name.toUpperCase()}</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">${weight}%</div>
            ${boost > 1.001 ? `<div style="font-size: 0.6rem; color: #34d399; margin-top: 0.25rem;">+${boostPct}% 🔥 Boost</div>` : ''}
        `;
        container.appendChild(card);
    });
}

// --- Global Actions ---

window.updateSetCount = (count) => {
    renderPredictionSets(state.allPredictions, parseInt(count));
};

window.searchRound = async (shouldScroll = true) => {
    const selector = getEl('round-selector');
    const roundNum = parseInt(selector.value);

    if (!roundNum) return;

    // 디바운스 처리 (실시간 드래그 시 부하 감소)
    if (state.searchTimeout) clearTimeout(state.searchTimeout);
    
    state.searchTimeout = setTimeout(async () => {
        // 로더 표시 (실시간일 때는 아주 살짝만 혹은 생략 가능하지만, 일단 유지)
        const loader = getEl('loader');
        if (shouldScroll && loader) {
            loader.style.opacity = '1';
            loader.style.visibility = 'visible';
        }

        state.isHistorical = true;
        const data = await fetchPredictionData(roundNum);

        if (data) {
            renderHistorySection(data);
            if (shouldScroll) {
                getEl('history-section').scrollIntoView({ behavior: 'smooth' });
            }
        } else {
            // 데이터 부재 처리 (생략: 이전 구현 유지)
            handleMissingData(roundNum);
        }

        if (shouldScroll && loader) {
            setTimeout(() => {
                loader.style.opacity = '0';
                setTimeout(() => loader.style.visibility = 'hidden', 800);
            }, 500);
        }
    }, shouldScroll ? 0 : 50); // 드래그 중일 때는 50ms 대기, 클릭 시에는 즉시 실행
};

/**
 * ❌ 데이터 부재 시 UI 처리 분리
 */
function handleMissingData(roundNum) {
    const listContainer = getEl('match-results-list');
    const chartContainer = getEl('match-summary-container');
    const winningDisplay = getEl('winning-numbers-display');
    
    if (winningDisplay) winningDisplay.innerHTML = '';
    if (listContainer) {
        listContainer.innerHTML = `
            <div style="padding: 3rem; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem;">
                    데이터를 찾을 수 없습니다
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem;">
                    ${roundNum}회차의 상세 분석 데이터가 아카이브에 존재하지 않습니다.
                </div>
            </div>
        `;
    }
    if (state.matchChart) {
        state.matchChart.destroy();
        state.matchChart = null;
    }
    if (chartContainer) chartContainer.style.display = 'none';
}

/**
 * 🔍 상세 적중 대조 렌더링 함수
 */
function renderMatchAnalysis(sets = null) {
    const listContainer = getEl('match-results-list');
    if (!listContainer) return;

    const targetSets = sets || state.historyPredictions || state.allPredictions || [];

    if (!state.winningNumbers || state.winningNumbers.length === 0) {
        listContainer.innerHTML = `<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">당첨 정보가 없습니다.</div>`;
        return;
    }

    const winningNums = state.winningNumbers;
    const stats = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
    
    const processedSets = targetSets.map(set => {
        const hitCount = set.numbers.filter(n => winningNums.includes(n)).length;
        if (hitCount <= 6) stats[hitCount]++;
        return { set: set.numbers, hitCount: hitCount };
    });

    renderMatchAnalysisChart(stats);

    const matchedSets = processedSets
        .filter(item => item.hitCount >= 3)
        .sort((a, b) => b.hitCount - a.hitCount);

    listContainer.innerHTML = '';

    if (matchedSets.length === 0) {
        listContainer.innerHTML = `<div style="padding: 3rem; text-align: center; color: var(--text-secondary);">3개 이상 적중된 내역이 없습니다.</div>`;
        return;
    }

    matchedSets.forEach(item => {
        const row = document.createElement('div');
        row.className = 'match-result-row';
        
        const numbersHtml = item.set.map(n => {
            const isMatched = winningNums.includes(n);
            const ballColorClass = getNumberColorClass(n);
            return `<span class="history-ball ${ballColorClass} ${isMatched ? 'hit' : 'miss'}">${n}</span>`;
        }).join('');

        row.innerHTML = `
            <div class="match-numbers-balls">${numbersHtml}</div>
        `;
        listContainer.appendChild(row);
    });
}




function renderPredictionSets(sets, count) {
    const container = getEl('prediction-sets');
    if (!container) return;

    container.innerHTML = '';
    const displaySets = sets.slice(0, count);

    displaySets.forEach((set, index) => {
        const card = createPredictionCard(set, index);
        container.appendChild(card);
    });

}

function createPredictionCard(set, index) {
    const card = document.createElement('div');
    card.className = 'prediction-card';
    card.style.animation = `fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) ${index * 0.05}s both`;

    card.innerHTML = `
        <div class="set-index">${index + 1}</div>
        <div class="numbers-row">
            ${set.numbers.map(n => `<div class="number-ball ${getNumberColorClass(n)}">${n}</div>`).join('')}
        </div>
        <div class="set-confidence">${set.confidence.toFixed(1)}%</div>
    `;
    return card;
}

function renderEngineInsights(predictions) {
    const container = getEl('engine-grid');
    if (!container) return;

    container.innerHTML = '';
    const labels = {
        'statistical': '통계 분석',
        'pattern': '패턴 분석',
        'timeseries': '시계열 분석',
        'lstm': 'LSTM 딥러닝',
        'graph': '그래프 이론',
        'numerology': '수학적 분석',
        'ml': '머신러닝',
        'gap': '간격 분석',
        'poisson': '포아송 확률 분석',
        'fourier': '푸리에 주파수 분석',
        'advanced_pattern': '심화 패턴 분석',
        'advancedpattern': '심화 패턴 분석',
        'sequence_correlation': '수열 상관관계 분석',
        'sequencecorrelation': '수열 상관관계 분석'
    };

    Object.entries(predictions).forEach(([key, nums], idx) => {
        const cleanKey = key.toLowerCase().replace(/_/g, '');
        const div = document.createElement('div');
        div.className = 'engine-card';
        div.style.cssText = `
            padding: 1.25rem;
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            animation: fadeInUp 0.6s ease-out ${0.5 + idx * 0.05}s backwards;
        `;
        div.innerHTML = `
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; text-align: center;">
                ${labels[cleanKey] || key}
            </div>
            <div id="engine-balls-${key}" class="numbers-row" style="justify-content: center;"></div>
        `;
        container.appendChild(div);
        renderBallRow(`engine-balls-${key}`, nums);
    });
}

function renderBallRow(containerId, numbers, size) {
    const container = getEl(containerId);
    if (!container) return;

    container.innerHTML = numbers.map(num => `
        <div class="number-ball ${getNumberColorClass(num)}">
            ${num}
        </div>
    `).join('');
}

function getRank(hitCount, isSecondRank = false) {
    if (hitCount === 6) return '<span class="rank-badge rank-1">1등</span>';
    if (hitCount === 5) {
        if (isSecondRank) return '<span class="rank-badge rank-2">2등</span>';
        return '<span class="rank-badge rank-3">3등</span>';
    }
    if (hitCount === 4) return '<span class="rank-badge rank-4">4등</span>';
    if (hitCount === 3) return '<span class="rank-badge rank-5">5등</span>';

    return `<span style="color: #64748b; font-size: 0.8rem;">${hitCount}개</span>`;
}

function getNumberColorClass(num) {
    if (num <= 10) return 'num-1-10';
    if (num <= 20) return 'num-11-20';
    if (num <= 30) return 'num-21-30';
    if (num <= 40) return 'num-31-40';
    return 'num-41-45';
}

/**
 * ⚽ 번호 볼 HTML 생성 헬퍼
 */
/**
 * ⚽ 번호 볼 HTML 생성 헬퍼
 */
function createBall(num, extraClass = '') {
    return `<div class="number-ball ${getNumberColorClass(num)} ${extraClass}">${num}</div>`;
}

// Tab System Removed
/**
 * 📊 적중 분석 차트 생성/업데이트
 */
function renderMatchAnalysisChart(stats) {
    const ctx = getEl('matchAnalysisChart');
    if (!ctx) return;

    if (state.matchChart) {
        state.matchChart.destroy();
    }

    const data = Object.values(stats);
    const labels = ['0개', '1개', '2개', '3개', '4개', '5개', '6개'];

    state.matchChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '적중 횟수',
                data: data,
                backgroundColor: labels.map((_, i) => i >= 3 ? 'rgba(56, 189, 248, 0.6)' : 'rgba(100, 116, 139, 0.3)'),
                borderColor: labels.map((_, i) => i >= 3 ? '#38bdf8' : '#64748b'),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#38bdf8',
                    bodyColor: '#fff',
                    borderColor: '#1e293b',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { 
                        color: '#94a3b8',
                        precision: 0
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}
