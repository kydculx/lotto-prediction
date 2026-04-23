let cachedData = null;
const state = {
    allPredictions: [],
    allResults: [], // 역대 당첨 번호 저장
    currentRound: null,
    isHistorical: false,
    winningNumbers: null
};

// DOM Utility: Get element with error check
const getEl = (id) => document.getElementById(id);

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
            // Simulated AI processing time for premium feel
            setTimeout(() => {
                renderDashboard(predictionData);
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
            throw new Error(targetRound
                ? `${targetRound}회차 데이터가 없습니다.`
                : '최신 분석 데이터를 불러올 수 없습니다.');
        }
        return await response.json();
    } catch (error) {
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

        // Populate Round Selector
        const selector = getEl('round-selector');
        if (selector) {
            selector.innerHTML = '<option value="" disabled selected>회차 선택</option>';
            for (let i = data.total_draws; i >= 1; i--) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `${i}회`;
                selector.appendChild(option);
            }
        }

        renderBallRow('latest-draw', data.latest_draw);

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

    const labels = Object.keys(freqData).sort((a, b) => a - b);
    const counts = labels.map(label => freqData[label]);

    const backgroundColors = labels.map(num => getChartColor(parseInt(num), 0.6));
    const borderColors = labels.map(num => getChartColor(parseInt(num), 1));

    new Chart(ctx, {
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
                        callback: function(value, index, values) {
                            // 모바일에서는 5의 배수만 표시하여 혼잡도 감소
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
function renderDashboard(data, setCount = null) {
    state.allPredictions = data.predicted_sets;
    state.currentRound = data.next_round;
    
    // 당첨번호 조회 로직 강화
    if (data.actual_winning_numbers) {
        state.winningNumbers = data.actual_winning_numbers;
    } else {
        // lotto_results.json에서 해당 회차의 당첨번호 찾기
        const historicalResult = state.allResults.find(r => r.round === data.next_round);
        state.winningNumbers = historicalResult ? historicalResult.numbers : null;
    }

    if (setCount === null) {
        const selector = getEl('set-count');
        setCount = selector ? parseInt(selector.value) : 10;
    }

    updateRoundHeader(data.next_round);
    renderSummaryStats(data.hot_cold);
    renderPredictionSets(state.allPredictions, setCount);
    renderEngineInsights(data.engine_predictions);
    renderDynamicWeights(data.final_weights, data.dynamic_boosts);
    
    // 상세 적중 대조 렌더링
    renderMatchAnalysis();
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

window.searchRound = async () => {
    const selector = getEl('round-selector');
    const roundNum = parseInt(selector.value);

    if (!roundNum) return; // Do nothing if default selected

    getEl('loader').style.opacity = '1';
    getEl('loader').style.visibility = 'visible';

    state.isHistorical = true;
    const data = await fetchPredictionData(roundNum);

    if (data) {
        renderDashboard(data);
    } else {
        state.isHistorical = false; // 실패 시 상태 복구
    }

    setTimeout(() => {
        getEl('loader').style.opacity = '0';
        setTimeout(() => getEl('loader').style.visibility = 'hidden', 800);
    }, 500);
};

/**
 * 🔍 상세 적중 대조 렌더링 함수
 */
function renderMatchAnalysis() {
    const section = getEl('match-analysis-section');
    const winningDisplay = getEl('winning-numbers-display');
    const tableBody = getEl('match-analysis-body');
    const filterSelect = getEl('min-hit-filter');

    if (!section || !tableBody) return;

    // 당첨번호가 있으면 대조 화면 표시
    if (!state.winningNumbers || state.winningNumbers.length === 0) {
        section.style.display = 'none';
        return;
    }
    section.style.display = 'block';

    const winningNums = state.winningNumbers;
    const minHit = filterSelect ? parseInt(filterSelect.value) : 2;

    // 2. 당첨 번호 상단 배너 표시
    winningDisplay.innerHTML = `
        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem; font-weight: 600;">
            ${state.currentRound}회 공식 당첨 번호
        </div>
        <div class="numbers-row" style="justify-content: center;">
            ${winningNums.map(n => createBall(n)).join('')}
        </div>
    `;

    // 3. 적중 조합 필터링 및 렌더링
    tableBody.innerHTML = '';
    
    // 순위 계산 (상위 스코프의 getRank 사용)
    const matchedSets = state.allPredictions
        .map(set => ({
            set: set.numbers,
            hitCount: set.numbers.filter(n => winningNums.includes(n)).length
        }))
        .filter(item => item.hitCount >= minHit)
        .sort((a, b) => b.hitCount - a.hitCount);

    if (matchedSets.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="2" style="padding: 3rem; color: #64748b;">선택한 조건(${minHit}개 이상)에 맞는 적중 조합이 없습니다.</td></tr>`;
        return;
    }

    matchedSets.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${getRank(item.hitCount)}</td>
            <td>
                <div class="match-row-numbers">
                    ${item.set.map(n => {
                        const isMatched = winningNums.includes(n);
                        return createBall(n, isMatched ? 'matched-num' : '');
                    }).join('')}
                </div>
            </td>
        `;
        tableBody.appendChild(tr);
    });
}

window.resetToLatest = async () => {
    getEl('loader').style.opacity = '1';
    getEl('loader').style.visibility = 'visible';

    state.isHistorical = false;
    const selector = getEl('round-selector');
    if (selector) selector.value = ""; // Reset dropdown

    const data = await fetchPredictionData();
    if (data) renderDashboard(data);

    setTimeout(() => {
        getEl('loader').style.opacity = '0';
        setTimeout(() => getEl('loader').style.visibility = 'hidden', 800);
    }, 500);
};

function updateRoundHeader(nextRound) {
    const header = getEl('round-header');
    const tag = document.querySelector('.section-tag');

    if (state.isHistorical) {
        header.textContent = `${nextRound}회차 과거 분석 결과`;
        if (tag) tag.textContent = 'HISTORICAL ANALYSIS';
        const resetButton = getEl('reset-button');
        if (resetButton) resetButton.style.display = 'block';
    } else {
        header.textContent = `${nextRound}회차 예상 분석 결과`;
        if (tag) tag.textContent = 'REAL-TIME ANALYSIS';
        const resetButton = getEl('reset-button');
        if (resetButton) resetButton.style.display = 'none';
    }
}

function renderSummaryStats(hotCold) {
    renderBallRow('hot-numbers', hotCold.hot.slice(0, 6).map(n => n[0]));
    renderBallRow('cold-numbers', hotCold.cold.slice(0, 6).map(n => n[0]));
    renderBallRow('overdue-numbers', hotCold.overdue.slice(0, 5).map(n => n[0]));
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

function getRank(hitCount) {
    switch (hitCount) {
        case 6: return '<span class="rank-badge rank-1">1등</span>';
        case 5: return '<span class="rank-badge rank-2">2등</span>';
        case 4: return '<span class="rank-badge rank-3">3등</span>';
        case 3: return '<span class="rank-badge rank-4">4등</span>';
        case 2: return '<span class="rank-badge rank-5">5등</span>';
        default: return `<span style="color: #64748b; font-size: 0.8rem;">${hitCount}개</span>`;
    }
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
function createBall(num, extraClass = '') {
    return `<div class="number-ball ${getNumberColorClass(num)} ${extraClass}">${num}</div>`;
}
