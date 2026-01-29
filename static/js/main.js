let cachedData = null;

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
        await Promise.all([fetchPredictionData(), fetchStats()]);
    } catch (error) {
        console.error('App initialization failed:', error);
    }
}

async function fetchPredictionData() {
    try {
        const response = await fetch('./data/prediction.json');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        cachedData = data;

        // Simulated AI processing time for premium feel
        setTimeout(() => {
            renderDashboard(data);
            finalizeAppLoad();
        }, 1500);

    } catch (error) {
        handleFetchError('분석 데이터를 불러오는 중 오류가 발생했습니다.', error);
    }
}

async function fetchStats() {
    try {
        const response = await fetch('./data/stats.json');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        const totalDrawsEl = getEl('total-draws');
        if (totalDrawsEl) {
            totalDrawsEl.innerText = `${data.total_draws.toLocaleString()}회`;
        }
        renderBallRow('latest-draw', data.latest_draw, 2);

        // Fetch frequency data and init chart
        const freqResponse = await fetch('./data/frequencies.json');
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

function handleFetchError(message, error) {
    console.error(error);
    alert(message);
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
                    ticks: { color: '#94a3b8', font: { size: 10 }, autoSkip: false }
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
    if (cachedData) {
        renderDashboard(cachedData, parseInt(count));
    }
}

/**
 * Main dashboard rendering entry point
 */
function renderDashboard(data, setCount = null) {
    if (setCount === null) {
        const selector = getEl('set-count');
        setCount = selector ? parseInt(selector.value) : 10;
    }

    updateRoundHeader(data.next_round);
    renderSummaryStats(data.hot_cold);
    renderPredictionSets(data.predicted_sets, setCount);
    renderEngineInsights(data.engine_predictions);
}

function updateRoundHeader(nextRound) {
    const el = getEl('round-header');
    if (el) el.innerText = `제 ${nextRound}회 예측 분석 결과`;
}

function renderSummaryStats(hotCold) {
    renderBallRow('hot-numbers', hotCold.hot.slice(0, 6).map(n => n[0]), 2.2);
    renderBallRow('cold-numbers', hotCold.cold.slice(0, 6).map(n => n[0]), 2.2);
    renderBallRow('overdue-numbers', hotCold.overdue.slice(0, 5).map(n => n[0]), 2.2);
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

    // Animate progress bars
    setTimeout(() => {
        document.querySelectorAll('.confidence-fill').forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    }, 100);
}

function createPredictionCard(set, index) {
    const card = document.createElement('div');
    card.className = 'prediction-card';
    card.style.animation = `fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) ${index * 0.05}s both`;

    card.innerHTML = `
        <div class="confidence-label">
            <span>앙상블 세트 ${index + 1}</span>
            <span style="color: #c084fc; font-weight: 800;">신뢰도 ${set.confidence.toFixed(1)}%</span>
        </div>
        <div class="numbers-row">
            ${set.numbers.map(n => `<div class="number-ball ${getNumberColorClass(n)}">${n}</div>`).join('')}
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: 0%;" data-width="${set.confidence}%"></div>
        </div>
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
        renderBallRow(`engine-balls-${key}`, nums, 2.0);
    });
}

function renderBallRow(containerId, numbers, size) {
    const container = getEl(containerId);
    if (!container) return;

    container.innerHTML = numbers.map(num => `
        <div class="number-ball ${getNumberColorClass(num)}" 
             style="width: ${size}rem; height: ${size}rem; font-size: ${size * 0.4}rem">
            ${num}
        </div>
    `).join('');
}

function getNumberColorClass(num) {
    if (num <= 10) return 'num-1-10';
    if (num <= 20) return 'num-11-20';
    if (num <= 30) return 'num-21-30';
    if (num <= 40) return 'num-31-40';
    return 'num-41-45';
}
