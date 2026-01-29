let cachedData = null; // 데이터를 전역 캐시에 저장

document.addEventListener('DOMContentLoaded', () => {
    // 자동 스크롤 방지: URL에 #dashboard 등이 있어도 무시하고 최상단 고정
    if (window.location.hash) {
        history.replaceState(null, null, window.location.pathname);
    }
    window.scrollTo(0, 0);

    fetchPredictionData();
    fetchStats();
});

async function fetchPredictionData() {
    try {
        const response = await fetch('./data/prediction.json');
        const data = await response.json();

        if (data.error) {
            alert('분석 오류: ' + data.error);
            return;
        }

        cachedData = data; // 데이터 저장

        setTimeout(() => {
            renderDashboard(data);
            // 렌더링 후 레이아웃 변화로 인해 스크롤이 이동하지 않도록 다시 한 번 고정
            window.scrollTo({ top: 0, behavior: 'instant' });
            document.getElementById('loader').style.opacity = '0';
            setTimeout(() => {
                document.getElementById('loader').style.display = 'none';
            }, 500);
        }, 1500);

    } catch (error) {
        console.error('Network error:', error);
        // 오류 발생 시에도 최소한의 화면은 보여줌
        setTimeout(() => {
            document.getElementById('loader').style.opacity = '0';
            setTimeout(() => {
                document.getElementById('loader').style.display = 'none';
            }, 500);
        }, 3000);
    }
}

async function fetchStats() {
    try {
        const response = await fetch('./data/stats.json');
        const data = await response.json();
        const totalDrawsEl = document.getElementById('total-draws');
        if (totalDrawsEl) {
            totalDrawsEl.innerText = `${data.total_draws.toLocaleString()}회`;
        }
        renderBallRow('latest-draw', data.latest_draw, 2);

        // 빈도수 데이터 호출 및 차트 초기화 (정적 JSON 파일 사용)
        const freqResponse = await fetch('./data/frequencies.json');
        const freqData = await freqResponse.json();
        initFrequencyChart(freqData);
    } catch (error) {
        console.error('Stats error:', error);
    }
}

function initFrequencyChart(freqData) {
    const ctx = document.getElementById('frequencyChart');
    if (!ctx) return;

    const labels = Object.keys(freqData).sort((a, b) => a - b);
    const counts = labels.map(label => freqData[label]);

    // 배경색 로직 (볼 색상 계열에 맞춰 그라데이션)
    const backgroundColors = labels.map(num => {
        const n = parseInt(num);
        if (n <= 10) return 'rgba(234, 179, 8, 0.6)'; // yellow
        if (n <= 20) return 'rgba(59, 130, 246, 0.6)'; // blue
        if (n <= 30) return 'rgba(239, 68, 68, 0.6)'; // red
        if (n <= 40) return 'rgba(107, 114, 128, 0.6)'; // gray
        return 'rgba(34, 197, 94, 0.6)'; // green
    });

    const borderColors = labels.map(num => {
        const n = parseInt(num);
        if (n <= 10) return 'rgba(234, 179, 8, 1)';
        if (n <= 20) return 'rgba(59, 130, 246, 1)';
        if (n <= 30) return 'rgba(239, 68, 68, 1)';
        if (n <= 40) return 'rgba(107, 114, 128, 1)';
        return 'rgba(34, 197, 94, 1)';
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
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
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 10 }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 10 },
                        autoSkip: false
                    }
                }
            }
        }
    });
}

// 개수 변경 시 호출되는 함수
window.updateSetCount = function (count) {
    if (cachedData) {
        renderDashboard(cachedData, parseInt(count));
    }
}

function renderDashboard(data, setCount = null) {
    // 선택된 개수가 없으면 현재 셀렉트 박스의 값을 사용
    if (setCount === null) {
        const selector = document.getElementById('set-count');
        setCount = selector ? parseInt(selector.value) : 10;
    }
    const roundHeader = document.getElementById('round-header');
    if (roundHeader) {
        roundHeader.innerText = `제 ${data.next_round}회 예측 분석 결과`;
    }

    renderBallRow('hot-numbers', data.hot_cold.hot.slice(0, 6).map(n => n[0]), 2.2);
    renderBallRow('cold-numbers', data.hot_cold.cold.slice(0, 6).map(n => n[0]), 2.2);

    const overdueContainer = document.getElementById('overdue-numbers');
    if (overdueContainer) {
        overdueContainer.innerHTML = '';
        renderBallRow('overdue-numbers', data.hot_cold.overdue.slice(0, 5).map(n => n[0]), 2.2);
    }

    const setsContainer = document.getElementById('prediction-sets');
    if (setsContainer) {
        setsContainer.innerHTML = '';
        // 선택된 개수만큼 슬라이싱
        const displaySets = data.predicted_sets.slice(0, setCount);

        displaySets.forEach((set, index) => {
            const card = document.createElement('div');
            card.className = 'prediction-card';
            // forwards 대신 both를 사용하여 지연 시간 중에도, 종료 후에도 opacity를 유지하도록 설정
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
            setsContainer.appendChild(card);
        });
    }

    setTimeout(() => {
        document.querySelectorAll('.confidence-fill').forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    }, 100);

    const engineGrid = document.getElementById('engine-grid');
    if (engineGrid) {
        engineGrid.innerHTML = '';
        const engineLabels = {
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

        Object.entries(data.engine_predictions).forEach(([key, nums], idx) => {
            const lowKey = key.toLowerCase().replace(/_/g, '');
            const div = document.createElement('div');
            div.className = 'engine-card';
            div.style.padding = '1.25rem';
            div.style.background = 'rgba(255,255,255,0.02)';
            div.style.border = '1px solid var(--card-border)';
            div.style.borderRadius = '1.25rem';
            div.style.animation = `fadeInUp 0.6s ease-out ${0.5 + idx * 0.05}s backwards`;
            div.innerHTML = `
                <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; text-align: center;">${engineLabels[lowKey] || key}</div>
                <div id="engine-balls-${key}" class="numbers-row" style="justify-content: center;"></div>
            `;
            engineGrid.appendChild(div);
            renderBallRow(`engine-balls-${key}`, nums, 2.0);
        });
    }
}

function renderBallRow(containerId, numbers, size) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    numbers.forEach(num => {
        const ball = document.createElement('div');
        ball.className = `number-ball ${getNumberColorClass(num)}`;
        ball.style.width = `${size}rem`;
        ball.style.height = `${size}rem`;
        ball.style.fontSize = `${size * 0.4}rem`;
        ball.innerText = num;
        container.appendChild(ball);
    });
}

function getNumberColorClass(num) {
    if (num <= 10) return 'num-1-10';
    if (num <= 20) return 'num-11-20';
    if (num <= 30) return 'num-21-30';
    if (num <= 40) return 'num-31-40';
    return 'num-41-45';
}
