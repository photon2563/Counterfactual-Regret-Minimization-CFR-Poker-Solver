/**
 * GTO Poker Solver — Dashboard Application
 * 
 * Renders convergence charts, strategy visualizations, and
 * algorithm comparison tables from training results.
 */

// ─── Color Palette for Chart Lines ───
const COLORS = {
    vanilla:  { line: '#3b82f6', bg: 'rgba(59, 130, 246, 0.15)' },
    cfr_plus: { line: '#10b981', bg: 'rgba(16, 185, 129, 0.15)' },
    dcfr:     { line: '#f59e0b', bg: 'rgba(245, 158, 11, 0.15)' },
    es_mccfr: { line: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.15)' },
};

const ALGO_NAMES = {
    vanilla:  'Vanilla CFR',
    cfr_plus: 'CFR+',
    dcfr:     'DCFR',
    es_mccfr: 'Ext. Sampling MCCFR',
};

// ─── State ───
let state = {
    game: 'kuhn',
    scale: 'log',
    kuhnData: null,
    leducData: null,
};

// ─── Embedded training results ───
// These will be loaded from JSON files or embedded directly
const KUHN_RESULTS = {
    vanilla: {
        iterations: [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        exploitability: [0.666667, 0.013010, 0.009004, 0.008576, 0.006073, 0.004665, 0.005133, 0.003717, 0.004708, 0.005543, 0.002972],
        time: 1.86,
        final_exploitability: 0.002972,
    },
    cfr_plus: {
        iterations: [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        exploitability: [0.541667, 0.009709, 0.006791, 0.005549, 0.004817, 0.004314, 0.003936, 0.003641, 0.003419, 0.003220, 0.003049],
        time: 3.23,
        final_exploitability: 0.003049,
    },
    dcfr: {
        iterations: [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        exploitability: [0.666667, 0.005836, 0.008802, 0.008403, 0.005416, 0.004953, 0.005077, 0.004682, 0.004529, 0.001718, 0.004056],
        time: 2.20,
        final_exploitability: 0.004056,
    },
    es_mccfr: {
        iterations: [1, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
        exploitability: [0.916667, 0.036227, 0.024876, 0.017990, 0.022776, 0.011213, 0.021065, 0.021552, 0.009700, 0.017481, 0.021069],
        time: 0.72,
        final_exploitability: 0.021069,
    },
};

// Kuhn Nash equilibrium (alpha = 1/3)
const KUHN_NASH = {
    '0:': { CHECK: 2/3, BET: 1/3 },
    '0:pb': { FOLD: 1.0, CALL: 0.0 },
    '1:': { CHECK: 1.0, BET: 0.0 },
    '1:pb': { FOLD: 1/3, CALL: 2/3 },
    '2:': { CHECK: 0.0, BET: 1.0 },
    '2:pb': { FOLD: 0.0, CALL: 1.0 },
    '0:p': { CHECK: 2/3, BET: 1/3 },
    '0:b': { FOLD: 1.0, CALL: 0.0 },
    '1:p': { CHECK: 1.0, BET: 0.0 },
    '1:b': { FOLD: 2/3, CALL: 1/3 },
    '2:p': { CHECK: 0.0, BET: 1.0 },
    '2:b': { FOLD: 0.0, CALL: 1.0 },
};

// CFR converged strategy approximation
const KUHN_CFR_STRATEGY = {
    '0:': { CHECK: 0.784, BET: 0.216, player: 'P0', card: 'Jack', context: 'Opening' },
    '0:pb': { FOLD: 1.000, CALL: 0.000, player: 'P0', card: 'Jack', context: 'Facing bet' },
    '1:': { CHECK: 1.000, BET: 0.000, player: 'P0', card: 'Queen', context: 'Opening' },
    '1:pb': { FOLD: 0.446, CALL: 0.554, player: 'P0', card: 'Queen', context: 'Facing bet' },
    '2:': { CHECK: 0.332, BET: 0.668, player: 'P0', card: 'King', context: 'Opening' },
    '2:pb': { FOLD: 0.000, CALL: 1.000, player: 'P0', card: 'King', context: 'Facing bet' },
    '0:p': { CHECK: 0.664, BET: 0.336, player: 'P1', card: 'Jack', context: 'After check' },
    '0:b': { FOLD: 1.000, CALL: 0.000, player: 'P1', card: 'Jack', context: 'Facing bet' },
    '1:p': { CHECK: 1.000, BET: 0.000, player: 'P1', card: 'Queen', context: 'After check' },
    '1:b': { FOLD: 0.663, CALL: 0.337, player: 'P1', card: 'Queen', context: 'Facing bet' },
    '2:p': { CHECK: 0.000, BET: 1.000, player: 'P1', card: 'King', context: 'After check' },
    '2:b': { FOLD: 0.000, CALL: 1.000, player: 'P1', card: 'King', context: 'Facing bet' },
};


// ─── Initialization ───
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initToggles();
    loadData();
    renderAll();
});


// ─── Tab Navigation ───
function initTabs() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
        });
    });
}


// ─── Toggle Controls ───
function initToggles() {
    // Game toggle
    document.querySelectorAll('#game-toggle .toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#game-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.game = btn.dataset.game;
            renderAll();
        });
    });

    // Scale toggle
    document.querySelectorAll('#scale-toggle .toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#scale-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.scale = btn.dataset.scale;
            renderChart();
        });
    });
}


// ─── Data Loading ───
function loadData() {
    state.kuhnData = KUHN_RESULTS;
    
    // Try to load from JSON files
    fetch('../benchmarks/data/kuhn_results.json')
        .then(r => r.json())
        .then(data => {
            state.kuhnData = data;
            renderAll();
        })
        .catch(() => {}); // Use embedded data

    fetch('../benchmarks/data/leduc_results.json')
        .then(r => r.json())
        .then(data => {
            state.leducData = data;
            if (state.game === 'leduc') renderAll();
        })
        .catch(() => {});
}


// ─── Render All ───
function renderAll() {
    renderStats();
    renderChart();
    renderComparisonTable();
    renderStrategyTab();
    renderNashTable();
}


// ─── Stats Cards ───
function renderStats() {
    const data = state.game === 'kuhn' ? state.kuhnData : state.leducData;
    if (!data) {
        document.getElementById('val-exploit').textContent = '—';
        document.getElementById('val-time').textContent = '—';
        document.getElementById('val-infosets').textContent = '—';
        document.getElementById('val-gamevalue').textContent = '—';
        return;
    }

    // Find best performing algorithm
    let bestAlgo = null;
    let bestExploit = Infinity;
    for (const [name, results] of Object.entries(data)) {
        if (results.final_exploitability < bestExploit) {
            bestExploit = results.final_exploitability;
            bestAlgo = name;
        }
    }

    document.getElementById('val-exploit').textContent = bestExploit.toFixed(6);
    document.getElementById('val-time').textContent = 
        Object.values(data).reduce((sum, d) => sum + d.time, 0).toFixed(2) + 's';
    document.getElementById('val-infosets').textContent = 
        state.game === 'kuhn' ? '12' : '288';
    document.getElementById('val-gamevalue').textContent = 
        state.game === 'kuhn' ? '-0.0556' : '—';
}


// ─── Convergence Chart ───
function renderChart() {
    const canvas = document.getElementById('convergence-canvas');
    const ctx = canvas.getContext('2d');
    const data = state.game === 'kuhn' ? state.kuhnData : state.leducData;
    
    if (!data) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#64748b';
        ctx.font = '16px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('No data available for this game', canvas.width / 2, canvas.height / 2);
        return;
    }

    // Set canvas resolution for retina
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    const W = rect.width;
    const H = rect.height;
    const padding = { top: 20, right: 30, bottom: 50, left: 80 };
    const plotW = W - padding.left - padding.right;
    const plotH = H - padding.top - padding.bottom;

    // Clear
    ctx.clearRect(0, 0, W, H);

    // Find data ranges
    let maxIter = 0;
    let maxExploit = 0;
    let minExploit = Infinity;
    
    for (const results of Object.values(data)) {
        maxIter = Math.max(maxIter, ...results.iterations);
        maxExploit = Math.max(maxExploit, ...results.exploitability);
        const nonZero = results.exploitability.filter(e => e > 0);
        if (nonZero.length > 0) {
            minExploit = Math.min(minExploit, ...nonZero);
        }
    }

    // Scale functions
    const xScale = (x) => padding.left + (x / maxIter) * plotW;
    
    let yScale;
    if (state.scale === 'log') {
        const logMin = Math.log10(Math.max(minExploit * 0.5, 1e-6));
        const logMax = Math.log10(maxExploit * 1.2);
        yScale = (y) => {
            if (y <= 0) return padding.top + plotH;
            const logY = Math.log10(y);
            return padding.top + plotH - ((logY - logMin) / (logMax - logMin)) * plotH;
        };
    } else {
        yScale = (y) => padding.top + plotH - (y / (maxExploit * 1.1)) * plotH;
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
    ctx.lineWidth = 1;
    
    // Y-axis grid
    const yTicks = state.scale === 'log' 
        ? [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    
    for (const tick of yTicks) {
        if (tick > maxExploit * 1.2) continue;
        const y = yScale(tick);
        if (y < padding.top || y > padding.top + plotH) continue;
        
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + plotW, y);
        ctx.stroke();
        
        ctx.fillStyle = '#64748b';
        ctx.font = '11px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText(tick < 0.01 ? tick.toExponential(0) : tick.toFixed(3), padding.left - 10, y + 4);
    }

    // X-axis ticks
    const xTicks = [0, maxIter * 0.25, maxIter * 0.5, maxIter * 0.75, maxIter];
    for (const tick of xTicks) {
        const x = xScale(tick);
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotH);
        ctx.stroke();
        
        ctx.fillStyle = '#64748b';
        ctx.font = '11px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(tick >= 1000 ? (tick/1000).toFixed(0) + 'K' : tick.toString(), x, padding.top + plotH + 20);
    }

    // Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Iterations', padding.left + plotW / 2, H - 5);
    
    ctx.save();
    ctx.translate(14, padding.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Exploitability', 0, 0);
    ctx.restore();

    // Draw data lines
    const legend = document.getElementById('chart-legend');
    legend.innerHTML = '';

    for (const [name, results] of Object.entries(data)) {
        const color = COLORS[name] || { line: '#fff', bg: 'rgba(255,255,255,0.1)' };
        
        // Line
        ctx.strokeStyle = color.line;
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        
        let started = false;
        for (let i = 0; i < results.iterations.length; i++) {
            const x = xScale(results.iterations[i]);
            const y = yScale(results.exploitability[i]);
            if (!started) {
                ctx.moveTo(x, y);
                started = true;
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Dots
        ctx.fillStyle = color.line;
        for (let i = 0; i < results.iterations.length; i++) {
            const x = xScale(results.iterations[i]);
            const y = yScale(results.exploitability[i]);
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }

        // Legend entry
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        legendItem.innerHTML = `
            <div class="legend-color" style="background:${color.line}"></div>
            <span>${ALGO_NAMES[name] || name}</span>
        `;
        legend.appendChild(legendItem);
    }
}


// ─── Comparison Table ───
function renderComparisonTable() {
    const data = state.game === 'kuhn' ? state.kuhnData : state.leducData;
    const tbody = document.getElementById('comparison-tbody');
    tbody.innerHTML = '';

    if (!data) return;

    // Find best exploitability
    let bestExploit = Infinity;
    for (const results of Object.values(data)) {
        bestExploit = Math.min(bestExploit, results.final_exploitability);
    }

    for (const [name, results] of Object.entries(data)) {
        const row = document.createElement('tr');
        const isBest = results.final_exploitability === bestExploit;
        
        // Convergence rate: exploit at 1000 / exploit at end
        const earlyExploit = results.exploitability[1] || results.exploitability[0];
        const rate = (earlyExploit / results.final_exploitability).toFixed(1) + 'x';
        
        row.innerHTML = `
            <td>${ALGO_NAMES[name] || name}</td>
            <td class="${isBest ? 'best' : ''}">${results.final_exploitability.toFixed(6)}</td>
            <td>${results.time.toFixed(2)}s</td>
            <td>${rate} reduction</td>
        `;
        tbody.appendChild(row);
    }
}


// ─── Strategy Visualization ───
function renderStrategyTab() {
    const grid = document.getElementById('strategy-grid');
    grid.innerHTML = '';

    const cards = ['Jack', 'Queen', 'King'];
    const cardKeys = [0, 1, 2];
    const cardClasses = ['jack', 'queen', 'king'];
    const cardSymbols = ['J♣', 'Q♦', 'K♠'];

    // P0 strategy cards
    for (let i = 0; i < 3; i++) {
        const cardRank = cardKeys[i];
        const openKey = `${cardRank}:`;
        const pbKey = `${cardRank}:pb`;
        const openStrat = KUHN_CFR_STRATEGY[openKey];
        const pbStrat = KUHN_CFR_STRATEGY[pbKey];

        const card = document.createElement('div');
        card.className = 'strategy-card glass';
        card.innerHTML = `
            <div class="strategy-card-header">
                <span class="strategy-card-rank ${cardClasses[i]}">${cardSymbols[i]}</span>
                <span class="strategy-card-player">Player 0</span>
            </div>
            <div style="font-size:0.7rem; color:var(--text-muted); margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.06em">Opening Action</div>
            ${renderActionBar('Check', openStrat.CHECK, 'check')}
            ${renderActionBar('Bet', openStrat.BET, 'bet')}
            <div style="font-size:0.7rem; color:var(--text-muted); margin:0.75rem 0 0.5rem; text-transform:uppercase; letter-spacing:0.06em">Facing Bet (check-bet)</div>
            ${renderActionBar('Fold', pbStrat.FOLD, 'fold')}
            ${renderActionBar('Call', pbStrat.CALL, 'call')}
        `;
        grid.appendChild(card);
    }

    // P1 strategy cards
    for (let i = 0; i < 3; i++) {
        const cardRank = cardKeys[i];
        const pKey = `${cardRank}:p`;
        const bKey = `${cardRank}:b`;
        const pStrat = KUHN_CFR_STRATEGY[pKey];
        const bStrat = KUHN_CFR_STRATEGY[bKey];

        const card = document.createElement('div');
        card.className = 'strategy-card glass';
        card.innerHTML = `
            <div class="strategy-card-header">
                <span class="strategy-card-rank ${cardClasses[i]}">${cardSymbols[i]}</span>
                <span class="strategy-card-player">Player 1</span>
            </div>
            <div style="font-size:0.7rem; color:var(--text-muted); margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.06em">After Opponent Checks</div>
            ${renderActionBar('Check', pStrat.CHECK, 'check')}
            ${renderActionBar('Bet', pStrat.BET, 'bet')}
            <div style="font-size:0.7rem; color:var(--text-muted); margin:0.75rem 0 0.5rem; text-transform:uppercase; letter-spacing:0.06em">Facing Bet</div>
            ${renderActionBar('Fold', bStrat.FOLD, 'fold')}
            ${renderActionBar('Call', bStrat.CALL, 'call')}
        `;
        grid.appendChild(card);
    }
}

function renderActionBar(label, prob, type) {
    return `
        <div class="strategy-action">
            <span class="strategy-action-label">${label}</span>
            <div class="strategy-bar-track">
                <div class="strategy-bar-fill ${type}" style="width:${prob * 100}%"></div>
            </div>
            <span class="strategy-prob">${(prob * 100).toFixed(1)}%</span>
        </div>
    `;
}


// ─── Nash Comparison Table ───
function renderNashTable() {
    const tbody = document.getElementById('nash-tbody');
    tbody.innerHTML = '';

    const cardNames = { 0: 'Jack', 1: 'Queen', 2: 'King' };
    const infoSets = ['0:', '0:pb', '1:', '1:pb', '2:', '2:pb', '0:p', '0:b', '1:p', '1:b', '2:p', '2:b'];

    for (const key of infoSets) {
        const cfr = KUHN_CFR_STRATEGY[key];
        const nash = KUHN_NASH[key];
        if (!cfr || !nash) continue;

        const actions = Object.keys(nash);
        for (const action of actions) {
            const cfrProb = cfr[action] || 0;
            const nashProb = nash[action] || 0;
            const error = Math.abs(cfrProb - nashProb);
            
            const errorClass = error < 0.05 ? 'error-low' : error < 0.15 ? 'error-mid' : 'error-high';
            
            const cardIdx = parseInt(key[0]);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td style="font-family:var(--font-mono)">${key}</td>
                <td>${cfr.player}</td>
                <td>${cardNames[cardIdx]}</td>
                <td>${action}</td>
                <td>${cfrProb.toFixed(3)}</td>
                <td>${nashProb.toFixed(3)}</td>
                <td class="${errorClass}">${error.toFixed(3)}</td>
            `;
            tbody.appendChild(row);
        }
    }
}


// ─── Resize handler ───
window.addEventListener('resize', () => {
    renderChart();
});
