/**
 * Backtest Frontend Module
 * 
 * Provides UI for running and visualizing walk-forward backtests
 * of level detection algorithms and HOD/LOD predictions.
 * 
 * Enhanced with:
 * - Per-level detail view (each individual level's outcome)
 * - Multi-timeframe comparison view
 * - Sortable/filterable tables
 */

class BacktestUI {
    constructor(apiBaseUrl = '', authToken = null) {
        this.apiBaseUrl = apiBaseUrl;
        this.authToken = authToken;
        this.isRunning = false;
        this.lastResults = null;
        this.activeTab = 'summary';
        this.levelSortCol = 'date';
        this.levelSortAsc = false;
        this.levelFilterAlgo = 'all';
        this.levelFilterOutcome = 'all';
    }

    /**
     * Run a backtest via the API
     */
    async runBacktest(params = {}) {
        const defaults = {
            ticker: 'SPY',
            timeframe: '1d',
            lookback_bars: 200,
            eval_bars: 5,
            step_bars: 1,
            tolerance_pct: 0.15,
            max_eval_points: 100,
            mode: 'full',
        };
        const body = { ...defaults, ...params };

        this.isRunning = true;

        try {
            const headers = { 'Content-Type': 'application/json' };
            if (this.authToken) {
                headers['Authorization'] = `Bearer ${this.authToken}`;
            }

            const response = await fetch(`${this.apiBaseUrl}/api/backtest`, {
                method: 'POST',
                headers,
                credentials: 'include',
                body: JSON.stringify(body),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            this.lastResults = data;
            return data;
        } catch (error) {
            console.error('Backtest error:', error);
            throw error;
        } finally {
            this.isRunning = false;
        }
    }

    /**
     * Render the backtest configuration form
     */
    renderForm(container) {
        container.innerHTML = `
            <div class="backtest-form">
                <h3>Backtest Configuration</h3>
                <div class="form-grid">
                    <div class="form-group">
                        <label for="bt-ticker">Ticker</label>
                        <input type="text" id="bt-ticker" value="SPY" placeholder="SPY, AAPL, etc.">
                    </div>
                    <div class="form-group">
                        <label for="bt-timeframe">Timeframe</label>
                        <select id="bt-timeframe">
                            <option value="1d" selected>1 Day</option>
                            <option value="1h">1 Hour</option>
                            <option value="4h">4 Hour</option>
                            <option value="15m">15 Min</option>
                            <option value="5m">5 Min</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="bt-mode">Mode</label>
                        <select id="bt-mode">
                            <option value="full" selected>Full (Levels + HOD/LOD)</option>
                            <option value="levels">Levels Only</option>
                            <option value="hodlod">HOD/LOD Only</option>
                            <option value="multi_timeframe">Multi-Timeframe Compare</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="bt-lookback">Lookback Bars</label>
                        <input type="number" id="bt-lookback" value="200" min="60" max="500">
                    </div>
                    <div class="form-group">
                        <label for="bt-eval-bars">Eval Bars (forward)</label>
                        <input type="number" id="bt-eval-bars" value="5" min="1" max="20">
                    </div>
                    <div class="form-group">
                        <label for="bt-tolerance">Tolerance (%)</label>
                        <input type="number" id="bt-tolerance" value="0.15" min="0.01" max="1.0" step="0.01">
                    </div>
                    <div class="form-group">
                        <label for="bt-max-points">Max Eval Points</label>
                        <input type="number" id="bt-max-points" value="100" min="10" max="500">
                    </div>
                </div>
                <div id="bt-multi-tf-options" class="bt-multi-tf-options" style="display:none;">
                    <label>Timeframes to Compare:</label>
                    <div class="tf-checkboxes">
                        <label><input type="checkbox" value="5m" class="bt-tf-check"> 5m</label>
                        <label><input type="checkbox" value="15m" class="bt-tf-check" checked> 15m</label>
                        <label><input type="checkbox" value="1h" class="bt-tf-check" checked> 1h</label>
                        <label><input type="checkbox" value="4h" class="bt-tf-check" checked> 4h</label>
                        <label><input type="checkbox" value="1d" class="bt-tf-check" checked> 1d</label>
                    </div>
                </div>
                <button id="bt-run" class="bt-run-btn">Run Backtest</button>
                <div id="bt-status" class="bt-status"></div>
            </div>
        `;

        // Toggle multi-timeframe options
        const modeSelect = container.querySelector('#bt-mode');
        const multiTfOptions = container.querySelector('#bt-multi-tf-options');
        modeSelect.addEventListener('change', () => {
            multiTfOptions.style.display = modeSelect.value === 'multi_timeframe' ? 'block' : 'none';
        });

        // Wire up the run button
        const runBtn = container.querySelector('#bt-run');
        runBtn.addEventListener('click', () => this._handleRun(container));
    }

    async _handleRun(container) {
        const statusEl = container.querySelector('#bt-status');
        const runBtn = container.querySelector('#bt-run');

        const params = {
            ticker: container.querySelector('#bt-ticker').value.trim() || 'SPY',
            timeframe: container.querySelector('#bt-timeframe').value,
            mode: container.querySelector('#bt-mode').value,
            lookback_bars: parseInt(container.querySelector('#bt-lookback').value) || 200,
            eval_bars: parseInt(container.querySelector('#bt-eval-bars').value) || 5,
            tolerance_pct: parseFloat(container.querySelector('#bt-tolerance').value) || 0.15,
            max_eval_points: parseInt(container.querySelector('#bt-max-points').value) || 100,
        };

        // For multi-timeframe mode, gather selected timeframes
        if (params.mode === 'multi_timeframe') {
            const checks = container.querySelectorAll('.bt-tf-check:checked');
            params.timeframes = Array.from(checks).map(c => c.value);
            if (params.timeframes.length === 0) {
                statusEl.textContent = 'Please select at least one timeframe.';
                statusEl.className = 'bt-status error';
                return;
            }
        }

        runBtn.disabled = true;
        runBtn.textContent = 'Running...';
        statusEl.textContent = 'Backtest in progress. This may take a few minutes...';
        statusEl.className = 'bt-status running';

        try {
            const results = await this.runBacktest(params);
            statusEl.textContent = 'Backtest complete!';
            statusEl.className = 'bt-status success';

            // Render results below the form
            let resultsContainer = container.querySelector('.backtest-results');
            if (!resultsContainer) {
                resultsContainer = document.createElement('div');
                resultsContainer.className = 'backtest-results';
                container.appendChild(resultsContainer);
            }
            this.renderResults(results, resultsContainer);
        } catch (error) {
            statusEl.textContent = `Error: ${error.message}`;
            statusEl.className = 'bt-status error';
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Run Backtest';
        }
    }

    /**
     * Render backtest results with tabs
     */
    renderResults(data, container) {
        if (!data || !data.success) {
            container.innerHTML = `<div class="bt-error">Backtest failed: ${data?.error || 'Unknown error'}</div>`;
            return;
        }

        // Detect what kind of results we have
        const hasLevels = (data.levels && data.levels.success) || data.algorithm_metrics;
        const hasHodlod = (data.hodlod && data.hodlod.success) || data.method_metrics;
        const hasMultiTF = data.per_timeframe;
        const hasLevelDetails = this._getLevelDetails(data).length > 0;

        // Build tab bar
        const tabs = [];
        tabs.push({ id: 'summary', label: 'Summary' });
        if (hasLevelDetails) {
            tabs.push({ id: 'per-level', label: 'Per-Level Detail' });
        }
        if (hasMultiTF) {
            tabs.push({ id: 'multi-tf', label: 'Timeframe Comparison' });
        }
        if (hasHodlod) {
            tabs.push({ id: 'hodlod', label: 'HOD/LOD' });
        }

        let html = `
            <div class="bt-results-container">
                <div class="bt-results-header">
                    <h3>Backtest Results: ${data.ticker || ''} ${data.timeframe ? '(' + data.timeframe + ')' : ''}</h3>
                    <span class="bt-timestamp">${data.timestamp || new Date().toISOString()}</span>
                </div>
                <div class="bt-tabs">
                    ${tabs.map(t => `<button class="bt-tab ${t.id === 'summary' ? 'active' : ''}" data-tab="${t.id}">${t.label}</button>`).join('')}
                </div>
                <div class="bt-tab-content" id="bt-tab-summary">
        `;

        // Summary tab content
        if (data.levels && data.levels.success) {
            html += this._renderLevelResults(data.levels);
        } else if (data.algorithm_metrics) {
            html += this._renderLevelResults(data);
        }

        if (hasMultiTF) {
            html += this._renderMultiTFSummary(data);
        }

        html += `</div>`;

        // Per-level detail tab
        if (hasLevelDetails) {
            html += `<div class="bt-tab-content" id="bt-tab-per-level" style="display:none;">`;
            html += this._renderPerLevelDetail(data);
            html += `</div>`;
        }

        // Multi-TF tab
        if (hasMultiTF) {
            html += `<div class="bt-tab-content" id="bt-tab-multi-tf" style="display:none;">`;
            html += this._renderMultiTFDetail(data);
            html += `</div>`;
        }

        // HOD/LOD tab
        if (hasHodlod) {
            html += `<div class="bt-tab-content" id="bt-tab-hodlod" style="display:none;">`;
            if (data.hodlod && data.hodlod.success) {
                html += this._renderHodLodResults(data.hodlod);
            } else if (data.method_metrics) {
                html += this._renderHodLodResults(data);
            }
            html += `</div>`;
        }

        html += '</div>';
        container.innerHTML = html;

        // Wire up tabs
        container.querySelectorAll('.bt-tab').forEach(btn => {
            btn.addEventListener('click', () => {
                container.querySelectorAll('.bt-tab').forEach(b => b.classList.remove('active'));
                container.querySelectorAll('.bt-tab-content').forEach(c => c.style.display = 'none');
                btn.classList.add('active');
                const tabId = btn.getAttribute('data-tab');
                const tabContent = container.querySelector(`#bt-tab-${tabId}`);
                if (tabContent) tabContent.style.display = 'block';
            });
        });

        // Wire up per-level filters and sorting
        if (hasLevelDetails) {
            this._wireUpLevelDetailControls(container, data);
        }
    }

    /**
     * Extract level_details from data (handles nested structures)
     */
    _getLevelDetails(data) {
        if (data.level_details && data.level_details.length > 0) {
            return data.level_details;
        }
        if (data.levels && data.levels.level_details && data.levels.level_details.length > 0) {
            return data.levels.level_details;
        }
        // For multi-timeframe, gather from all timeframes
        if (data.per_timeframe) {
            const allDetails = [];
            for (const [tf, tfData] of Object.entries(data.per_timeframe)) {
                if (tfData.level_details) {
                    tfData.level_details.forEach(d => {
                        allDetails.push({ ...d, timeframe: tf });
                    });
                }
            }
            return allDetails;
        }
        return [];
    }

    _renderLevelResults(levelData) {
        const metrics = levelData.algorithm_metrics;
        if (!metrics) return '';

        // Sort algorithms by hit rate descending
        const sorted = Object.entries(metrics)
            .filter(([_, m]) => m.total_levels_generated > 0)
            .sort((a, b) => b[1].hit_rate - a[1].hit_rate);

        let html = `
            <div class="bt-section">
                <h4>Level Detection Backtest</h4>
                <div class="bt-meta">
                    ${levelData.eval_points || 0} evaluation points | 
                    ${levelData.total_bars || 0} total bars |
                    Lookback: ${levelData.lookback_bars || 0} | 
                    Eval window: ${levelData.eval_bars || 0} bars |
                    Tolerance: ${levelData.tolerance_pct || 0}%
                </div>
                <table class="bt-table">
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            <th>Levels</th>
                            <th>Avg/Eval</th>
                            <th>Hit Rate</th>
                            <th>Bounce Rate</th>
                            <th>Break Rate</th>
                            <th>False Pos.</th>
                            <th>Avg Dist %</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        for (const [name, m] of sorted) {
            const hitClass = m.hit_rate >= 50 ? 'good' : m.hit_rate >= 25 ? 'ok' : 'bad';
            const bounceClass = m.bounce_rate >= 60 ? 'good' : m.bounce_rate >= 40 ? 'ok' : 'bad';

            html += `
                <tr>
                    <td class="algo-name">${name}</td>
                    <td>${m.total_levels_generated}</td>
                    <td>${m.avg_levels_per_eval}</td>
                    <td class="${hitClass}">${m.hit_rate}%</td>
                    <td class="${bounceClass}">${m.bounce_rate}%</td>
                    <td>${m.break_rate}%</td>
                    <td>${m.false_positive_rate}%</td>
                    <td>${m.avg_distance_pct}%</td>
                </tr>
            `;
        }

        // Show algorithms with 0 levels
        const empty = Object.entries(metrics)
            .filter(([_, m]) => m.total_levels_generated === 0);
        for (const [name, _] of empty) {
            html += `
                <tr class="empty-row">
                    <td class="algo-name">${name}</td>
                    <td colspan="7" class="no-data">No levels generated (model/dependency missing?)</td>
                </tr>
            `;
        }

        html += `
                    </tbody>
                </table>
            </div>
        `;

        return html;
    }

    /**
     * Render per-level detail table with filters
     */
    _renderPerLevelDetail(data) {
        const details = this._getLevelDetails(data);
        if (!details.length) return '<p class="bt-no-data">No per-level data available.</p>';

        // Get unique algorithms and outcomes for filters
        const algos = [...new Set(details.map(d => d.algorithm))].sort();
        const outcomes = [...new Set(details.map(d => d.outcome))].sort();

        let html = `
            <div class="bt-section">
                <h4>Per-Level Detail</h4>
                <p class="bt-meta">Every individual level detected during the backtest and its outcome.</p>
                <div class="bt-filters">
                    <div class="bt-filter-group">
                        <label>Algorithm:</label>
                        <select id="bt-level-algo-filter">
                            <option value="all">All Algorithms</option>
                            ${algos.map(a => `<option value="${a}">${a}</option>`).join('')}
                        </select>
                    </div>
                    <div class="bt-filter-group">
                        <label>Outcome:</label>
                        <select id="bt-level-outcome-filter">
                            <option value="all">All Outcomes</option>
                            ${outcomes.map(o => `<option value="${o}">${o}</option>`).join('')}
                        </select>
                    </div>
                    <div class="bt-filter-group">
                        <label>Type:</label>
                        <select id="bt-level-type-filter">
                            <option value="all">All Types</option>
                            <option value="support">Support</option>
                            <option value="resistance">Resistance</option>
                        </select>
                    </div>
                    <span id="bt-level-count" class="bt-filter-count"></span>
                </div>
                <div class="bt-table-wrapper">
                    <table class="bt-table bt-level-table" id="bt-level-detail-table">
                        <thead>
                            <tr>
                                <th class="sortable" data-col="date">Date</th>
                                ${details[0].timeframe !== undefined ? '<th class="sortable" data-col="timeframe">TF</th>' : ''}
                                <th class="sortable" data-col="algorithm">Algorithm</th>
                                <th class="sortable" data-col="level_price">Level Price</th>
                                <th class="sortable" data-col="current_price">Spot Price</th>
                                <th class="sortable" data-col="distance_pct">Distance %</th>
                                <th class="sortable" data-col="level_type">Type</th>
                                <th class="sortable" data-col="strength">Strength</th>
                                <th class="sortable" data-col="outcome">Outcome</th>
                            </tr>
                        </thead>
                        <tbody id="bt-level-detail-body">
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        return html;
    }

    /**
     * Wire up filtering/sorting for level detail table
     */
    _wireUpLevelDetailControls(container, data) {
        const details = this._getLevelDetails(data);
        if (!details.length) return;

        const tbody = container.querySelector('#bt-level-detail-body');
        const countEl = container.querySelector('#bt-level-count');
        const algoFilter = container.querySelector('#bt-level-algo-filter');
        const outcomeFilter = container.querySelector('#bt-level-outcome-filter');
        const typeFilter = container.querySelector('#bt-level-type-filter');
        const hasTF = details[0].timeframe !== undefined;

        if (!tbody) return;

        const renderRows = () => {
            let filtered = details;
            if (algoFilter && algoFilter.value !== 'all') {
                filtered = filtered.filter(d => d.algorithm === algoFilter.value);
            }
            if (outcomeFilter && outcomeFilter.value !== 'all') {
                filtered = filtered.filter(d => d.outcome === outcomeFilter.value);
            }
            if (typeFilter && typeFilter.value !== 'all') {
                filtered = filtered.filter(d => d.level_type === typeFilter.value);
            }

            // Sort
            filtered.sort((a, b) => {
                let va = a[this.levelSortCol];
                let vb = b[this.levelSortCol];
                if (typeof va === 'number' && typeof vb === 'number') {
                    return this.levelSortAsc ? va - vb : vb - va;
                }
                va = String(va || '');
                vb = String(vb || '');
                return this.levelSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
            });

            // Cap at 500 rows for performance
            const capped = filtered.slice(0, 500);

            if (countEl) {
                countEl.textContent = `Showing ${capped.length} of ${filtered.length} levels`;
            }

            tbody.innerHTML = capped.map(d => {
                const outcomeClass = d.outcome === 'bounced' ? 'good'
                    : d.outcome === 'broke' ? 'bad'
                    : d.outcome === 'touched' ? 'ok'
                    : '';
                const typeClass = d.level_type === 'support' ? 'support-tag' : 'resistance-tag';
                return `<tr>
                    <td>${d.date ? d.date.split(' ')[0] : ''}</td>
                    ${hasTF ? `<td class="tf-tag">${d.timeframe || ''}</td>` : ''}
                    <td class="algo-name">${d.algorithm}</td>
                    <td>$${d.level_price.toFixed(2)}</td>
                    <td>$${d.current_price.toFixed(2)}</td>
                    <td>${d.distance_pct.toFixed(3)}%</td>
                    <td><span class="${typeClass}">${d.level_type}</span></td>
                    <td>${d.strength.toFixed(3)}</td>
                    <td class="${outcomeClass}">${d.outcome}</td>
                </tr>`;
            }).join('');
        };

        // Initial render
        renderRows();

        // Filters
        if (algoFilter) algoFilter.addEventListener('change', renderRows);
        if (outcomeFilter) outcomeFilter.addEventListener('change', renderRows);
        if (typeFilter) typeFilter.addEventListener('change', renderRows);

        // Sortable headers
        container.querySelectorAll('.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const col = th.getAttribute('data-col');
                if (this.levelSortCol === col) {
                    this.levelSortAsc = !this.levelSortAsc;
                } else {
                    this.levelSortCol = col;
                    this.levelSortAsc = true;
                }
                // Update sort indicators
                container.querySelectorAll('.sortable').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                th.classList.add(this.levelSortAsc ? 'sort-asc' : 'sort-desc');
                renderRows();
            });
        });
    }

    /**
     * Render multi-timeframe summary on the summary tab
     */
    _renderMultiTFSummary(data) {
        if (!data.per_timeframe) return '';

        const timeframes = Object.keys(data.per_timeframe);

        // Collect all algorithm names across all timeframes
        const allAlgos = new Set();
        for (const tf of timeframes) {
            const tfData = data.per_timeframe[tf];
            if (tfData.algorithm_metrics) {
                Object.keys(tfData.algorithm_metrics).forEach(a => allAlgos.add(a));
            }
        }

        let html = `
            <div class="bt-section">
                <h4>Multi-Timeframe Overview</h4>
                <p class="bt-meta">Hit rates across timeframes for ${data.ticker}</p>
                <table class="bt-table">
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            ${timeframes.map(tf => `<th>${tf} Hit%</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;

        for (const algo of [...allAlgos].sort()) {
            html += `<tr><td class="algo-name">${algo}</td>`;
            for (const tf of timeframes) {
                const tfData = data.per_timeframe[tf];
                const m = tfData.algorithm_metrics ? tfData.algorithm_metrics[algo] : null;
                if (m && m.total_levels_generated > 0) {
                    const cls = m.hit_rate >= 50 ? 'good' : m.hit_rate >= 25 ? 'ok' : 'bad';
                    html += `<td class="${cls}">${m.hit_rate}%</td>`;
                } else {
                    html += `<td class="no-data">-</td>`;
                }
            }
            html += `</tr>`;
        }

        html += `</tbody></table></div>`;
        return html;
    }

    /**
     * Render detailed multi-timeframe tab with full metrics per TF
     */
    _renderMultiTFDetail(data) {
        if (!data.per_timeframe) return '';

        let html = '';
        for (const [tf, tfData] of Object.entries(data.per_timeframe)) {
            if (!tfData.success) {
                html += `<div class="bt-section"><h4>${tf}</h4><p class="bt-error-inline">Failed: ${tfData.error || 'Unknown'}</p></div>`;
                continue;
            }
            html += `<div class="bt-section"><h4>Timeframe: ${tf}</h4>`;
            html += this._renderLevelResults(tfData);
            html += `</div>`;
        }

        return html;
    }

    _renderHodLodResults(hodlodData) {
        const metrics = hodlodData.method_metrics;
        if (!metrics) return '';

        let html = `
            <div class="bt-section">
                <h4>HOD/LOD Prediction Backtest</h4>
                <div class="bt-meta">
                    ${hodlodData.eval_points || 0} evaluation points | 
                    ${hodlodData.total_bars || 0} total bars |
                    Lookback: ${hodlodData.lookback_bars || 0}
                </div>
                <table class="bt-table">
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Points</th>
                            <th>HOD MAE</th>
                            <th>LOD MAE</th>
                            <th>HOD MAPE</th>
                            <th>LOD MAPE</th>
                            <th>Containment</th>
                            <th>HOD Cons.</th>
                            <th>LOD Cons.</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        const methodLabels = {
            'statistical_1std': 'Statistical (1\u03C3)',
            'statistical_2std': 'Statistical (2\u03C3)',
            'statistical_3std': 'Statistical (3\u03C3)',
            'level_constrained': 'Level-Constrained',
        };

        for (const [name, m] of Object.entries(metrics)) {
            if (m.error) continue;
            const label = methodLabels[name] || name;
            const containClass = m.containment_rate >= 80 ? 'good' : m.containment_rate >= 60 ? 'ok' : 'bad';

            html += `
                <tr>
                    <td class="algo-name">${label}</td>
                    <td>${m.total_eval_points}</td>
                    <td>$${m.hod_mae.toFixed(2)}</td>
                    <td>$${m.lod_mae.toFixed(2)}</td>
                    <td>${m.hod_mape.toFixed(2)}%</td>
                    <td>${m.lod_mape.toFixed(2)}%</td>
                    <td class="${containClass}">${m.containment_rate}%</td>
                    <td>${m.hod_conservative_rate}%</td>
                    <td>${m.lod_conservative_rate}%</td>
                </tr>
            `;
        }

        html += `
                    </tbody>
                </table>
                <div class="bt-legend">
                    <span><strong>MAE</strong> = Mean Absolute Error ($ from actual)</span>
                    <span><strong>MAPE</strong> = Mean Abs % Error</span>
                    <span><strong>Containment</strong> = % where predicted range contained actual HOD/LOD</span>
                    <span><strong>Conservative</strong> = % where prediction was beyond actual (safe side)</span>
                </div>
            </div>
        `;

        return html;
    }
}

// CSS Styles
const BACKTEST_STYLES = `
.backtest-form {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 20px;
    color: #e5e5e5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin-bottom: 20px;
}

.backtest-form h3 {
    margin: 0 0 16px 0;
    font-size: 1.25rem;
    color: #fff;
    border-bottom: 1px solid #333;
    padding-bottom: 10px;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.form-group label {
    font-size: 0.75rem;
    color: #9ca3af;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.form-group input,
.form-group select {
    background: #252525;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 8px 10px;
    color: #e5e5e5;
    font-size: 0.875rem;
}

.form-group input:focus,
.form-group select:focus {
    outline: none;
    border-color: #3b82f6;
}

.bt-multi-tf-options {
    margin-bottom: 16px;
    padding: 12px;
    background: #252525;
    border-radius: 8px;
}

.bt-multi-tf-options > label {
    font-size: 0.8rem;
    color: #9ca3af;
    font-weight: 600;
    display: block;
    margin-bottom: 8px;
}

.tf-checkboxes {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}

.tf-checkboxes label {
    font-size: 0.85rem;
    color: #e5e5e5;
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
}

.bt-run-btn {
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
}

.bt-run-btn:hover {
    background: #2563eb;
}

.bt-run-btn:disabled {
    background: #4b5563;
    cursor: not-allowed;
}

.bt-status {
    margin-top: 10px;
    font-size: 0.875rem;
    min-height: 1.5em;
}

.bt-status.running {
    color: #f59e0b;
}

.bt-status.success {
    color: #10b981;
}

.bt-status.error {
    color: #ef4444;
}

.bt-results-container {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 20px;
    color: #e5e5e5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.bt-results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #333;
    padding-bottom: 10px;
    margin-bottom: 12px;
}

.bt-results-header h3 {
    margin: 0;
    font-size: 1.25rem;
    color: #fff;
}

.bt-timestamp {
    font-size: 0.75rem;
    color: #6b7280;
}

/* Tabs */
.bt-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
    border-bottom: 2px solid #333;
    padding-bottom: 0;
}

.bt-tab {
    background: transparent;
    border: none;
    color: #9ca3af;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 8px 16px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
}

.bt-tab:hover {
    color: #e5e5e5;
}

.bt-tab.active {
    color: #3b82f6;
    border-bottom-color: #3b82f6;
}

.bt-section {
    margin-bottom: 24px;
}

.bt-section h4 {
    margin: 0 0 8px 0;
    font-size: 1.1rem;
    color: #fff;
}

.bt-meta {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 12px;
}

.bt-no-data {
    color: #6b7280;
    font-style: italic;
    padding: 20px 0;
}

/* Filters */
.bt-filters {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
}

.bt-filter-group {
    display: flex;
    align-items: center;
    gap: 6px;
}

.bt-filter-group label {
    font-size: 0.75rem;
    color: #9ca3af;
    font-weight: 600;
}

.bt-filter-group select {
    background: #252525;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 4px 8px;
    color: #e5e5e5;
    font-size: 0.8rem;
}

.bt-filter-count {
    font-size: 0.8rem;
    color: #6b7280;
    margin-left: auto;
}

/* Table */
.bt-table-wrapper {
    overflow-x: auto;
    max-height: 600px;
    overflow-y: auto;
}

.bt-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}

.bt-table thead th {
    background: #252525;
    color: #9ca3af;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    border-bottom: 2px solid #404040;
    position: sticky;
    top: 0;
    z-index: 1;
}

.bt-table thead th.sortable {
    cursor: pointer;
    user-select: none;
}

.bt-table thead th.sortable:hover {
    color: #e5e5e5;
}

.bt-table thead th.sort-asc::after {
    content: ' \\2191';
}

.bt-table thead th.sort-desc::after {
    content: ' \\2193';
}

.bt-table tbody td {
    padding: 8px 10px;
    border-bottom: 1px solid #2a2a2a;
    color: #e5e5e5;
}

.bt-table tbody tr:hover {
    background: #252525;
}

.bt-table .algo-name {
    font-weight: 600;
    color: #fff;
}

.bt-table .good {
    color: #10b981;
    font-weight: 600;
}

.bt-table .ok {
    color: #f59e0b;
    font-weight: 600;
}

.bt-table .bad {
    color: #ef4444;
    font-weight: 600;
}

.bt-table .no-data {
    color: #6b7280;
    font-style: italic;
}

.bt-table .empty-row {
    opacity: 0.5;
}

.support-tag {
    background: #064e3b;
    color: #6ee7b7;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.resistance-tag {
    background: #7f1d1d;
    color: #fca5a5;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.tf-tag {
    font-weight: 600;
    color: #93c5fd;
}

.bt-legend {
    margin-top: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    font-size: 0.75rem;
    color: #6b7280;
}

.bt-error {
    padding: 20px;
    background: #1a1a1a;
    border-radius: 12px;
    border-left: 4px solid #ef4444;
    color: #ef4444;
}

.bt-error-inline {
    color: #ef4444;
    font-size: 0.85rem;
}
`;

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BacktestUI, BACKTEST_STYLES };
}

if (typeof window !== 'undefined') {
    window.BacktestUI = BacktestUI;
    window.BACKTEST_STYLES = BACKTEST_STYLES;
}
