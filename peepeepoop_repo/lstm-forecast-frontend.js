/**
 * LSTM Forecast Frontend Component
 * "Where is price going today?" - Level-Based LSTM Forecast
 * 
 * Usage:
 *   import { LSTMForecast } from './lstm-forecast-frontend.js';
 *   const forecast = new LSTMForecast(apiBaseUrl, authToken);
 *   await forecast.getForecast('SPY', '5m', 20);
 */

class LSTMForecast {
    constructor(apiBaseUrl = 'https://fartpie.onrender.com', authToken = null) {
        this.apiBaseUrl = apiBaseUrl;
        this.authToken = authToken;
    }

    /**
     * Get LSTM forecast for a ticker
     * @param {string} ticker - Stock symbol (e.g., 'SPY', 'NQ=F')
     * @param {string} timeframe - Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
     * @param {number} lookback - Lookback window (default: 20)
     * @returns {Promise<Object>} Forecast data
     */
    async getForecast(ticker, timeframe = '5m', lookback = 20) {
        try {
            const url = new URL(`${this.apiBaseUrl}/api/lstm-forecast`);
            url.searchParams.set('ticker', ticker);
            url.searchParams.set('timeframe', timeframe);
            url.searchParams.set('lookback', lookback.toString());

            const headers = {
                'Content-Type': 'application/json'
            };

            if (this.authToken) {
                headers['Authorization'] = `Bearer ${this.authToken}`;
            }

            const response = await fetch(url.toString(), {
                method: 'GET',
                headers: headers,
                credentials: 'include' // For cookie-based auth
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('LSTM Forecast Error:', error);
            throw error;
        }
    }

    /**
     * Format forecast for display
     * @param {Object} forecastData - Response from getForecast()
     * @returns {Object} Formatted display data
     */
    formatForecast(forecastData) {
        if (!forecastData.success) {
            return {
                error: forecastData.error,
                suggestion: forecastData.suggestion || null
            };
        }

        const answer = forecastData.answer;
        const bounds = forecastData.theoretical_bounds;
        const levels = forecastData.levels_detected;

        return {
            question: forecastData.question,
            target: {
                price: answer.target_price,
                move: answer.target_pct_move,
                moveFormatted: `${answer.target_pct_move >= 0 ? '+' : ''}${answer.target_pct_move.toFixed(2)}%`,
                confidence: answer.confidence,
                confidenceFormatted: `${(answer.confidence * 100).toFixed(1)}%`,
                timeMinutes: answer.expected_time_minutes,
                timeBars: answer.expected_time_bars,
                timeFormatted: this.formatTime(answer.expected_time_minutes)
            },
            closestLevel: answer.closest_level,
            attention: answer.attention_focus,
            bounds: {
                hod1std: bounds.hod_1std,
                lod1std: bounds.lod_1std,
                hod2std: bounds.hod_2std,
                lod2std: bounds.lod_2std,
                hod3std: bounds.hod_3std,
                lod3std: bounds.lod_3std,
                hodPM: bounds.hod_premarket,
                lodPM: bounds.lod_premarket,
                hodID: bounds.hod_intraday,
                lodID: bounds.lod_intraday
            },
            levels: levels,
            allLevels: forecastData.all_levels || [],
            currentPrice: answer.target_price / (1 + answer.target_pct_move / 100),
            modelUsed: forecastData.model_used
        };
    }

    /**
     * Format time in minutes to human-readable string
     */
    formatTime(minutes) {
        if (minutes < 60) {
            return `${Math.round(minutes)} minutes`;
        } else if (minutes < 1440) {
            const hours = Math.floor(minutes / 60);
            const mins = Math.round(minutes % 60);
            return mins > 0 ? `${hours}h ${mins}m` : `${hours} hours`;
        } else {
            const days = Math.floor(minutes / 1440);
            const hours = Math.floor((minutes % 1440) / 60);
            return hours > 0 ? `${days}d ${hours}h` : `${days} days`;
        }
    }

    /**
     * Render forecast as HTML
     * @param {Object} formattedData - Output from formatForecast()
     * @param {HTMLElement} container - Container element to render into
     */
    renderForecast(formattedData, container) {
        if (formattedData.error) {
            container.innerHTML = `
                <div class="lstm-forecast-error">
                    <h3>Forecast Error</h3>
                    <p>${formattedData.error}</p>
                    ${formattedData.suggestion ? `<p class="suggestion">${formattedData.suggestion}</p>` : ''}
                </div>
            `;
            return;
        }

        const target = formattedData.target;
        const isUp = target.move >= 0;
        const confidenceColor = target.confidence > 0.7 ? '#6fcf97' : target.confidence > 0.5 ? '#f2c94c' : '#eb5757';

        container.innerHTML = `
            <div class="lstm-forecast-container">
                <div class="lstm-forecast-header">
                    <h3>${formattedData.question}</h3>
                    <span class="model-badge">${formattedData.modelUsed}</span>
                </div>

                <div class="lstm-forecast-target">
                    <div class="target-price">
                        <div class="target-label">Target Price</div>
                        <div class="target-value">$${target.price.toFixed(2)}</div>
                        <div class="target-move ${isUp ? 'up' : 'down'}">
                            ${target.moveFormatted}
                        </div>
                    </div>

                    <div class="target-metrics">
                        <div class="metric">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value" style="color: ${confidenceColor}">
                                ${target.confidenceFormatted}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Expected Time</div>
                            <div class="metric-value">${target.timeFormatted}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Time (Bars)</div>
                            <div class="metric-value">${target.timeBars}</div>
                        </div>
                    </div>
                </div>

                ${formattedData.closestLevel ? `
                    <div class="lstm-forecast-level">
                        <div class="level-label">Closest Level</div>
                        <div class="level-info">
                            <span class="level-price">$${formattedData.closestLevel.price?.toFixed(2) || 'N/A'}</span>
                            ${formattedData.closestLevel.strength ? `
                                <span class="level-strength">Strength: ${(formattedData.closestLevel.strength * 100).toFixed(0)}%</span>
                            ` : ''}
                        </div>
                    </div>
                ` : ''}

                <div class="lstm-forecast-bounds">
                    <div class="bounds-header">Theoretical Bounds</div>
                    <div class="bounds-grid">
                        ${formattedData.bounds.hod1std ? `
                        <div class="bound-item">
                            <div class="bound-label">HOD 1σ</div>
                            <div class="bound-value green">$${formattedData.bounds.hod1std.toFixed(2)}</div>
                        </div>
                        <div class="bound-item">
                            <div class="bound-label">LOD 1σ</div>
                            <div class="bound-value red">$${formattedData.bounds.lod1std.toFixed(2)}</div>
                        </div>
                        <div class="bound-item">
                            <div class="bound-label">HOD 2σ</div>
                            <div class="bound-value green">$${formattedData.bounds.hod2std.toFixed(2)}</div>
                        </div>
                        <div class="bound-item">
                            <div class="bound-label">LOD 2σ</div>
                            <div class="bound-value red">$${formattedData.bounds.lod2std.toFixed(2)}</div>
                        </div>
                        <div class="bound-item">
                            <div class="bound-label">HOD 3σ</div>
                            <div class="bound-value green">$${formattedData.bounds.hod3std.toFixed(2)}</div>
                        </div>
                        <div class="bound-item">
                            <div class="bound-label">LOD 3σ</div>
                            <div class="bound-value red">$${formattedData.bounds.lod3std.toFixed(2)}</div>
                        </div>
                        ` : ''}
                        <div class="bound-item">
                            <div class="bound-label">Pre-market HOD</div>
                            <div class="bound-value">$${formattedData.bounds.hodPM.toFixed(2)}</div>
                        </div>
                        <div class="bound-item">
                            <div class="bound-label">Pre-market LOD</div>
                            <div class="bound-value">$${formattedData.bounds.lodPM.toFixed(2)}</div>
                        </div>
                        <div class="bound-item">
                            <div class="bound-label">Intraday HOD</div>
                            <div class="bound-value">$${formattedData.bounds.hodID.toFixed(2)}</div>
                        </div>
                        <div class="bound-item">
                            <div class="bound-label">Intraday LOD</div>
                            <div class="bound-value">$${formattedData.bounds.lodID.toFixed(2)}</div>
                        </div>
                    </div>
                </div>

                <div class="lstm-forecast-levels">
                    <div class="levels-header">Levels Detected</div>
                    <div class="levels-grid">
                        <div class="level-type">HDBSCAN: ${formattedData.levels.hdbscan}</div>
                        <div class="level-type">OPTICS: ${formattedData.levels.optics}</div>
                        <div class="level-type">Interaction: ${formattedData.levels.interaction}</div>
                        <div class="level-type">ML-Confluence: ${formattedData.levels.ml_confluence}</div>
                        <div class="level-type">Multiscale: ${formattedData.levels.multiscale}</div>
                        <div class="level-type">Neural Net: ${formattedData.levels.neural_network}</div>
                    </div>
                    ${(() => {
                        const allLevels = formattedData.allLevels || [];
                        const cp = formattedData.currentPrice || 0;
                        const resistance = allLevels.filter(l => (l.price || 0) > cp);
                        const support = allLevels.filter(l => (l.price || 0) <= cp);
                        const renderRow = (l, type) => {
                            const dist = cp ? (((l.price - cp) / cp) * 100).toFixed(2) : '0.00';
                            const color = type === 'resistance' ? '#6fcf97' : '#eb5757';
                            const str = l.strength ? (typeof l.strength === 'number' && l.strength <= 1 ? (l.strength * 100).toFixed(0) + '%' : Math.round(l.strength) + '%') : '—';
                            return `<div class="level-row"><span class="level-row-price" style="color:${color}">$${l.price.toFixed(2)}</span><span class="level-row-type">${l.type || l.category || ''}</span><span class="level-row-str">${str}</span><span class="level-row-dist" style="color:${parseFloat(dist) >= 0 ? '#6fcf97' : '#eb5757'}">${parseFloat(dist) >= 0 ? '+' : ''}${dist}%</span></div>`;
                        };
                        let html = '';
                        if (resistance.length > 0) {
                            html += `<div class="level-list-section"><div class="level-list-header">Resistance (${resistance.length})</div>${resistance.slice(0, 15).map(l => renderRow(l, 'resistance')).join('')}</div>`;
                        }
                        if (support.length > 0) {
                            html += `<div class="level-list-section"><div class="level-list-header">Support (${support.length})</div>${support.slice(0, 15).map(l => renderRow(l, 'support')).join('')}</div>`;
                        }
                        return html;
                    })()}
                </div>

                ${formattedData.attention && formattedData.attention.weights ? `
                    <div class="lstm-forecast-attention">
                        <div class="attention-header">Attention Focus</div>
                        <div class="attention-info">
                            Most important bar: ${formattedData.attention.most_important_bar}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }
}

// CSS Styles (add to your stylesheet or inline)
const LSTM_FORECAST_STYLES = `
.lstm-forecast-container {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 2px;
    padding: 32px;
    color: #e8e6e3;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.lstm-forecast-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 28px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.lstm-forecast-header h3 {
    margin: 0;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 400;
    color: #ffffff;
    letter-spacing: 0.02em;
}

.model-badge {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 5px 12px;
    border-radius: 1px;
}

.lstm-forecast-target {
    margin: 24px 0;
}

.target-price {
    text-align: center;
    margin-bottom: 28px;
    padding: 32px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.target-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.3);
    margin-bottom: 12px;
}

.target-value {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 3rem;
    font-weight: 300;
    color: #ffffff;
    margin: 8px 0;
    letter-spacing: -0.01em;
}

.target-move {
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 6px 16px;
    display: inline-block;
    border-radius: 1px;
}

.target-move.up {
    color: #6fcf97;
    border: 1px solid rgba(111, 207, 151, 0.15);
}

.target-move.down {
    color: #eb5757;
    border: 1px solid rgba(235, 87, 87, 0.15);
}

.target-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 2px;
    overflow: hidden;
}

.metric {
    text-align: center;
    padding: 20px 12px;
    background: #0a0a0a;
}

.metric-label {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.3);
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1rem;
    font-weight: 400;
    color: #ffffff;
    letter-spacing: 0.02em;
}

.lstm-forecast-level {
    margin: 24px 0;
    padding: 20px 24px;
    border-left: 1px solid rgba(255, 255, 255, 0.15);
    background: rgba(255, 255, 255, 0.02);
}

.level-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.3);
    margin-bottom: 12px;
}

.level-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.level-price {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.4rem;
    font-weight: 400;
    color: #ffffff;
}

.level-strength {
    font-size: 0.75rem;
    font-weight: 400;
    color: rgba(255, 255, 255, 0.35);
    letter-spacing: 0.05em;
}

.lstm-forecast-bounds,
.lstm-forecast-levels,
.lstm-forecast-attention {
    margin: 24px 0;
    padding: 24px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 2px;
}

.bounds-header,
.levels-header,
.attention-header {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.3);
    margin-bottom: 16px;
}

.bounds-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 2px;
    overflow: hidden;
}

.bound-item {
    padding: 16px;
    background: #0a0a0a;
}

.bound-label {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.25);
    margin-bottom: 6px;
}

.bound-value {
    font-size: 0.95rem;
    font-weight: 400;
    color: #ffffff;
    letter-spacing: 0.02em;
}

.bound-value.green {
    color: #6fcf97;
}

.bound-value.red {
    color: #eb5757;
}

.level-list-section {
    margin-top: 16px;
}

.level-list-header {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.4);
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.level-row {
    display: grid;
    grid-template-columns: 1fr 1fr auto auto;
    gap: 8px;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    font-size: 0.8rem;
}

.level-row-price {
    font-weight: 500;
    letter-spacing: 0.02em;
}

.level-row-type {
    color: rgba(255, 255, 255, 0.45);
    font-size: 0.75rem;
    font-weight: 300;
}

.level-row-str {
    color: rgba(255, 255, 255, 0.5);
    font-size: 0.75rem;
    text-align: right;
}

.level-row-dist {
    font-size: 0.75rem;
    font-weight: 400;
    text-align: right;
    min-width: 55px;
}

.levels-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 2px;
    overflow: hidden;
}

.level-type {
    padding: 12px 16px;
    background: #0a0a0a;
    font-size: 0.8rem;
    text-align: center;
    color: rgba(255, 255, 255, 0.6);
    font-weight: 300;
    letter-spacing: 0.02em;
}

.attention-info {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.5);
    font-weight: 300;
    letter-spacing: 0.02em;
}

.lstm-forecast-error {
    padding: 32px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 2px solid #eb5757;
    border-radius: 2px;
}

.lstm-forecast-error h3 {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-weight: 400;
    font-size: 1.1rem;
    color: #eb5757;
    margin-top: 0;
    margin-bottom: 12px;
}

.lstm-forecast-error p {
    color: rgba(255, 255, 255, 0.5);
    margin: 8px 0;
    font-size: 0.85rem;
    font-weight: 300;
    line-height: 1.6;
}

.suggestion {
    color: rgba(255, 255, 255, 0.4) !important;
    font-style: italic;
}
`;

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LSTMForecast, LSTM_FORECAST_STYLES };
}

// Also make available globally
if (typeof window !== 'undefined') {
    window.LSTMForecast = LSTMForecast;
    window.LSTM_FORECAST_STYLES = LSTM_FORECAST_STYLES;
}
