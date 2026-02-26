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
    constructor(apiBaseUrl = 'http://localhost:5001', authToken = null) {
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
                hodPM: bounds.hod_premarket,
                lodPM: bounds.lod_premarket,
                hodID: bounds.hod_intraday,
                lodID: bounds.lod_intraday
            },
            levels: levels,
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
                    <h3>⚠️ Forecast Error</h3>
                    <p>${formattedData.error}</p>
                    ${formattedData.suggestion ? `<p class="suggestion">💡 ${formattedData.suggestion}</p>` : ''}
                </div>
            `;
            return;
        }

        const target = formattedData.target;
        const isUp = target.move >= 0;
        const confidenceColor = target.confidence > 0.7 ? '#10b981' : target.confidence > 0.5 ? '#f59e0b' : '#ef4444';

        container.innerHTML = `
            <div class="lstm-forecast-container">
                <div class="lstm-forecast-header">
                    <h3>🎯 ${formattedData.question}</h3>
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
                        <div class="level-label">🎯 Closest Level</div>
                        <div class="level-info">
                            <span class="level-price">$${formattedData.closestLevel.price?.toFixed(2) || 'N/A'}</span>
                            ${formattedData.closestLevel.strength ? `
                                <span class="level-strength">Strength: ${(formattedData.closestLevel.strength * 100).toFixed(0)}%</span>
                            ` : ''}
                        </div>
                    </div>
                ` : ''}

                <div class="lstm-forecast-bounds">
                    <div class="bounds-header">📊 Theoretical Bounds</div>
                    <div class="bounds-grid">
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
                    <div class="levels-header">🔍 Levels Detected</div>
                    <div class="levels-grid">
                        <div class="level-type">HDBSCAN: ${formattedData.levels.hdbscan}</div>
                        <div class="level-type">OPTICS: ${formattedData.levels.optics}</div>
                        <div class="level-type">Interaction: ${formattedData.levels.interaction}</div>
                        <div class="level-type">ML-Confluence: ${formattedData.levels.ml_confluence}</div>
                        <div class="level-type">Multiscale: ${formattedData.levels.multiscale}</div>
                    </div>
                </div>

                ${formattedData.attention && formattedData.attention.weights ? `
                    <div class="lstm-forecast-attention">
                        <div class="attention-header">👁️ Attention Focus</div>
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
    background: #1a1a1a;
    border-radius: 12px;
    padding: 20px;
    color: #e5e5e5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.lstm-forecast-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    border-bottom: 1px solid #333;
    padding-bottom: 10px;
}

.lstm-forecast-header h3 {
    margin: 0;
    font-size: 1.25rem;
    color: #fff;
}

.model-badge {
    background: #3b82f6;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.lstm-forecast-target {
    margin: 20px 0;
}

.target-price {
    text-align: center;
    margin-bottom: 20px;
}

.target-label {
    font-size: 0.875rem;
    color: #9ca3af;
    margin-bottom: 8px;
}

.target-value {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin: 8px 0;
}

.target-move {
    font-size: 1.25rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-block;
}

.target-move.up {
    color: #10b981;
    background: rgba(16, 185, 129, 0.1);
}

.target-move.down {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
}

.target-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 20px;
}

.metric {
    text-align: center;
    padding: 12px;
    background: #252525;
    border-radius: 8px;
}

.metric-label {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.125rem;
    font-weight: 600;
    color: #fff;
}

.lstm-forecast-level {
    margin: 20px 0;
    padding: 16px;
    background: #252525;
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
}

.level-label {
    font-size: 0.875rem;
    color: #9ca3af;
    margin-bottom: 8px;
}

.level-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.level-price {
    font-size: 1.25rem;
    font-weight: 600;
    color: #fff;
}

.level-strength {
    font-size: 0.875rem;
    color: #9ca3af;
}

.lstm-forecast-bounds,
.lstm-forecast-levels,
.lstm-forecast-attention {
    margin: 20px 0;
    padding: 16px;
    background: #252525;
    border-radius: 8px;
}

.bounds-header,
.levels-header,
.attention-header {
    font-size: 0.875rem;
    color: #9ca3af;
    margin-bottom: 12px;
    font-weight: 600;
}

.bounds-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}

.bound-item {
    padding: 8px;
    background: #1a1a1a;
    border-radius: 6px;
}

.bound-label {
    font-size: 0.75rem;
    color: #6b7280;
    margin-bottom: 4px;
}

.bound-value {
    font-size: 1rem;
    font-weight: 600;
    color: #fff;
}

.levels-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 8px;
}

.level-type {
    padding: 8px;
    background: #1a1a1a;
    border-radius: 6px;
    font-size: 0.875rem;
    text-align: center;
    color: #e5e5e5;
}

.attention-info {
    font-size: 0.875rem;
    color: #e5e5e5;
}

.lstm-forecast-error {
    padding: 20px;
    background: #1a1a1a;
    border-radius: 12px;
    border-left: 4px solid #ef4444;
}

.lstm-forecast-error h3 {
    color: #ef4444;
    margin-top: 0;
}

.lstm-forecast-error p {
    color: #e5e5e5;
    margin: 8px 0;
}

.suggestion {
    color: #f59e0b !important;
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
