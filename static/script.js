/**
 * Anime Hybrid Recommender - Frontend JavaScript
 * ================================================
 * Handles all UI interactions, API calls, and state management.
 */

// =============================================================================
// State Management
// =============================================================================

const state = {
    selectedAnime: null,
    weights: {
        alpha: 0.4,
        beta: 0.4,
        gamma: 0.2
    },
    topK: 5,
    recommendations: [],
    isLoading: false
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    // Search
    searchInput: document.getElementById('searchInput'),
    searchDropdown: document.getElementById('searchDropdown'),
    recommendBtn: document.getElementById('recommendBtn'),
    suggestionTags: document.getElementById('suggestionTags'),

    // Weight sliders
    alphaSlider: document.getElementById('alphaSlider'),
    betaSlider: document.getElementById('betaSlider'),
    gammaSlider: document.getElementById('gammaSlider'),
    alphaValue: document.getElementById('alphaValue'),
    betaValue: document.getElementById('betaValue'),
    gammaValue: document.getElementById('gammaValue'),
    weightSum: document.getElementById('weightSum'),

    // Top K
    topkInput: document.getElementById('topkInput'),

    // Query anime card
    queryAnimeCard: document.getElementById('queryAnimeCard'),
    queryAnimeName: document.getElementById('queryAnimeName'),
    queryAnimeScore: document.getElementById('queryAnimeScore'),
    queryAnimeType: document.getElementById('queryAnimeType'),
    queryAnimeThemes: document.getElementById('queryAnimeThemes'),

    // Results
    loadingSpinner: document.getElementById('loadingSpinner'),
    emptyState: document.getElementById('emptyState'),
    resultsContainer: document.getElementById('resultsContainer'),
    resultsCount: document.getElementById('resultsCount'),
    weightsDisplay: document.getElementById('weightsDisplay'),
    resultsBody: document.getElementById('resultsBody'),

    // Chart
    chartCard: document.getElementById('chartCard'),
    chartContainer: document.getElementById('chartContainer'),

    // Metrics
    precisionValue: document.getElementById('precisionValue'),
    recallValue: document.getElementById('recallValue'),
    ndcgValue: document.getElementById('ndcgValue'),
    bestConfigValue: document.getElementById('bestConfigValue'),

    // Toast
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage')
};

// =============================================================================
// API Functions
// =============================================================================

const API_BASE = '/api';

async function searchAnime(query) {
    try {
        const response = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error('Search failed');
        return await response.json();
    } catch (error) {
        console.error('Search error:', error);
        showToast('Search failed. Please try again.', 'error');
        return { results: [] };
    }
}

async function getRecommendations(title, topK, alpha, beta, gamma) {
    try {
        const response = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title,
                top_k: topK,
                alpha,
                beta,
                gamma
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Recommendation failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Recommendation error:', error);
        showToast(error.message, 'error');
        return null;
    }
}

async function getMetrics() {
    try {
        const response = await fetch(`${API_BASE}/metrics`);
        if (!response.ok) throw new Error('Failed to load metrics');
        return await response.json();
    } catch (error) {
        console.error('Metrics error:', error);
        return null;
    }
}

// =============================================================================
// UI Functions
// =============================================================================

function showToast(message, type = 'info') {
    elements.toast.className = `toast ${type}`;
    elements.toastMessage.textContent = message;
    elements.toast.classList.remove('hidden');

    setTimeout(() => {
        elements.toast.classList.add('hidden');
    }, 3000);
}

function updateWeightDisplay() {
    const alpha = parseInt(elements.alphaSlider.value) / 100;
    const beta = parseInt(elements.betaSlider.value) / 100;
    const gamma = parseInt(elements.gammaSlider.value) / 100;

    state.weights = { alpha, beta, gamma };

    elements.alphaValue.textContent = alpha.toFixed(2);
    elements.betaValue.textContent = beta.toFixed(2);
    elements.gammaValue.textContent = gamma.toFixed(2);

    const sum = alpha + beta + gamma;
    const sumElement = elements.weightSum.querySelector('.sum-value');
    sumElement.textContent = sum.toFixed(2);

    if (Math.abs(sum - 1.0) < 0.01) {
        sumElement.className = 'sum-value valid';
    } else {
        sumElement.className = 'sum-value invalid';
    }
}

function setWeightPreset(alpha, beta, gamma) {
    elements.alphaSlider.value = alpha;
    elements.betaSlider.value = beta;
    elements.gammaSlider.value = gamma;
    updateWeightDisplay();
}

function showLoading() {
    state.isLoading = true;
    elements.loadingSpinner.classList.remove('hidden');
    elements.emptyState.classList.add('hidden');
    elements.resultsContainer.classList.add('hidden');
    elements.chartCard.classList.add('hidden');
    elements.recommendBtn.disabled = true;
}

function hideLoading() {
    state.isLoading = false;
    elements.loadingSpinner.classList.add('hidden');
    elements.recommendBtn.disabled = !state.selectedAnime;
}

function displaySearchResults(results) {
    if (results.length === 0) {
        elements.searchDropdown.classList.add('hidden');
        return;
    }

    elements.searchDropdown.innerHTML = results.map(anime => `
        <div class="search-item" data-name="${anime.name}">
            <span class="search-item-name">${anime.name}</span>
            <div class="search-item-meta">
                <span class="search-item-score">⭐ ${anime.score.toFixed(2)}</span>
                <span class="search-item-type">${anime.type}</span>
            </div>
        </div>
    `).join('');

    elements.searchDropdown.classList.remove('hidden');

    // Add click handlers
    elements.searchDropdown.querySelectorAll('.search-item').forEach(item => {
        item.addEventListener('click', () => {
            selectAnime(item.dataset.name);
        });
    });
}

function selectAnime(name) {
    state.selectedAnime = name;
    elements.searchInput.value = name;
    elements.searchDropdown.classList.add('hidden');
    elements.recommendBtn.disabled = false;
}

function displayQueryAnime(anime) {
    elements.queryAnimeName.textContent = anime.name;
    elements.queryAnimeScore.querySelector('.score-value').textContent = anime.score.toFixed(2);
    elements.queryAnimeType.textContent = anime.type || 'Unknown';
    elements.queryAnimeThemes.textContent = anime.themes || 'No themes';

    // Set image
    const imageEl = document.getElementById('queryAnimeImage');
    if (anime.image_url && anime.image_url.trim() !== '') {
        imageEl.src = anime.image_url;
        imageEl.style.display = 'block';
    } else {
        imageEl.style.display = 'none';
    }

    // Set link
    const linkEl = document.getElementById('queryAnimeLink');
    const externalLinkEl = document.getElementById('queryAnimeExternalLink');

    if (anime.anime_url && anime.anime_url.trim() !== '') {
        linkEl.href = anime.anime_url;
        externalLinkEl.href = anime.anime_url;
        externalLinkEl.classList.remove('hidden');
    } else {
        linkEl.href = '#';
        externalLinkEl.classList.add('hidden');
    }

    elements.queryAnimeCard.classList.remove('hidden');
}

function displayRecommendations(data) {
    state.recommendations = data.recommendations;

    // Update results header
    elements.resultsCount.textContent = `${data.total_results} results`;
    elements.weightsDisplay.innerHTML = `
        <span class="weight-alpha">α=${data.weights.alpha.toFixed(2)}</span>
        <span class="weight-beta">β=${data.weights.beta.toFixed(2)}</span>
        <span class="weight-gamma">γ=${data.weights.gamma.toFixed(2)}</span>
    `;

    // Generate table rows
    elements.resultsBody.innerHTML = data.recommendations.map((rec, idx) => {
        const rankClass = idx < 3 ? `rank-${idx + 1}` : '';
        const hasImage = rec.image_url && rec.image_url.trim() !== '';
        const hasLink = rec.anime_url && rec.anime_url.trim() !== '';

        const imageHtml = hasImage
            ? `<img src="${rec.image_url}" class="anime-image-small" alt="${rec.name}" onerror="this.style.display='none'">`
            : '<span class="no-image">📺</span>';

        const nameHtml = hasLink
            ? `<a href="${rec.anime_url}" target="_blank" class="table-name-link">${rec.name} <span class="link-icon">↗</span></a>`
            : rec.name;

        return `
            <tr>
                <td class="col-rank">
                    <span class="rank-badge ${rankClass}">${rec.rank}</span>
                </td>
                <td class="col-image">${imageHtml}</td>
                <td class="col-name">${nameHtml}</td>
                <td class="col-score score-cell">${rec.score.toFixed(2)}</td>
                <td class="col-type">
                    <span class="type-badge">${rec.type}</span>
                </td>
                <td class="col-hybrid hybrid-cell">${rec.hybrid_score.toFixed(3)}</td>
                <td class="col-tfidf tfidf-cell">${rec.tfidf_sim.toFixed(3)}</td>
                <td class="col-sbert sbert-cell">${rec.sbert_sim.toFixed(3)}</td>
                <td class="col-themes">${rec.themes}</td>
            </tr>
        `;
    }).join('');

    // Show results
    elements.emptyState.classList.add('hidden');
    elements.resultsContainer.classList.remove('hidden');

    // Display chart
    displayChart(data.recommendations);
}

function displayChart(recommendations) {
    const maxTfidf = Math.max(...recommendations.map(r => r.tfidf_sim));
    const maxSbert = Math.max(...recommendations.map(r => r.sbert_sim));
    const maxVal = Math.max(maxTfidf, maxSbert, 0.1);

    elements.chartContainer.innerHTML = recommendations.slice(0, 5).map(rec => {
        const tfidfWidth = (rec.tfidf_sim / maxVal) * 100;
        const sbertWidth = (rec.sbert_sim / maxVal) * 100;

        return `
            <div class="chart-bar">
                <span class="chart-label" title="${rec.name}">${rec.name.substring(0, 20)}${rec.name.length > 20 ? '...' : ''}</span>
                <div class="chart-bars">
                    <div class="bar bar-tfidf" style="width: ${tfidfWidth}%" title="TF-IDF: ${rec.tfidf_sim.toFixed(3)}"></div>
                    <div class="bar bar-sbert" style="width: ${sbertWidth}%" title="SBERT: ${rec.sbert_sim.toFixed(3)}"></div>
                </div>
                <div class="chart-values">
                    <span style="color: var(--tfidf-color)">${rec.tfidf_sim.toFixed(2)}</span>
                    <span style="color: var(--sbert-color)">${rec.sbert_sim.toFixed(2)}</span>
                </div>
            </div>
        `;
    }).join('');

    elements.chartCard.classList.remove('hidden');
}

function displayMetrics(metrics) {
    if (!metrics) return;

    elements.precisionValue.textContent = metrics.precision_at_5.toFixed(3);
    elements.recallValue.textContent = metrics.recall_at_5.toFixed(3);
    elements.ndcgValue.textContent = metrics.ndcg_at_5.toFixed(3);
    elements.bestConfigValue.textContent = metrics.best_config.name;
}

// =============================================================================
// Event Handlers
// =============================================================================

let searchTimeout = null;

elements.searchInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();

    // Clear previous timeout
    if (searchTimeout) clearTimeout(searchTimeout);

    if (query.length < 2) {
        elements.searchDropdown.classList.add('hidden');
        state.selectedAnime = null;
        elements.recommendBtn.disabled = true;
        return;
    }

    // Debounce search
    searchTimeout = setTimeout(async () => {
        const data = await searchAnime(query);
        displaySearchResults(data.results);
    }, 300);
});

elements.searchInput.addEventListener('focus', () => {
    if (elements.searchDropdown.children.length > 0) {
        elements.searchDropdown.classList.remove('hidden');
    }
});

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-container')) {
        elements.searchDropdown.classList.add('hidden');
    }
});

// Weight sliders
elements.alphaSlider.addEventListener('input', updateWeightDisplay);
elements.betaSlider.addEventListener('input', updateWeightDisplay);
elements.gammaSlider.addEventListener('input', updateWeightDisplay);

// Preset buttons
document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
        setWeightPreset(
            parseInt(btn.dataset.alpha),
            parseInt(btn.dataset.beta),
            parseInt(btn.dataset.gamma)
        );
    });
});

// Top K input
elements.topkInput.addEventListener('change', (e) => {
    let value = parseInt(e.target.value);
    if (value < 1) value = 1;
    if (value > 20) value = 20;
    e.target.value = value;
    state.topK = value;
});

// Suggestion tags
elements.suggestionTags.querySelectorAll('.tag').forEach(tag => {
    tag.addEventListener('click', () => {
        selectAnime(tag.dataset.anime);
    });
});

// Recommend button
elements.recommendBtn.addEventListener('click', async () => {
    if (!state.selectedAnime) {
        showToast('Please select an anime first', 'error');
        return;
    }

    const sum = state.weights.alpha + state.weights.beta + state.weights.gamma;
    if (Math.abs(sum - 1.0) > 0.01) {
        showToast('Weights must sum to 1.0', 'error');
        return;
    }

    showLoading();

    const data = await getRecommendations(
        state.selectedAnime,
        state.topK,
        state.weights.alpha,
        state.weights.beta,
        state.weights.gamma
    );

    hideLoading();

    if (data) {
        displayQueryAnime(data.query_anime);
        displayRecommendations(data);
        showToast(`Found ${data.total_results} recommendations!`, 'success');
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Enter to recommend
    if (e.key === 'Enter' && state.selectedAnime && !state.isLoading) {
        elements.recommendBtn.click();
    }

    // Escape to close dropdown
    if (e.key === 'Escape') {
        elements.searchDropdown.classList.add('hidden');
    }
});

// =============================================================================
// Initialization
// =============================================================================

async function init() {
    console.log('🎌 Anime Hybrid Recommender - Initializing...');

    // Check API health
    try {
        const response = await fetch(`${API_BASE}/health`);
        const health = await response.json();

        if (health.status !== 'healthy') {
            showToast('Model not loaded. Please check server.', 'error');
        } else {
            console.log(`✅ API healthy. ${health.anime_count} anime available.`);
        }
    } catch (error) {
        showToast('Cannot connect to API server', 'error');
    }

    // Load metrics
    const metrics = await getMetrics();
    displayMetrics(metrics);

    // Initialize weight display
    updateWeightDisplay();

    console.log('✅ Initialization complete');
}

// Start
document.addEventListener('DOMContentLoaded', init);
