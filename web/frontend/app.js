/**
 * Anime Recommendation System - Web Demo
 * Frontend JavaScript Application
 */

// =============================================================================
// Configuration & State
// =============================================================================

const API_BASE = '';  // Same origin
const JIKAN_API = 'https://api.jikan.moe/v4';
const PLACEHOLDER_IMAGE = 'https://via.placeholder.com/225x350/1e293b/94a3b8?text=No+Image';

const state = {
    currentPage: 'home',
    selectedAnime: null,
    selectedMethod: 'hybrid',
    topK: 10,
    isLoading: false,
    apiStatus: null,
    cache: {
        search: {},
        recommendations: {},
        popular: {},
        images: {}  // Cache for anime images
    },
    imageQueue: [],  // Queue for fetching images
    isFetchingImages: false
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    // Navigation
    navLinks: document.querySelectorAll('.nav-link'),
    pages: document.querySelectorAll('.page'),
    apiStatus: document.getElementById('apiStatus'),

    // Home Page
    searchInput: document.getElementById('searchInput'),
    searchBtn: document.getElementById('searchBtn'),
    autocompleteDropdown: document.getElementById('autocompleteDropdown'),
    quickTags: document.querySelectorAll('.quick-tag'),
    selectedAnimeCard: document.getElementById('selectedAnimeCard'),
    controlsSection: document.getElementById('controlsSection'),
    methodBtns: document.querySelectorAll('.method-btn'),
    topkInput: document.getElementById('topkInput'),
    getRecommendationsBtn: document.getElementById('getRecommendationsBtn'),
    recommendationsSection: document.getElementById('recommendationsSection'),
    recommendationsGrid: document.getElementById('recommendationsGrid'),
    resultsCount: document.getElementById('resultsCount'),

    // Popular Page
    popularTabs: document.querySelectorAll('.popular-tab'),
    genreFilter: document.getElementById('genreFilter'),
    popularGrid: document.getElementById('popularGrid'),

    // User Page
    userIdInput: document.getElementById('userIdInput'),
    getUserRecsBtn: document.getElementById('getUserRecsBtn'),
    demoUserBtns: document.querySelectorAll('.demo-user-btn'),
    userStrategyInfo: document.getElementById('userStrategyInfo'),
    strategyBadge: document.getElementById('strategyBadge'),
    coldStartNotice: document.getElementById('coldStartNotice'),
    userRecsGrid: document.getElementById('userRecsGrid'),

    // Compare Page
    compareSearchInput: document.getElementById('compareSearchInput'),
    compareSearchBtn: document.getElementById('compareSearchBtn'),
    compareAutocomplete: document.getElementById('compareAutocomplete'),
    comparisonGrid: document.getElementById('comparisonGrid'),
    contentResults: document.getElementById('contentResults'),
    collaborativeResults: document.getElementById('collaborativeResults'),
    hybridResults: document.getElementById('hybridResults'),

    // Weights
    weightContent: document.getElementById('weightContent'),
    weightCollaborative: document.getElementById('weightCollaborative'),
    weightImplicit: document.getElementById('weightImplicit'),
    weightPopularity: document.getElementById('weightPopularity'),
    applyWeightsBtn: document.getElementById('applyWeightsBtn'),

    // UI Elements
    toast: document.getElementById('toast'),
    toastIcon: document.getElementById('toastIcon'),
    toastMessage: document.getElementById('toastMessage'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    backToSearchBtn: document.getElementById('backToSearchBtn')
};

// =============================================================================
// API Functions
// =============================================================================

async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

async function checkApiStatus() {
    try {
        const data = await apiRequest('/api/status');
        state.apiStatus = data;
        updateStatusIndicator(true, data.status === 'ready');
        return data;
    } catch (error) {
        updateStatusIndicator(false);
        return null;
    }
}

async function searchAnime(query, limit = 10) {
    const cacheKey = `${query}-${limit}`;
    if (state.cache.search[cacheKey]) {
        return state.cache.search[cacheKey];
    }

    const data = await apiRequest(`/api/search?q=${encodeURIComponent(query)}&top_k=${limit}`);
    state.cache.search[cacheKey] = data;
    return data;
}

async function getAutocomplete(query) {
    return await apiRequest(`/api/autocomplete?q=${encodeURIComponent(query)}&limit=8`);
}

async function getAnimeDetail(animeId) {
    return await apiRequest(`/api/anime/${animeId}`);
}

async function getRecommendations(animeId, method = 'hybrid', topK = 10) {
    const cacheKey = `${animeId}-${method}-${topK}`;
    if (state.cache.recommendations[cacheKey]) {
        return state.cache.recommendations[cacheKey];
    }

    const data = await apiRequest(`/api/recommend/anime/${animeId}?method=${method}&top_k=${topK}`);
    state.cache.recommendations[cacheKey] = data;
    return data;
}

async function getRecommendationsByName(animeName, method = 'hybrid', topK = 10) {
    return await apiRequest(`/api/recommend/anime/name/${encodeURIComponent(animeName)}?method=${method}&top_k=${topK}`);
}

async function getUserRecommendations(userId, topK = 10) {
    return await apiRequest(`/api/recommend/user/${userId}?top_k=${topK}`);
}

async function getPopular(type = 'top_rated', topK = 20, genre = null) {
    const cacheKey = `${type}-${topK}-${genre || 'all'}`;
    if (state.cache.popular[cacheKey]) {
        return state.cache.popular[cacheKey];
    }

    let url = `/api/popular?type=${type}&top_k=${topK}`;
    if (genre) url += `&genre=${encodeURIComponent(genre)}`;

    const data = await apiRequest(url);
    state.cache.popular[cacheKey] = data;
    return data;
}

async function getGenres() {
    return await apiRequest('/api/genres');
}

async function compareRecommendations(animeId, topK = 5) {
    return await apiRequest(`/api/compare?anime_id=${animeId}&top_k=${topK}`);
}

async function updateWeights(weights) {
    return await apiRequest('/api/weights', {
        method: 'PUT',
        body: JSON.stringify(weights)
    });
}

// =============================================================================
// UI Functions
// =============================================================================

function updateStatusIndicator(connected, modelLoaded = false) {
    const dot = elements.apiStatus.querySelector('.status-dot');
    const text = elements.apiStatus.querySelector('.status-text');

    dot.classList.remove('connected', 'error');

    if (!connected) {
        dot.classList.add('error');
        text.textContent = 'Disconnected';
    } else if (modelLoaded) {
        dot.classList.add('connected');
        text.textContent = 'Ready';
    } else {
        text.textContent = 'No Model';
    }
}

function showToast(message, type = 'info') {
    const labels = {
        info: '[Info]',
        success: '[Success]',
        error: '[Error]',
        warning: '[Warning]'
    };

    elements.toastIcon.textContent = labels[type] || labels.info;
    elements.toastMessage.textContent = message;
    elements.toast.classList.remove('hidden');

    setTimeout(() => {
        elements.toast.classList.add('hidden');
    }, 3000);
}

function showLoading(show = true) {
    state.isLoading = show;
    if (show) {
        elements.loadingOverlay.classList.remove('hidden');
    } else {
        elements.loadingOverlay.classList.add('hidden');
    }
}

function switchPage(pageName) {
    state.currentPage = pageName;

    elements.navLinks.forEach(link => {
        link.classList.toggle('active', link.dataset.page === pageName);
    });

    elements.pages.forEach(page => {
        page.classList.toggle('active', page.id === `page-${pageName}`);
    });

    // Load page-specific data
    if (pageName === 'popular') {
        loadPopularPage();
    }
}

function getImageUrl(anime) {
    // Check cache first
    if (state.cache.images[anime.mal_id]) {
        return state.cache.images[anime.mal_id];
    }

    if (anime.image_url && !anime.image_url.includes('undefined')) {
        return anime.image_url;
    }
    // Return placeholder, will be replaced when Jikan fetches
    return PLACEHOLDER_IMAGE;
}

// Fetch image from Jikan API
async function fetchAnimeImageFromJikan(malId) {
    // Check cache
    if (state.cache.images[malId]) {
        return state.cache.images[malId];
    }

    try {
        const response = await fetch(`${JIKAN_API}/anime/${malId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        const imageUrl = data.data?.images?.jpg?.large_image_url ||
                        data.data?.images?.jpg?.image_url ||
                        PLACEHOLDER_IMAGE;

        // Cache the result
        state.cache.images[malId] = imageUrl;
        return imageUrl;
    } catch (error) {
        console.warn(`Failed to fetch image for anime ${malId}:`, error);
        return PLACEHOLDER_IMAGE;
    }
}

// Process image queue with rate limiting (Jikan API: 3 req/sec)
async function processImageQueue() {
    if (state.isFetchingImages || state.imageQueue.length === 0) {
        return;
    }

    state.isFetchingImages = true;

    while (state.imageQueue.length > 0) {
        const { malId, imgElement } = state.imageQueue.shift();

        // Skip if already cached
        if (state.cache.images[malId]) {
            if (imgElement) {
                imgElement.src = state.cache.images[malId];
            }
            continue;
        }

        try {
            const imageUrl = await fetchAnimeImageFromJikan(malId);
            if (imgElement && imgElement.isConnected) {
                imgElement.src = imageUrl;
            }
            // Update all images with same mal_id
            document.querySelectorAll(`img[data-mal-id="${malId}"]`).forEach(img => {
                img.src = imageUrl;
            });
        } catch (error) {
            console.warn(`Error fetching image for ${malId}:`, error);
        }

        // Rate limiting: wait 350ms between requests (≈3 req/sec)
        await new Promise(resolve => setTimeout(resolve, 350));
    }

    state.isFetchingImages = false;
}

// Queue anime for image fetching
function queueImageFetch(malId, imgElement = null) {
    // Don't queue if already cached
    if (state.cache.images[malId]) {
        if (imgElement) {
            imgElement.src = state.cache.images[malId];
        }
        return;
    }

    // Check if already in queue
    const exists = state.imageQueue.some(item => item.malId === malId);
    if (!exists) {
        state.imageQueue.push({ malId, imgElement });
        processImageQueue();
    }
}

// Batch fetch images for multiple anime
function fetchImagesForAnimeList(animeList) {
    animeList.forEach(anime => {
        if (anime.mal_id && !state.cache.images[anime.mal_id]) {
            queueImageFetch(anime.mal_id);
        }
    });
}

function handleImageError(img) {
    img.onerror = null;
    img.src = PLACEHOLDER_IMAGE;
}

// =============================================================================
// Render Functions
// =============================================================================

function renderAnimeCard(anime, rank = null) {
    const imageUrl = getImageUrl(anime);
    const score = anime.score ? anime.score.toFixed(1) : 'N/A';
    const similarity = anime.similarity ? (anime.similarity * 100).toFixed(0) : null;
    const hybridScore = anime.hybrid_score ? (anime.hybrid_score * 100).toFixed(0) : null;

    // Queue image fetch from Jikan if not cached
    if (!state.cache.images[anime.mal_id]) {
        queueImageFetch(anime.mal_id);
    }

    return `
        <div class="anime-card" data-mal-id="${anime.mal_id}" onclick="handleAnimeClick(${anime.mal_id})">
            <div class="anime-card-image">
                <img src="${imageUrl}" alt="${anime.name}" data-mal-id="${anime.mal_id}" onerror="handleImageError(this)">
                ${rank ? `<span class="anime-card-rank">${rank}</span>` : ''}
                <span class="anime-card-score">${score}</span>
                ${similarity ? `<span class="anime-card-similarity">${similarity}% match</span>` : ''}
                ${hybridScore && !similarity ? `<span class="anime-card-similarity">${hybridScore}%</span>` : ''}
            </div>
            <div class="anime-card-info">
                <h3 class="anime-card-title">${anime.name}</h3>
                <div class="anime-card-meta">
                    <span>${anime.type || 'Unknown'}</span>
                </div>
                <div class="anime-card-genres">${anime.genres || ''}</div>
            </div>
        </div>
    `;
}

function renderAutocompleteItem(anime) {
    const imageUrl = getImageUrl(anime);
    const score = anime.score ? anime.score.toFixed(1) : 'N/A';

    // Queue image fetch
    if (!state.cache.images[anime.mal_id]) {
        queueImageFetch(anime.mal_id);
    }

    return `
        <div class="autocomplete-item" data-mal-id="${anime.mal_id}" data-name="${anime.name}">
            <img src="${imageUrl}" alt="${anime.name}" data-mal-id="${anime.mal_id}" onerror="handleImageError(this)">
            <div class="autocomplete-item-info">
                <div class="autocomplete-item-name">${anime.name}</div>
                <div class="autocomplete-item-meta">
                    <span>${score}</span>
                    <span>${anime.type || 'Unknown'}</span>
                </div>
            </div>
        </div>
    `;
}

async function renderSelectedAnime(anime) {
    let imageUrl = getImageUrl(anime);

    // Fetch image from Jikan if not cached
    if (!state.cache.images[anime.mal_id]) {
        imageUrl = await fetchAnimeImageFromJikan(anime.mal_id);
    } else {
        imageUrl = state.cache.images[anime.mal_id];
    }

    const imgElement = document.getElementById('selectedAnimeImage');
    imgElement.src = imageUrl;
    imgElement.dataset.malId = anime.mal_id;
    imgElement.onerror = function() { handleImageError(this); };

    document.getElementById('selectedAnimeName').textContent = anime.name;
    document.getElementById('selectedAnimeEnglish').textContent = anime.english_name || anime.name;

    const scoreEl = document.getElementById('selectedAnimeScore');
    scoreEl.querySelector('.badge-value').textContent = anime.score ? anime.score.toFixed(1) : 'N/A';

    document.getElementById('selectedAnimeType').textContent = anime.type || 'Unknown';

    // Genres
    const genresEl = document.getElementById('selectedAnimeGenres');
    if (anime.genres) {
        const genres = anime.genres.split(',').map(g => `<span class="genre-tag">${g.trim()}</span>`).join('');
        genresEl.innerHTML = genres;
    } else {
        genresEl.innerHTML = '<span class="genre-tag">Unknown</span>';
    }

    // Synopsis
    document.getElementById('selectedAnimeSynopsis').textContent = anime.synopsis || 'No synopsis available.';

    // MAL Link
    document.getElementById('selectedAnimeMAL').href = `https://myanimelist.net/anime/${anime.mal_id}`;

    elements.selectedAnimeCard.classList.remove('hidden');
    elements.controlsSection.classList.remove('hidden');
}

function renderRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        elements.recommendationsGrid.innerHTML = '<p class="no-results">No recommendations found.</p>';
        elements.resultsCount.textContent = '0 results';
        return;
    }

    const html = recommendations.map((anime, idx) => renderAnimeCard(anime, idx + 1)).join('');
    elements.recommendationsGrid.innerHTML = html;
    elements.resultsCount.textContent = `${recommendations.length} results`;
    elements.recommendationsSection.classList.remove('hidden');
}

function renderPopularGrid(animeList) {
    if (!animeList || animeList.length === 0) {
        elements.popularGrid.innerHTML = '<p class="no-results">No anime found.</p>';
        return;
    }

    const html = animeList.map((anime, idx) => renderAnimeCard(anime, idx + 1)).join('');
    elements.popularGrid.innerHTML = html;
}

function renderUserRecommendations(data) {
    // Update strategy info
    elements.userStrategyInfo.classList.remove('hidden');
    elements.strategyBadge.querySelector('.strategy-text').textContent = `Strategy: ${data.strategy}`;

    if (data.is_cold_start) {
        elements.coldStartNotice.classList.remove('hidden');
    } else {
        elements.coldStartNotice.classList.add('hidden');
    }

    // Render grid
    if (!data.recommendations || data.recommendations.length === 0) {
        elements.userRecsGrid.innerHTML = '<p class="no-results">No recommendations found.</p>';
        return;
    }

    const html = data.recommendations.map((anime, idx) => renderAnimeCard(anime, idx + 1)).join('');
    elements.userRecsGrid.innerHTML = html;
}

function renderCompareItem(anime) {
    const imageUrl = getImageUrl(anime);
    const score = anime.score ? anime.score.toFixed(1) : 'N/A';
    const similarity = anime.similarity ? (anime.similarity * 100).toFixed(0) :
                       anime.hybrid_score ? (anime.hybrid_score * 100).toFixed(0) : 'N/A';

    // Queue image fetch
    if (!state.cache.images[anime.mal_id]) {
        queueImageFetch(anime.mal_id);
    }

    return `
        <div class="compare-item">
            <img src="${imageUrl}" alt="${anime.name}" data-mal-id="${anime.mal_id}" onerror="handleImageError(this)">
            <div class="compare-item-info">
                <div class="compare-item-name">${anime.name}</div>
                <div class="compare-item-score">${score} | ${similarity}% match</div>
            </div>
        </div>
    `;
}

function renderComparison(results) {
    elements.comparisonGrid.classList.remove('hidden');

    // Content results
    if (results.content && results.content.length > 0) {
        elements.contentResults.innerHTML = results.content.map(renderCompareItem).join('');
    } else {
        elements.contentResults.innerHTML = '<p class="no-results">No results</p>';
    }

    // Collaborative results
    if (results.collaborative && results.collaborative.length > 0) {
        elements.collaborativeResults.innerHTML = results.collaborative.map(renderCompareItem).join('');
    } else {
        elements.collaborativeResults.innerHTML = '<p class="no-results">No results</p>';
    }

    // Hybrid results
    if (results.hybrid && results.hybrid.length > 0) {
        elements.hybridResults.innerHTML = results.hybrid.map(renderCompareItem).join('');
    } else {
        elements.hybridResults.innerHTML = '<p class="no-results">No results</p>';
    }
}

// =============================================================================
// Event Handlers
// =============================================================================

function handleAnimeClick(malId) {
    // Could navigate to detail page or show modal
    window.open(`https://myanimelist.net/anime/${malId}`, '_blank');
}

async function handleSearch() {
    const query = elements.searchInput.value.trim();
    if (!query) {
        showToast('Please enter an anime name', 'warning');
        return;
    }

    showLoading(true);
    try {
        const data = await searchAnime(query, 1);
        if (data.results && data.results.length > 0) {
            state.selectedAnime = data.results[0];
            await renderSelectedAnime(state.selectedAnime);
            elements.autocompleteDropdown.classList.remove('visible');
        } else {
            showToast('No anime found', 'warning');
        }
    } catch (error) {
        showToast(`Search failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function handleAutocomplete() {
    const query = elements.searchInput.value.trim();
    if (query.length < 2) {
        elements.autocompleteDropdown.classList.remove('visible');
        return;
    }

    try {
        const data = await getAutocomplete(query);
        if (data.suggestions && data.suggestions.length > 0) {
            const html = data.suggestions.map(renderAutocompleteItem).join('');
            elements.autocompleteDropdown.innerHTML = html;
            elements.autocompleteDropdown.classList.add('visible');

            // Add click handlers
            elements.autocompleteDropdown.querySelectorAll('.autocomplete-item').forEach(item => {
                item.addEventListener('click', () => {
                    const malId = parseInt(item.dataset.malId);
                    const name = item.dataset.name;
                    elements.searchInput.value = name;
                    elements.autocompleteDropdown.classList.remove('visible');
                    selectAnimeById(malId);
                });
            });
        } else {
            elements.autocompleteDropdown.classList.remove('visible');
        }
    } catch (error) {
        console.error('Autocomplete error:', error);
    }
}

async function selectAnimeById(malId) {
    showLoading(true);
    try {
        const data = await getAnimeDetail(malId);
        if (data.anime) {
            state.selectedAnime = data.anime;
            await renderSelectedAnime(state.selectedAnime);
        }
    } catch (error) {
        showToast(`Failed to load anime: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function handleGetRecommendations() {
    if (!state.selectedAnime) {
        showToast('Please select an anime first', 'warning');
        return;
    }

    const topK = parseInt(elements.topkInput.value) || 10;

    showLoading(true);
    try {
        const data = await getRecommendations(state.selectedAnime.mal_id, state.selectedMethod, topK);
        if (data.recommendations) {
            renderRecommendations(data.recommendations);
            showToast(`Found ${data.count} recommendations`, 'success');
        }
    } catch (error) {
        showToast(`Failed to get recommendations: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function handleQuickTag(animeName) {
    elements.searchInput.value = animeName;
    showLoading(true);
    try {
        const data = await searchAnime(animeName, 1);
        if (data.results && data.results.length > 0) {
            state.selectedAnime = data.results[0];
            await renderSelectedAnime(state.selectedAnime);
        }
    } catch (error) {
        showToast(`Failed to find ${animeName}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function loadPopularPage() {
    const activeTab = document.querySelector('.popular-tab.active');
    const type = activeTab ? activeTab.dataset.type : 'top_rated';
    const genre = elements.genreFilter.value || null;

    elements.popularGrid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>Loading...</p></div>';

    try {
        const data = await getPopular(type, 20, genre);
        renderPopularGrid(data.anime);
    } catch (error) {
        elements.popularGrid.innerHTML = `<p class="no-results">Failed to load: ${error.message}</p>`;
    }
}

async function loadGenres() {
    try {
        const data = await getGenres();
        if (data.genres) {
            const options = data.genres.map(g => `<option value="${g}">${g}</option>`).join('');
            elements.genreFilter.innerHTML = '<option value="">All Genres</option>' + options;
        }
    } catch (error) {
        console.error('Failed to load genres:', error);
    }
}

async function handleUserRecommendations() {
    const userId = parseInt(elements.userIdInput.value);
    if (!userId || userId < 1) {
        showToast('Please enter a valid user ID', 'warning');
        return;
    }

    showLoading(true);
    try {
        const data = await getUserRecommendations(userId);
        renderUserRecommendations(data);
        showToast(`Found ${data.count} recommendations for user ${userId}`, 'success');
    } catch (error) {
        showToast(`Failed to get recommendations: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function handleCompare() {
    const query = elements.compareSearchInput.value.trim();
    if (!query) {
        showToast('Please enter an anime name', 'warning');
        return;
    }

    showLoading(true);
    try {
        // First search for the anime
        const searchData = await searchAnime(query, 1);
        if (!searchData.results || searchData.results.length === 0) {
            showToast('Anime not found', 'warning');
            return;
        }

        const anime = searchData.results[0];
        const data = await compareRecommendations(anime.mal_id);

        if (data.results) {
            renderComparison(data.results);
            showToast('Comparison loaded', 'success');
        }
    } catch (error) {
        showToast(`Comparison failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function handleApplyWeights() {
    const weights = {
        content: parseFloat(elements.weightContent.value) / 100,
        collaborative: parseFloat(elements.weightCollaborative.value) / 100,
        implicit: parseFloat(elements.weightImplicit.value) / 100,
        popularity: parseFloat(elements.weightPopularity.value) / 100
    };

    showLoading(true);
    try {
        await updateWeights(weights);
        // Clear cache since weights changed
        state.cache.recommendations = {};
        showToast('Weights updated successfully', 'success');
    } catch (error) {
        showToast(`Failed to update weights: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// =============================================================================
// Event Listeners
// =============================================================================

function initEventListeners() {
    // Navigation
    elements.navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchPage(link.dataset.page);
        });
    });

    // Search
    elements.searchInput.addEventListener('input', debounce(handleAutocomplete, 300));
    elements.searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });
    elements.searchBtn.addEventListener('click', handleSearch);

    // Close autocomplete on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-wrapper')) {
            elements.autocompleteDropdown.classList.remove('visible');
        }
        if (!e.target.closest('.compare-input-section')) {
            elements.compareAutocomplete.classList.remove('visible');
        }
    });

    // Quick tags
    elements.quickTags.forEach(tag => {
        tag.addEventListener('click', () => handleQuickTag(tag.dataset.anime));
    });

    // Method selector
    elements.methodBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.methodBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.selectedMethod = btn.dataset.method;
        });
    });

    // Get recommendations
    elements.getRecommendationsBtn.addEventListener('click', handleGetRecommendations);

    // Popular tabs
    elements.popularTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            elements.popularTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            loadPopularPage();
        });
    });

    // Genre filter
    elements.genreFilter.addEventListener('change', loadPopularPage);

    // User recommendations
    elements.getUserRecsBtn.addEventListener('click', handleUserRecommendations);
    elements.userIdInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleUserRecommendations();
    });

    // Demo users
    elements.demoUserBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.userIdInput.value = btn.dataset.user;
            handleUserRecommendations();
        });
    });

    // Compare
    elements.compareSearchBtn.addEventListener('click', handleCompare);
    elements.compareSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleCompare();
    });

    // Weights
    const weightSliders = [elements.weightContent, elements.weightCollaborative,
                          elements.weightImplicit, elements.weightPopularity];
    weightSliders.forEach(slider => {
        slider.addEventListener('input', () => {
            const valueEl = document.getElementById(`${slider.id}Value`);
            if (valueEl) {
                valueEl.textContent = (slider.value / 100).toFixed(2);
            }
        });
    });
    elements.applyWeightsBtn.addEventListener('click', handleApplyWeights);

    // Back button
    if (elements.backToSearchBtn) {
        elements.backToSearchBtn.addEventListener('click', () => switchPage('home'));
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// =============================================================================
// Initialization
// =============================================================================

async function init() {
    console.log('🎌 Anime Recommender initializing...');

    // Check API status
    await checkApiStatus();

    // Load genres for filter
    await loadGenres();

    // Initialize event listeners
    initEventListeners();

    console.log('✅ Anime Recommender ready!');
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
