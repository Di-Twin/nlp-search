import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './FoodSearch.css';

const FoodSearch = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [meta, setMeta] = useState(null);

    // Debounced search function
    const debouncedSearch = useCallback(
        async (searchQuery) => {
            if (searchQuery.length < 3) {
                setResults([]);
                setMeta(null);
                return;
            }

            setLoading(true);
            setError(null);

            try {
                const response = await axios.post('http://localhost:8000/api/search', {
                    query: searchQuery,
                    limit: 8
                }, {
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Client-Id': 'abc1'
                    }
                });

                setResults(response.data.results);
                setMeta(response.data.meta);
            } catch (err) {
                setError(err.response?.data?.detail || 'An error occurred while searching');
            } finally {
                setLoading(false);
            }
        },
        []
    );

    // Effect for auto-search
    useEffect(() => {
        const timer = setTimeout(() => {
            debouncedSearch(query);
        }, 300); // 300ms delay

        return () => clearTimeout(timer);
    }, [query, debouncedSearch]);

    const handleSearch = (e) => {
        e.preventDefault();
        debouncedSearch(query);
    };

    return (
        <div className="food-search-container">
            <div className="search-header">
                <h1>Food Search</h1>
                <p className="subtitle">Discover delicious and nutritious food items</p>
            </div>

            <form onSubmit={handleSearch} className="search-form">
                <div className="search-input-wrapper">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search for food items... (min. 3 characters)"
                        className="search-input"
                        aria-label="Search food items"
                    />
                    <button 
                        type="submit" 
                        className="search-button"
                        disabled={loading || !query.trim()}
                        aria-label="Search"
                    >
                        {loading ? (
                            <span className="loading-spinner"></span>
                        ) : (
                            <svg className="search-icon" viewBox="0 0 24 24">
                                <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                            </svg>
                        )}
                    </button>
                </div>
            </form>

            {error && (
                <div className="error-message" role="alert">
                    <svg className="error-icon" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                    </svg>
                    {error}
                </div>
            )}

            {meta && (
                <div className="search-meta">
                    <div className="meta-item">
                        <span className="meta-label">Results:</span>
                        <span className="meta-value">{meta.total_results}</span>
                    </div>
                    <div className="meta-item">
                        <span className="meta-label">Time:</span>
                        <span className="meta-value">{meta.duration_ms.toFixed(2)}ms</span>
                    </div>
                    <div className="meta-item">
                        <span className="meta-label">Cache:</span>
                        <span className={`meta-value cache-${meta.cache.toLowerCase()}`}>
                            {meta.cache}
                        </span>
                    </div>
                </div>
            )}

            <div className="results-grid">
                {results.map((item) => (
                    <div key={item.id} className="food-card">
                        <div className="food-image">
                            <img 
                                src={item.image_url || '/placeholder-food.jpg'} 
                                alt={item.food_name}
                                loading="lazy"
                            />
                        </div>
                        <div className="food-content">
                            <h3 className="food-name">{item.food_name}</h3>
                            <p className="food-description">{item.description}</p>
                            
                            {item.nutrition && (
                                <div className="nutrition-info">
                                    <div className="nutrition-item">
                                        <span className="nutrition-label">Calories</span>
                                        <span className="nutrition-value">{item.nutrition.energy_kcal} kcal</span>
                                    </div>
                                    <div className="nutrition-item">
                                        <span className="nutrition-label">Protein</span>
                                        <span className="nutrition-value">{item.nutrition.protein_g}g</span>
                                    </div>
                                    <div className="nutrition-item">
                                        <span className="nutrition-label">Carbs</span>
                                        <span className="nutrition-value">{item.nutrition.carbohydrates_g}g</span>
                                    </div>
                                    <div className="nutrition-item">
                                        <span className="nutrition-label">Fat</span>
                                        <span className="nutrition-value">{item.nutrition.fat_g}g</span>
                                    </div>
                                </div>
                            )}

                            {item.scores && (
                                <div className="match-score">
                                    <div className="score-bar">
                                        <div 
                                            className="score-fill"
                                            style={{ width: `${item.scores.combined * 100}%` }}
                                        />
                                    </div>
                                    <span className="score-text">
                                        Match: {Math.round(item.scores.combined * 100)}%
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default FoodSearch; 