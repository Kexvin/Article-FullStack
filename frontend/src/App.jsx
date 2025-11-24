import React, { useState, useEffect } from 'react';
import { Trash2, TrendingUp, TrendingDown, Minus, Plus, BarChart3 } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function SentimentAnalysisApp() {
  const [articles, setArticles] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [selectedArticle, setSelectedArticle] = useState(null);

  useEffect(() => {
    fetchArticles();
    fetchStats();
  }, []);

  const fetchArticles = async () => {
    try {
      const response = await fetch(`${API_BASE}/articles`);
      const data = await response.json();
      setArticles(data);
    } catch (err) {
      setError('Failed to fetch articles');
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats');
    }
  };

  const handleSubmit = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE}/articles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title || null, text }),
      });

      if (!response.ok) throw new Error('Failed to analyze article');

      const newArticle = await response.json();
      
      await fetchArticles();
      await fetchStats();
      
      // Auto-select the newly created article
      setSelectedArticle(newArticle);
      
      setTitle('');
      setText('');
      setShowForm(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  const handleDelete = async (id) => {
    if (!window.confirm('Delete this article?')) return;

    try {
      await fetch(`${API_BASE}/articles/${id}`, { method: 'DELETE' });
      await fetchArticles();
      await fetchStats();
      if (selectedArticle?.id === id) setSelectedArticle(null);
    } catch (err) {
      setError('Failed to delete article');
    }
  };

  const viewArticle = async (id) => {
    try {
      const response = await fetch(`${API_BASE}/articles/${id}`);
      const data = await response.json();
      setSelectedArticle(data);
    } catch (err) {
      setError('Failed to fetch article details');
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'POSITIVE': return 'text-green-600 bg-green-50';
      case 'NEGATIVE': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'POSITIVE': return <TrendingUp className="w-5 h-5" />;
      case 'NEGATIVE': return <TrendingDown className="w-5 h-5" />;
      default: return <Minus className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                Financial Article Sentiment Analysis
              </h1>
              <p className="text-gray-600">
                Analyze sentiment in financial news articles using AI
              </p>
            </div>
            <button
              onClick={() => setShowForm(!showForm)}
              className="flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition"
            >
              <Plus className="w-5 h-5" />
              New Article
            </button>
          </div>
        </div>

        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center gap-3">
                <BarChart3 className="w-8 h-8 text-indigo-600" />
                <div>
                  <p className="text-gray-600 text-sm">Total Articles</p>
                  <p className="text-2xl font-bold text-gray-800">{stats.total_articles}</p>
                </div>
              </div>
            </div>
            <div className="bg-green-50 rounded-lg shadow p-6">
              <div className="flex items-center gap-3">
                <TrendingUp className="w-8 h-8 text-green-600" />
                <div>
                  <p className="text-gray-600 text-sm">Positive</p>
                  <p className="text-2xl font-bold text-green-600">{stats.positive}</p>
                </div>
              </div>
            </div>
            <div className="bg-gray-50 rounded-lg shadow p-6">
              <div className="flex items-center gap-3">
                <Minus className="w-8 h-8 text-gray-600" />
                <div>
                  <p className="text-gray-600 text-sm">Neutral</p>
                  <p className="text-2xl font-bold text-gray-600">{stats.neutral}</p>
                </div>
              </div>
            </div>
            <div className="bg-red-50 rounded-lg shadow p-6">
              <div className="flex items-center gap-3">
                <TrendingDown className="w-8 h-8 text-red-600" />
                <div>
                  <p className="text-gray-600 text-sm">Negative</p>
                  <p className="text-2xl font-bold text-red-600">{stats.negative}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {showForm && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              Add New Article
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Article Title (Optional)
                </label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="Enter article title..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Article Text *
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={8}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="Paste article text here..."
                />
              </div>
              {error && (
                <div className="bg-red-50 text-red-600 px-4 py-3 rounded-lg">
                  {error}
                </div>
              )}
              <div className="flex gap-3">
                <button
                  onClick={handleSubmit}
                  disabled={loading || !text.trim()}
                  className="flex-1 bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 transition"
                >
                  {loading ? 'Analyzing...' : 'Analyze Article'}
                </button>
                <button
                  onClick={() => setShowForm(false)}
                  className="px-6 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              Analyzed Articles ({articles.length})
            </h2>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {articles.map((article) => (
                <div
                  key={article.id}
                  onClick={() => viewArticle(article.id)}
                  className={`p-4 border rounded-lg cursor-pointer transition hover:shadow-md ${
                    selectedArticle?.id === article.id ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-800 mb-1">
                        {article.title || 'Untitled Article'}
                      </h3>
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(article.overall_sentiment)}`}>
                          {getSentimentIcon(article.overall_sentiment)}
                          {article.overall_sentiment}
                        </span>
                        <span className="text-sm text-gray-500">
                          Score: {article.sentiment_score.toFixed(2)}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500">
                        {new Date(article.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(article.id);
                      }}
                      className="text-red-500 hover:text-red-700 p-2"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
              {articles.length === 0 && (
                <p className="text-gray-500 text-center py-8">
                  No articles yet. Add one to get started!
                </p>
              )}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              Article Details
            </h2>
            {selectedArticle ? (
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-gray-700 mb-2">Title</h3>
                  <p className="text-gray-800">
                    {selectedArticle.title || 'Untitled'}
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-700 mb-2">Sentiment Analysis</h3>
                  <div className="flex items-center gap-3 mb-3">
                    <span className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-lg font-medium ${getSentimentColor(selectedArticle.overall_sentiment)}`}>
                      {getSentimentIcon(selectedArticle.overall_sentiment)}
                      {selectedArticle.overall_sentiment}
                    </span>
                    <span className="text-lg font-semibold text-gray-700">
                      Score: {selectedArticle.sentiment_score.toFixed(3)}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <div className="bg-green-50 p-2 rounded text-center">
                      <p className="font-semibold text-green-600">
                        {selectedArticle.positive_count}
                      </p>
                      <p className="text-gray-600">Positive</p>
                    </div>
                    <div className="bg-gray-50 p-2 rounded text-center">
                      <p className="font-semibold text-gray-600">
                        {selectedArticle.neutral_count}
                      </p>
                      <p className="text-gray-600">Neutral</p>
                    </div>
                    <div className="bg-red-50 p-2 rounded text-center">
                      <p className="font-semibold text-red-600">
                        {selectedArticle.negative_count}
                      </p>
                      <p className="text-gray-600">Negative</p>
                    </div>
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-700 mb-2">Article Text</h3>
                  <div className="bg-gray-50 p-4 rounded-lg max-h-64 overflow-y-auto">
                    <p className="text-gray-700 text-sm whitespace-pre-wrap">
                      {selectedArticle.text}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">
                Select an article to view details
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}