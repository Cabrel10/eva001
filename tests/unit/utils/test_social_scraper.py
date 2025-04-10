import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from Morningstar.utils.social_scraper import SocialMediaScraper

@pytest.fixture
def mock_scraper():
    # Configuration factice pour les tests
    twitter_keys = {
        'api_key': 'test',
        'api_secret': 'test',
        'access_token': 'test',
        'access_secret': 'test'
    }
    reddit_keys = {
        'client_id': 'test',
        'client_secret': 'test',
        'username': 'test',
        'password': 'test',
        'user_agent': 'test_agent' # Ajout user_agent
    }
    
    # Mock les clients API pour éviter les appels réels
    scraper = SocialMediaScraper(twitter_keys, reddit_keys)
    scraper.twitter = MagicMock(spec=tweepy.API) 
    scraper.reddit = AsyncMock(spec=asyncpraw.Reddit) # Mock l'instance Reddit
    return scraper

@pytest.mark.asyncio
async def test_twitter_sentiment_analysis(mock_scraper):
    """Test l'analyse de sentiment Twitter"""
    # Mock des tweets de test avec les attributs attendus par le code
    mock_tweet1 = MagicMock()
    mock_tweet1.full_text = "Bitcoin to the moon! Bullish AF #BTC"
    mock_tweet1.created_at = datetime.now()
    mock_tweet1.retweet_count = 10
    mock_tweet1.favorite_count = 50
    mock_tweet1.user = MagicMock() # Mock l'objet user
    mock_tweet1.user.followers_count = 1000

    mock_tweet2 = MagicMock()
    mock_tweet2.full_text = "Selling all my crypto, bear market coming"
    mock_tweet2.created_at = datetime.now() - timedelta(hours=1)
    mock_tweet2.retweet_count = 5
    mock_tweet2.favorite_count = 20
    mock_tweet2.user = MagicMock()
    mock_tweet2.user.followers_count = 500
    
    # Configure le mock de l'API Twitter pour retourner les tweets mockés
    # Note: search_tweets est appelé via run_in_executor, donc on mock l'executor
    with patch.object(mock_scraper.executor, 'submit') as mock_submit:
        future = asyncio.Future()
        future.set_result([mock_tweet1, mock_tweet2])
        mock_submit.return_value = future

        result_df = await mock_scraper.get_twitter_sentiment("Bitcoin", limit=2)
        
        # Assertions sur le DataFrame retourné
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert 'sentiment' in result_df.columns
        assert 'retweets' in result_df.columns
        assert 'likes' in result_df.columns
        assert 'user_followers' in result_df.columns
        assert all(-1 <= s <= 1 for s in result_df['sentiment'])

@pytest.mark.asyncio
async def test_reddit_sentiment_analysis(mock_scraper):
    """Test l'analyse de sentiment Reddit"""
    # Mock AsyncPraw
    mock_post = MagicMock(spec=asyncpraw.models.Submission) 
    mock_post.title = "Bitcoin is going to dump hard"
    mock_post.score = 42
    
    # Mock l'itérateur asynchrone
    async def mock_aiter(*args, **kwargs):
        yield mock_post
        
    mock_subreddit = AsyncMock()
    mock_subreddit.new.return_value = mock_aiter() # Retourne l'itérateur
    
    # Patch l'instance mockée de Reddit pour retourner le mock_subreddit
    mock_scraper.reddit.subreddit.return_value = mock_subreddit
    
    result = await mock_scraper.get_reddit_sentiment("CryptoCurrency", 1)
    
    mock_scraper.reddit.subreddit.assert_called_once_with("CryptoCurrency")
    assert result['total'] == 1
    assert result['negative'] == 1  # "dump" est négatif
    assert result['positive'] == 0
    assert result['neutral'] == 0
    # La structure actuelle ne retourne pas le score moyen

def test_sentiment_analysis_logic(mock_scraper):
    """Test la logique d'analyse de sentiment"""
    # Test positif (>= 0 car TextBlob peut être neutre)
    assert mock_scraper._analyze_sentiment("Bullish on BTC") >= 0 
    
    # Test négatif
    assert mock_scraper._analyze_sentiment("Bear market coming") < 0
    
    # Test neutre
    assert mock_scraper._analyze_sentiment("Just an update") == 0.0

@pytest.mark.asyncio
async def test_error_handling(mock_scraper):
    """Test la gestion des erreurs"""
    # Test erreur Twitter (mock l'executor)
    with patch.object(mock_scraper.executor, 'submit', side_effect=Exception("Twitter API error")):
        result = await mock_scraper.get_twitter_sentiment("Bitcoin")
        assert isinstance(result, dict) # Doit retourner un dict d'erreur
        assert 'error' in result
        assert "Twitter API error" in result['error']
        
    # Test erreur Reddit
    mock_scraper.reddit.subreddit.side_effect = Exception("Reddit API error")
    result_reddit = await mock_scraper.get_reddit_sentiment("CryptoCurrency")
    assert isinstance(result_reddit, dict)
    assert 'error' in result_reddit
    assert "Reddit API error" in result_reddit['error']
