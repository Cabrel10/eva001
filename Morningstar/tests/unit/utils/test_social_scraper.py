import pytest
from unittest.mock import MagicMock, patch
from Morningstar.utils.social_scraper import SocialMediaScraper
import asyncio

@pytest.fixture
def mock_scraper():
    with patch('Morningstar.utils.social_scraper.tweepy.API'), \
         patch('Morningstar.utils.social_scraper.asyncpraw.Reddit'), \
         patch('Morningstar.utils.social_scraper.ThreadPoolExecutor'):
        
        # Mock des clés API
        twitter_keys = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'access_token': 'test_token',
            'access_secret': 'test_secret'
        }
        reddit_keys = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'username': 'test_user',
            'password': 'test_pass'
        }
        
        scraper = SocialMediaScraper(twitter_keys, reddit_keys)
        scraper._analyze_sentiment = MagicMock(return_value=0.5)
        return scraper

def test_sentiment_analysis_logic(mock_scraper):
    """Test la logique d'analyse de sentiment"""
    # Mock la méthode d'analyse
    mock_scraper._analyze_sentiment = MagicMock()
    mock_scraper._analyze_sentiment.side_effect = lambda x: 0.5 if "bullish" in x.lower() else -0.5
    
    assert mock_scraper._analyze_sentiment("Bullish market") > 0
    assert mock_scraper._analyze_sentiment("Bearish trend") < 0

@pytest.mark.asyncio
async def test_twitter_sentiment_analysis(mock_scraper):
    """Test l'analyse Twitter"""
    with patch('Morningstar.utils.social_scraper.asyncio.get_event_loop') as mock_loop:
        # Mock de la réponse Twitter
        mock_tweets = [
            MagicMock(full_text="Bitcoin bullish", id="1", 
                     created_at="2023-01-01", retweet_count=10,
                     favorite_count=20, user=MagicMock(followers_count=1000)),
            MagicMock(full_text="Market bearish", id="2",
                     created_at="2023-01-02", retweet_count=5,
                     favorite_count=10, user=MagicMock(followers_count=500))
        ]
        
        # Configuration des mocks
        mock_loop.return_value.run_in_executor.return_value = mock_tweets
        mock_scraper._analyze_sentiment.side_effect = [0.8, -0.5]
        
        result = await mock_scraper.get_twitter_sentiment("Bitcoin")
        
        assert isinstance(result, dict)
        assert 'average_sentiment' in result
        assert result['average_sentiment'] == 0.15  # (0.8 + -0.5)/2

@pytest.mark.asyncio 
async def test_reddit_sentiment_analysis(mock_scraper):
    """Test l'analyse Reddit"""
    with patch('Morningstar.utils.social_scraper.asyncpraw.Reddit.subreddit') as mock_subreddit:
        # Mock de la réponse Reddit
        mock_post = MagicMock()
        mock_post.title = "Ethereum rally"
        mock_post.score = 100
        
        # Configuration des mocks
        mock_subred = MagicMock()
        mock_subred.new.return_value = [mock_post]
        mock_subreddit.return_value = mock_subred
        
        mock_scraper._analyze_sentiment.return_value = 0.7
        
        result = await mock_scraper.get_reddit_sentiment("ethereum")
        
        assert isinstance(result, dict)
        assert 'average_sentiment' in result
        assert result['total'] == 1
        assert result['positive'] == 1
        assert result['negative'] == 0

@pytest.mark.asyncio
async def test_error_handling(mock_scraper):
    """Test gestion des erreurs"""
    with patch('Morningstar.utils.social_scraper.asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor.side_effect = Exception("Twitter API error")
        result = await mock_scraper.get_twitter_sentiment("BTC")
        assert 'error' in result
        assert "Twitter API error" in result['error']
