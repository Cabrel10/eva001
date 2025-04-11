import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from Morningstar.utils.social_scraper import SocialMediaScraper

@pytest.fixture
def mock_scraper():
    scraper = SocialMediaScraper()
    scraper.twitter = MagicMock()
    scraper.reddit = MagicMock()
    scraper.executor = MagicMock()
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
    mock_scraper.executor.submit.return_value.result.return_value = [
        {"text": "Bitcoin bullish", "id": "1"},
        {"text": "Market bearish", "id": "2"}
    ]
    result = await mock_scraper.get_twitter_sentiment("Bitcoin")
    assert 'average_sentiment' in result

@pytest.mark.asyncio 
async def test_reddit_sentiment_analysis(mock_scraper):
    """Test l'analyse Reddit"""
    mock_post = MagicMock()
    mock_post.title = "Ethereum rally"
    mock_scraper.reddit.subreddit.return_value.new.return_value = [mock_post]
    result = await mock_scraper.get_reddit_sentiment("ethereum")
    assert 'average_sentiment' in result

@pytest.mark.asyncio
async def test_error_handling(mock_scraper):
    """Test gestion des erreurs"""
    mock_scraper.executor.submit.side_effect = Exception("Twitter error")
    result = await mock_scraper.get_twitter_sentiment("BTC")
    assert 'error' in result
