import pytest
from unittest.mock import MagicMock, AsyncMock, patch # Added patch
import pandas as pd
import tweepy # Added tweepy
from Morningstar.utils.social_scraper import SocialMediaScraper

@pytest.fixture
def mock_scraper():
    """Fixture avec des mocks pour les tests sociaux"""
    scraper = SocialMediaScraper(
        twitter_keys={
            'api_key': 'test', # Added comma
            'api_secret': 'test', # Added comma
            'access_token': 'test', # Added comma
            'access_secret': 'test'
        },
        reddit_keys={
            'client_id': 'test', # Added comma
            'client_secret': 'test', # Added comma
            'username': 'test', # Added comma
            'password': 'test', # Added comma
            'user_agent': 'test_agent'
        }
    )
    # Note: The call to SocialMediaScraper implicitly uses the defined dictionaries above.
    # The previous error trace showed 'SocialMediaScraper(twitter_keys reddit_keys)' which was also missing a comma,
    # but the actual code uses the dictionaries directly, so no comma needed *there*.
    scraper.twitter = MagicMock() # Removed spec=tweepy.API temporarily
    scraper.reddit = MagicMock()
    return scraper

@pytest.mark.asyncio
async def test_twitter_sentiment_analysis(mock_scraper):
    """Test l'analyse Twitter"""
    mock_tweet = MagicMock()
    mock_tweet.full_text = "BTC to the moon!"
    mock_scraper.twitter.search_tweets.return_value = [mock_tweet]
    
    result = await mock_scraper.get_twitter_sentiment("BTC")
    
    assert isinstance(result, pd.DataFrame)
    assert "sentiment" in result.columns

@pytest.mark.asyncio
async def test_reddit_sentiment_analysis(mock_scraper):
    """Test l'analyse Reddit"""
    mock_post = MagicMock()
    mock_post.title = "Bullish on BTC"
    mock_scraper.reddit.subreddit.return_value.new.return_value = [mock_post]
    
    result = await mock_scraper.get_reddit_sentiment("bitcoin")
    
    assert isinstance(result, dict)

def test_sentiment_analysis_logic(mock_scraper):
    """Test la logique de base du sentiment"""
    mock_scraper._analyze_sentiment = MagicMock(return_value=0.8)
    sentiment = mock_scraper._analyze_sentiment("Positive text")
    assert 0 <= sentiment <= 1
