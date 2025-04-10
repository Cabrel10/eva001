import tweepy
import asyncpraw
import asyncio
import aiohttp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class SocialMediaScraper:
    """
    Scraper optimisé pour les réseaux sociaux avec :
    - Gestion des rate limits
    - Cache des résultats
    - Analyse de sentiment basique
    """
    def __init__(self, twitter_keys: Dict, reddit_keys: Dict):
        # Configuration Twitter
        self.twitter_auth = tweepy.OAuth1UserHandler(
            twitter_keys['api_key'],
            twitter_keys['api_secret'],
            twitter_keys['access_token'],
            twitter_keys['access_secret']
        )
        self.twitter = tweepy.API(self.twitter_auth, wait_on_rate_limit=True)
        
        # Configuration Reddit
        self.reddit = asyncpraw.Reddit(
            client_id=reddit_keys['client_id'],
            client_secret=reddit_keys['client_secret'],
            username=reddit_keys['username'],
            password=reddit_keys['password'],
            user_agent="MorningstarBot/1.0"
        )
        
        # Cache et taux limite
        self._cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def get_twitter_sentiment(self, query: str, 
                                  start_date: str = None,
                                  end_date: str = None,
                                  limit: int = 100) -> pd.DataFrame:
        """
        Analyse les tweets avec plage date et retourne un DataFrame
        """
        cache_key = f"twitter_{query}_{start_date}_{end_date}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            # Construction de la requête avec dates
            query_str = f"{query}"
            if start_date:
                query_str += f" since:{start_date}"
            if end_date:
                query_str += f" until:{end_date}"
                
            tweets = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.twitter.search_tweets(q=query_str, count=limit, tweet_mode='extended')
            )
            
            # Création d'un DataFrame avec toutes les métriques
            data = []
            for tweet in tweets:
                sentiment = self._analyze_sentiment(tweet.full_text)
                data.append({
                    'timestamp': tweet.created_at,
                    'text': tweet.full_text,
                    'sentiment': sentiment,
                    'retweets': tweet.retweet_count,
                    'likes': tweet.favorite_count,
                    'user_followers': tweet.user.followers_count
                })
            
            result = pd.DataFrame(data)
            result.set_index('timestamp', inplace=True)
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Twitter error: {e}")
            return {'error': str(e)}

    async def get_reddit_sentiment(self, subreddit: str, limit: int = 50) -> Dict:
        """
        Analyse les posts Reddit récents
        """
        cache_key = f"reddit_{subreddit}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            sub = await self.reddit.subreddit(subreddit)
            posts = []
            async for post in sub.new(limit=limit):
                posts.append({
                    'title': post.title,
                    'score': post.score,
                    'sentiment': self._analyze_sentiment(post.title)
                })
                
            # Agrégation des résultats
            positive = sum(1 for p in posts if p['sentiment'] > 0.5)
            negative = sum(1 for p in posts if p['sentiment'] < -0.5)
            
            result = {
                'total': len(posts),
                'positive': positive,
                'negative': negative,
                'neutral': len(posts) - positive - negative,
                'updated_at': datetime.now().isoformat()
            }
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Reddit error: {e}")
            return {'error': str(e)}

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyse de sentiment avec TextBlob
        """
        from textblob import TextBlob
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    async def get_github_activity(self, repo: str,
                                start_date: str = None,
                                end_date: str = None) -> pd.DataFrame:
        """
        Récupère l'activité GitHub d'un repo
        Args:
            repo: Format 'owner/repo'
            start_date: Date de début YYYY-MM-DD
            end_date: Date de fin YYYY-MM-DD
        Returns:
            DataFrame avec colonnes:
            - date
            - commits
            - stars
            - forks
            - issues_opened
            - issues_closed
        """
        cache_key = f"github_{repo}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
            
        try:
            # Implémentation API GitHub
            url = f"https://api.github.com/repos/{repo}/stats/commit_activity"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
            
            # Traitement des données
            activity = []
            for week in data:
                week_date = datetime.fromtimestamp(week['week']).strftime('%Y-%m-%d')
                if (not start_date or week_date >= start_date) and \
                   (not end_date or week_date <= end_date):
                    activity.append({
                        'date': week_date,
                        'commits': week['total'],
                        'stars': 0,  # À compléter avec autre endpoint
                        'forks': 0,  # À compléter avec autre endpoint
                        'issues_opened': 0,
                        'issues_closed': 0
                    })
            
            result = pd.DataFrame(activity)
            result.set_index('date', inplace=True)
            self._cache[cache_key] = result
            return result.copy()
            
        except Exception as e:
            print(f"GitHub error: {e}")
            return pd.DataFrame()
