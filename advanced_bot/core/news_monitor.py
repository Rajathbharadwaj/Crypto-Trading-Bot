import asyncio
import tweepy
from newsapi import NewsApiClient
import json
import logging
from typing import Dict, List, Callable
import aiohttp
import time

class NewsMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.callbacks: List[Callable] = []
        self.last_processed_tweet_id = None
        self.last_news_timestamp = time.time()
        
        # Initialize Twitter API
        self.twitter_client = tweepy.Client(
            bearer_token=self.config.TWITTER_BEARER_TOKEN,
            consumer_key=self.config.TWITTER_API_KEY,
            consumer_secret=self.config.TWITTER_API_SECRET,
            access_token=self.config.TWITTER_ACCESS_TOKEN,
            access_token_secret=self.config.TWITTER_ACCESS_SECRET
        )
        
        # Initialize News API
        self.news_client = NewsApiClient(api_key=self.config.NEWS_API_KEY)

    def register_callback(self, callback: Callable):
        """Register a callback function to be called when news is detected"""
        self.callbacks.append(callback)

    async def monitor_twitter(self):
        """Monitor Twitter for relevant crypto news"""
        while True:
            try:
                # Get tweets from influential accounts
                for influencer in self.config.NEWS_TRADING['influencers']:
                    tweets = self.twitter_client.get_user_tweets(
                        username=influencer,
                        max_results=10,
                        since_id=self.last_processed_tweet_id
                    )
                    
                    if tweets.data:
                        for tweet in tweets.data:
                            # Check if tweet contains relevant keywords
                            if any(keyword.lower() in tweet.text.lower() 
                                  for keyword in self.config.NEWS_TRADING['keywords']):
                                
                                news_data = {
                                    'source': 'twitter',
                                    'author': influencer,
                                    'content': tweet.text,
                                    'timestamp': tweet.created_at,
                                    'url': f"https://twitter.com/{influencer}/status/{tweet.id}"
                                }
                                
                                # Notify all registered callbacks
                                for callback in self.callbacks:
                                    await callback(news_data)
                        
                        self.last_processed_tweet_id = tweets.data[0].id

                await asyncio.sleep(2)  # Rate limit compliance

            except Exception as e:
                self.logger.error(f"Error monitoring Twitter: {str(e)}")
                await asyncio.sleep(5)

    async def monitor_news_api(self):
        """Monitor NewsAPI for crypto news"""
        while True:
            try:
                news = self.news_client.get_everything(
                    q='bitcoin OR cryptocurrency OR crypto',
                    language='en',
                    sort_by='publishedAt'
                )

                for article in news['articles']:
                    if article['publishedAt'] > self.last_news_timestamp:
                        if any(keyword.lower() in article['title'].lower() 
                              for keyword in self.config.NEWS_TRADING['keywords']):
                            
                            news_data = {
                                'source': 'newsapi',
                                'author': article['source']['name'],
                                'content': article['title'],
                                'timestamp': article['publishedAt'],
                                'url': article['url']
                            }
                            
                            # Notify all registered callbacks
                            for callback in self.callbacks:
                                await callback(news_data)

                self.last_news_timestamp = time.time()
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error monitoring NewsAPI: {str(e)}")
                await asyncio.sleep(60)

    async def monitor_exchange_announcements(self):
        """Monitor major exchange announcement pages"""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Monitor Binance announcements
                    async with session.get('https://www.binance.com/en/support/announcement/c-48') as response:
                        if response.status == 200:
                            data = await response.json()
                            for announcement in data['data']:
                                if any(keyword.lower() in announcement['title'].lower() 
                                      for keyword in self.config.NEWS_TRADING['keywords']):
                                    
                                    news_data = {
                                        'source': 'binance',
                                        'author': 'Binance',
                                        'content': announcement['title'],
                                        'timestamp': announcement['time'],
                                        'url': f"https://www.binance.com/en/support/announcement/{announcement['id']}"
                                    }
                                    
                                    # Notify all registered callbacks
                                    for callback in self.callbacks:
                                        await callback(news_data)

                    await asyncio.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    self.logger.error(f"Error monitoring exchange announcements: {str(e)}")
                    await asyncio.sleep(60)

    async def run(self):
        """Run all news monitoring tasks"""
        await asyncio.gather(
            self.monitor_twitter(),
            self.monitor_news_api(),
            self.monitor_exchange_announcements()
        )
