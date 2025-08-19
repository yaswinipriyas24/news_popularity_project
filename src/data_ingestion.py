import requests
import json
import pandas as pd
import numpy as np

def fetch_and_save_data():
    """Fetches news and social media data and saves it to the raw folder."""
    
    # 1. Fetch News Articles from Newsdata.io
    API_KEY = "NEWSDATA_API_KEY"  # Use your actual API key here
    BASE_URL = "https://newsdata.io/api/1/news"
    params = {
        'apikey': API_KEY,
        'q': 'technology',
        'language': 'en',
        'country': 'in'
    }

    print("Fetching news articles...")
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        news_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        print("Please check your API key and internet connection.")
        return

    if 'results' in news_data and len(news_data['results']) > 0:
        articles_df = pd.DataFrame(news_data['results'])
        articles_df.to_csv('../data/raw/news_articles.csv', index=False)
        print("Successfully saved news articles to data/raw/news_articles.csv")
    else:
        print("No articles found in the API response.")
        return

    # 2. Simulate Social Media Data
    print("Simulating social media data...")
    social_data = []
    if 'results' in news_data:
        for article in news_data['results']:
            data = {
                "url": article['link'],
                "shares": np.random.randint(0, 500),
                "likes": np.random.randint(0, 1000),
                "comments": np.random.randint(0, 200)
            }
            social_data.append(data)
    
    with open('../data/raw/social_media.json', 'w') as f:
        json.dump(social_data, f, indent=4)
    print("Successfully saved simulated social media data to data/raw/social_media.json")

if __name__ == "__main__":
    fetch_and_save_data()
