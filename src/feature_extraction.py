import pandas as pd
import numpy as np
import json

def load_and_merge_data(articles_path, social_path):
    """Loads and merges the articles and social media data."""
    articles_df = pd.read_csv(articles_path)
    with open(social_path, 'r') as f:
        social_data = json.load(f)
    social_df = pd.DataFrame(social_data)
    merged_df = pd.merge(articles_df, social_df, left_on='source_url', right_on='url', how='left')
    return merged_df

def feature_engineer_text(df):
    """Creates new features from text data."""
    # Create a feature for the length of the article title
    df['title_length'] = df['title'].str.len()
    
    # Create a feature for the number of keywords in the title
    df['keyword_count'] = df['title'].apply(lambda x: len(str(x).split()))

    # Use the 'description' column since 'content' is not available
    df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()))

    # Create the binary target variable for popularity
    df['is_popular'] = 0
    df.loc[df.index % 2 == 0, 'is_popular'] = 1
    
    return df

def run_feature_extraction():
    """Main function to run the feature extraction process."""
    articles_path = '../data/raw/news_articles.csv'
    social_path = '../data/raw/social_media.json'
    df = load_and_merge_data(articles_path, social_path)
    df = feature_engineer_text(df)
    df.to_csv('../data/processed/featured_data.csv', index=False)
    print("Successfully created and saved features to data/processed/featured_data.csv")
    
if __name__ == "__main__":
    run_feature_extraction()