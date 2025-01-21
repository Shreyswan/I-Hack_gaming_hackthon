#######################################################
# Owners: Aditi Pawar, Himanish Goel, Shreyas Sawant
# File name: data_handler.py
# Purpose: Read data and and clean the tweets.
#######################################################

# Import necessary libraries
import re
import time
from config import *
import pandas as pd
from langdetect import detect, DetectorFactory

# Ensure consistent language detection
DetectorFactory.seed = 0

# Function for cleaning the tweets by removing unnecessary content
def data_cleaner(text):
    text = re.sub(r'@\w+', '', text) # Remove @ mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    text = ''.join([word for word in text.split() if word not in FALLBACK_STOPWORDS]) # Stopword removal for data reduction
    
    try:
        # Check if the text is in English
        if detect(text) != 'en':
            return None  # Mark non-English sentences as None
    except:
        return None  # If language detection fails, consider it non-English
    return text.strip()  # Return cleaned text


def main():
    # Apply the cleaning function
    filename = "cyberbullying_tweets.csv"
    data = pd.read_csv(filename)

    start = time.time()
    data['cleaned_text'] = data['tweet_text'].apply(data_cleaner)

    # Remove rows with None values (non-English sentences)
    data = data.dropna(subset=['cleaned_text'])
    end = time.time()

    # Describing the dataset.
    print(data.describe())
    print("TOTAL TIME TAKEN:", end - start) # Time for execution: 122-123 secs (Machine: Mac mini)

    data.to_csv("clean_dataset.csv", columns = data.columns)

if __name__ == '__main__':
    main()