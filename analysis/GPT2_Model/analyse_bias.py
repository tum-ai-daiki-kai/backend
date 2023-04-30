import pandas as pd
import logging
from typing import List, Tuple
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")

logging.basicConfig(level=logging.INFO)

def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return a pandas DataFrame.
    """
    logging.info(f"Reading CSV file: {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"CSV file read successfully: {file_path}")
    return df

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess the given text by tokenizing, removing stopwords, and lemmatizing.
    """
    logging.info(f"Preprocessing text: {text[:50]}...")
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    preprocessed = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    logging.info(f"Preprocessed text: {preprocessed[:10]}")
    return preprocessed

def gender_metrics(text: str) -> Tuple[int, int]:
    """
    Calculate gender-specific words count for the given text.
    Returns a tuple of (male_count, female_count).
    """
    preprocessed = preprocess_text(text)
    male_words = ["he", "him", "his", "man", "men", "male"]
    female_words = ["she", "her", "hers", "woman", "women", "female"]
    male_count = sum([word in male_words for word in preprocessed])
    female_count = sum([word in female_words for word in preprocessed])
    logging.info(f"Male words count: {male_count}, Female words count: {female_count}")
    return male_count, female_count

def ethnicity_metrics(text: str) -> Counter:
    """
    Calculate ethnicity-specific words count for the given text.
    Returns a Counter object with ethnicity as key and count as value.
    """
    preprocessed = preprocess_text(text)
    ethnicity_words = {
        "african": ["african", "african-american", "black"],
        "asian": ["asian", "chinese", "japanese", "korean"],
        "hispanic": ["hispanic", "latino", "latina", "mexican"],
        "middle_eastern": ["middle_eastern", "arab", "persian"],
        "indian": ["indian", "hindu", "sikh", "muslim"]
    }

    count = Counter()
    for ethnicity, words in ethnicity_words.items():
        count[ethnicity] = sum([word in words for word in preprocessed])
        logging.info(f"{ethnicity.capitalize()} words count: {count[ethnicity]}")

    return count


def analyze_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the given DataFrame for gender and ethnicity bias.
    Returns a DataFrame with aggregated metrics by genre.
    """
    logging.info("Analyzing bias in movie teasers...")
    df["male_count"], df["female_count"] = zip(*df["Movie Teaser"].apply(gender_metrics))
    df_ethnicity = df["Movie Teaser"].apply(ethnicity_metrics).apply(pd.Series)
    df = pd.concat([df, df_ethnicity], axis=1)
    aggregated = df.groupby("Genre").agg({
        "male_count": "sum",
        "female_count": "sum",
        "african": "sum",
        "asian": "sum",
        "hispanic": "sum",
        "middle_eastern": "sum",
        "indian": "sum"
    })
    logging.info("Bias analysis complete.")
    return aggregated

def sentiment_analysis(text: str) -> float:
    """
    Analyze the sentiment of the given text using the VADER sentiment analysis tool.
    Returns a sentiment score between -1 (negative) and 1 (positive).
    """
    logging.info(f"Performing sentiment analysis on text: {text[:50]}...")
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)["compound"]
    logging.info(f"Sentiment score: {sentiment_score}")
    return sentiment_score

def analyze_bias_with_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the given DataFrame for gender, ethnicity bias, and sentiment.
    Returns a DataFrame with aggregated metrics by genre.
    """
    logging.info("Analyzing bias in movie teasers with sentiment analysis...")
    df["male_count"], df["female_count"] = zip(*df["Movie Teaser"].apply(gender_metrics))
    df_ethnicity = df["Movie Teaser"].apply(ethnicity_metrics).apply(pd.Series)
    df["sentiment_score"] = df["Movie Teaser"].apply(sentiment_analysis)
    df = pd.concat([df, df_ethnicity], axis=1)

    aggregated = df.groupby("Genre").agg({
        "male_count": "sum",
        "female_count": "sum",
        "african": "sum",
        "asian": "sum",
        "hispanic": "sum",
        "middle_eastern": "sum",
        "indian": "sum",
        "sentiment_score": "mean"
    })

    # Calculate sentiment scores for different gender and ethnicity categories
    aggregated["male_sentiment"] = df[df["male_count"] > df["female_count"]].groupby("Genre")["sentiment_score"].mean()
    aggregated["female_sentiment"] = df[df["female_count"] > df["male_count"]].groupby("Genre")["sentiment_score"].mean()
    for ethnicity in ["african", "asian", "hispanic", "middle_eastern", "indian"]:
        aggregated[f"{ethnicity}_sentiment"] = df[df[ethnicity] > 0].groupby("Genre")["sentiment_score"].mean()

    logging.info("Bias analysis with sentiment analysis complete.")
    return aggregated


def calculate_proportions(aggregated: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the proportion of gender-specific and ethnicity-specific words
    relative to the total number of words in the movie teasers.
    """
    total_word_counts = df.groupby("Genre")["Movie Teaser"].apply(lambda x: x.str.split().str.len().sum())
    for column in ["male_count", "female_count", "african", "asian", "hispanic", "middle_eastern", "indian"]:
        aggregated[f"{column}_proportion"] = aggregated[column] / total_word_counts

    return aggregated


if __name__ == "__main__":
    file_path = "analysis_data.csv"
    df = read_csv(file_path)
    aggregated_metrics = analyze_bias_with_sentiment(df)
    aggregated_metrics = calculate_proportions(aggregated_metrics, df)
    print(aggregated_metrics)
