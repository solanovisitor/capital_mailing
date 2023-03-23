# utils.py

import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas_datareader as pdr
import smtplib
import ssl
from email.message import EmailMessage

from config import YFINANCE_API_KEY, S3_BUCKET_NAME

def get_historical_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def get_sentiment_data(api_key):
    newsapi = NewsApiClient(api_key=api_key)
    analyzer = SentimentIntensityAnalyzer()

    assets = get_asset_list()
    sentiment_data = {}

    for asset in assets:
        # Fetch news articles for the asset
        articles = newsapi.get_everything(q=asset, language='en', sort_by='relevancy', page_size=20)

        # Calculate sentiment scores using VADER
        sentiment_scores = []

        for article in articles['articles']:
            sentiment_score = analyzer.polarity_scores(article['title'] + ' ' + article['description'])
            sentiment_scores.append(sentiment_score['compound'])

        # Calculate the average sentiment score for the asset
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

        # Add the sentiment data to the dictionary
        sentiment_data[asset] = avg_sentiment_score

    return sentiment_data

def get_fundamental_data(api_key):
    assets = get_asset_list()
    fundamental_data = {}

    for asset in assets:
        try:
            # Fetch financial data using pandas-datareader and Quandl
            financial_data = pdr.get_data_quandl(asset, api_key=api_key)

            # Extract relevant fundamental data (e.g., P/E ratio, revenue growth)
            pe_ratio = financial_data['PE']
            revenue_growth = financial_data['RevenueGrowth']

            # Add the fundamental data to the dictionary
            fundamental_data[asset] = {'PE': pe_ratio, 'RevenueGrowth': revenue_growth}

        except Exception as e:
            print(f"Error fetching fundamental data for {asset}: {e}")

    return fundamental_data

def get_asset_list():
    # Define a list of assets you want to analyze
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
    return assets

def get_model_path(model_name):
    return f"s3://{S3_BUCKET_NAME}/{model_name}"

def preprocess_data_today(historical_data, sentiment_data, fundamental_data):
    """
    Implement data preprocessing and feature extraction for a single data point (today).
    Combine historical_data, sentiment_data, and fundamental_data into a single feature vector (X_today).
    """
    # Perform the necessary preprocessing and feature extraction steps for the given data
    pass

def send_email(recipient, subject, content):
    message = EmailMessage()
    message.set_content(content)
    message["Subject"] = subject
    message["From"] = "your_email@example.com"
    message["To"] = recipient

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.example.com", 465, context=context) as server:
        server.login("your_email@example.com", "your_email_password")
        server.send_message(message)

def get_email_recipients():
    """
    Implement a function to get the list of email recipients.
    This could be from a database, a file, or a hardcoded list.
    """
    # Example: return a hardcoded list of recipients
    return ["customer1@example.com", "customer2@example.com", "customer3@example.com"]
