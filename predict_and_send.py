import boto3
import pickle
import pandas as pd
from datetime import datetime, timedelta
from utils import (get_asset_list, get_historical_data, get_sentiment_data, get_fundamental_data, 
                   preprocess_data_today, send_email, get_email_recipients)

from config import S3_BUCKET_NAME, NEWS_API_KEY, QUANDL_API_KEY

s3 = boto3.client('s3')

def download_model(asset, model_type):
    model_name = f"{asset}_{model_type}_model.pkl"
    s3.download_file(S3_BUCKET_NAME, model_name, model_name)

    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    
    return model

def generate_predictions():
    assets = get_asset_list()
    today = datetime.today()

    predictions = []

    for asset in assets:
        # Fetch historical data
        historical_data = get_historical_data(asset, today - timedelta(days=5 * 365), today)

        # Fetch sentiment data
        sentiment_data = get_sentiment_data(NEWS_API_KEY)

        # Fetch fundamental data
        fundamental_data = get_fundamental_data(QUANDL_API_KEY)

        # Preprocess the data for a single data point (today)
        X_today = preprocess_data_today(historical_data, sentiment_data, fundamental_data)

        # Download and load the trained models for the asset
        classification_model = download_model(asset, 'classification')
        regression_model = download_model(asset, 'regression')

        # Generate predictions
        predicted_signal = classification_model.predict(X_today)[0]
        predicted_price = regression_model.predict(X_today)[0]

        predictions.append({
            'Asset': asset,
            'PredictedSignal': predicted_signal,
            'PredictedPrice': predicted_price,
            'Date': today
        })

    return predictions

def summarize_predictions(predictions):
    summary = "Asset Predictions:\n\n"

    for prediction in predictions:
        signal = "Buy" if prediction['PredictedSignal'] == 1 else "Sell"
        summary += f"{prediction['Asset']}: {signal} - Predicted Price in 20 days: {prediction['PredictedPrice']:.2f}\n"

    return summary

def main():
    predictions = generate_predictions()
    summary = summarize_predictions(predictions)

    print(summary)

    # Send the summary via email to the customer base
    subject = "Asset Predictions"
    recipients = get_email_recipients()

    for recipient in recipients:
        send_email(recipient, subject, summary)

if __name__ == "__main__":
    main()
