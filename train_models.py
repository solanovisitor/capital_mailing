# train_models.py

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import utils
import boto3
import pickle
import numpy as np
from datetime import datetime, timedelta

s3 = boto3.client('s3')

def train_models():
    assets = utils.get_asset_list()

    # Define the start and end dates for fetching historical data
    end_date = datetime.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=5 * 365)

    for asset in assets:
        # Fetch historical data
        historical_data = utils.get_historical_data(asset, start_date, end_date)
        
        # Fetch sentiment data
        sentiment_data = utils.get_sentiment_data(utils.NEWS_API_KEY)
        
        # Fetch fundamental data
        fundamental_data = utils.get_fundamental_data(utils.QUANDL_API_KEY)

        # Preprocess the data
        X, y_classification, y_regression = preprocess_data(historical_data, sentiment_data, fundamental_data)

        # Split the data into training and testing sets
        X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)
        _, _, y_reg_train, y_reg_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)

        # Train the classification model
        classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        classification_model.fit(X_train, y_class_train)

        # Evaluate the classification model
        y_class_pred = classification_model.predict(X_test)
        class_accuracy = (y_class_pred == y_class_test).mean()

        print(f"Asset: {asset} - Classification Accuracy: {class_accuracy:.4f}")

        # Train the regression model
        regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
        regression_model.fit(X_train, y_reg_train)

        # Evaluate the regression model
        y_reg_pred = regression_model.predict(X_test)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        mse = mean_squared_error(y_reg_test, y_reg_pred)

        print(f"Asset: {asset} - Regression MAE: {mae:.4f} - MSE: {mse:.4f}")

        # Save the trained models
        classification_model_name = f"{asset}_classification_model.pkl"
        regression_model_name = f"{asset}_regression_model.pkl"

        with open(classification_model_name, 'wb') as f:
            pickle.dump(classification_model, f)
        with open(regression_model_name, 'wb') as f:
            pickle.dump(regression_model, f)

        # Upload the models to S3
        s3.upload_file(classification_model_name, utils.S3_BUCKET_NAME, classification_model_name)
        s3.upload_file(regression_model_name, utils.S3_BUCKET_NAME, regression_model_name)


def preprocess_data(historical_data, sentiment_data, fundamental_data):
    # Define the prediction horizon
    horizon = 20

    # Combine historical_data, sentiment_data, and fundamental_data
    combined_data = historical_data.copy()
    ticker = historical_data.index[0][0]
    combined_data['Sentiment'] = sentiment_data[ticker]
    combined_data['PE'] = fundamental_data[ticker]['PE']
    combined_data['RevenueGrowth'] = fundamental_data[ticker]['RevenueGrowth']

    # Calculate the return for the next 'horizon' days
    combined_data['FutureReturn'] = combined_data['Close'].shift(-horizon) / combined_data['Close'] - 1

    # Calculate features (e.g., moving averages, RSI, MACD)
    combined_data['SMA_10'] = combined_data['Close'].rolling(window=10).mean()
    combined_data['SMA_30'] = combined_data['Close'].rolling(window=30).mean()

    # Calculate the RSI
    delta = combined_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    combined_data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate the MACD
    combined_data['EMA_12'] = combined_data['Close'].ewm(span=12).mean()
    combined_data['EMA_26'] = combined_data['Close'].ewm(span=26).mean()
    combined_data['MACD'] = combined_data['EMA_12'] - combined_data['EMA_26']

    # Drop rows with NaN values
    combined_data.dropna(inplace=True)

    # Prepare the dataset for classification and regression tasks
    X = combined_data[['SMA_10', 'SMA_30', 'RSI', 'MACD', 'Sentiment', 'PE', 'RevenueGrowth']].values

    # Classification: Buy (1) or Sell (0)
    y_classification = np.where(combined_data['FutureReturn'] > 0, 1, 0)

    # Regression: Predict the price in 'horizon' days
    y_regression = combined_data['Close'].shift(-horizon).dropna().values

    return X, y_classification, y_regression

if __name__ == "__main__":
    train_models()