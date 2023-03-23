import pandas as pd
from datetime import datetime, timedelta
from utils import (get_asset_list, get_historical_data, get_db_connection, fetch_asset_data_from_db, 
                   update_asset_data_in_db, calculate_classification_metrics)

def update_actual_values():
    assets = get_asset_list()
    today = datetime.today()

    for asset in assets:
        # Fetch the asset data from the database
        asset_data = fetch_asset_data_from_db(asset)

        # Check if the actual values need to be updated
        if asset_data['Date'].date() == today.date():
            # Fetch the actual price and return
            historical_data = get_historical_data(asset, today - timedelta(days=1), today)
            actual_price = historical_data.iloc[-1]['Close']
            actual_return = (actual_price / historical_data.iloc[-2]['Close']) - 1

            # Determine the actual signal
            actual_signal = 1 if actual_return > 0 else 0

            # Calculate the mean absolute error (MAE) and mean squared error (MSE)
            mae = abs(asset_data['PredictedPrice'] - actual_price)
            mse = (asset_data['PredictedPrice'] - actual_price) ** 2

            # Calculate the classification label
            classification_label = 1 if asset_data['PredictedSignal'] == actual_signal else 0

            # Update the asset data in the database
            update_asset_data_in_db(asset, actual_price, actual_return, actual_signal, mae, mse, classification_label)

def calculate_and_store_metrics():
    assets = get_asset_list()
    db_conn = get_db_connection()

    for asset in assets:
        # Fetch the updated asset data from the database
        asset_data = fetch_asset_data_from_db(asset)

        # Calculate classification metrics (precision, recall, accuracy, etc.)
        metrics = calculate_classification_metrics(asset_data)

        # Create a table in the database to save the calculated metrics
        table_name = f"{asset}_classification_metrics"
        metrics.to_sql(table_name, db_conn, if_exists='replace', index=False)

    db_conn.close()

def main():
    update_actual_values()
    calculate_and_store_metrics()

if __name__ == "__main__":
    main()
