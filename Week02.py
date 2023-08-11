import argparse
import pandas as pd
import numpy as np

def calculate_moving_average(series, window_size):
    return series.rolling(window=window_size).mean()

def calculate_lag_feature(series, lag):
    return series.shift(lag)

def main():
    parser = argparse.ArgumentParser(description="Generate product features based on given data.")
    parser.add_argument("--min-date", type=str, default="2021-01-08", help="Start date of the date range")
    parser.add_argument("--max-date", type=str, default="2021-05-30", help="End date of the date range")
    args = parser.parse_args()

    brand_df = pd.read_csv(r"C:\Users\Sinan\Downloads\brand.csv")
    product_df = pd.read_csv(r"C:\Users\Sinan\Downloads\product.csv")
    store_df = pd.read_csv(r"C:\Users\Sinan\Downloads\store.csv")
    sales_df = pd.read_csv(r"C:\Users\Sinan\Downloads\sales.csv")


    sales_df["date"] = pd.to_datetime(sales_df["date"])

    # Filter data based on date range
    filtered_sales_df = sales_df[(sales_df["date"] >= args.min_date) & (sales_df["date"] <= args.max_date)]

    # Merge data to create the final output
    merged_sales_df = filtered_sales_df.merge(product_df, left_on="product", right_on="id", suffixes=("", "_product"))
    merged_sales_df = merged_sales_df.merge(store_df, left_on="store", right_on="id", suffixes=("", "_store"))
    
    # Create a new 'brand' column using 'name' from brand_df
    merged_sales_df = merged_sales_df.merge(brand_df, left_on="brand", right_on="name", suffixes=("", "_brand"))
    merged_sales_df['brand_id'] = merged_sales_df['id_brand'].astype('int64')
    merged_sales_df.drop(columns=['id_brand', 'name_brand'], inplace=True)

    # Calculate Moving Averages and Lag Features
    merged_sales_df["sales_product"] = merged_sales_df.groupby(["product", "store", "date"])["quantity"].transform("sum")
    merged_sales_df["MA7_P"] = merged_sales_df.groupby("product")["sales_product"].apply(lambda x: calculate_moving_average(x, 7)).reset_index(drop=True)
    merged_sales_df["LAG7_P"] = merged_sales_df.groupby("product")["sales_product"].apply(lambda x: calculate_lag_feature(x, 7)).reset_index(drop=True)
    merged_sales_df["sales_brand"] = merged_sales_df.groupby(["brand_id", "store", "date"])["quantity"].transform("sum")
    merged_sales_df["MA7_B"] = merged_sales_df.groupby("brand_id")["sales_brand"].apply(lambda x: calculate_moving_average(x, 7)).reset_index(drop=True)
    merged_sales_df["LAG7_B"] = merged_sales_df.groupby("brand_id")["sales_brand"].apply(lambda x: calculate_lag_feature(x, 7)).reset_index(drop=True)
    merged_sales_df["sales_store"] = merged_sales_df.groupby(["store", "date"])["quantity"].transform("sum")
    merged_sales_df["MA7_S"] = merged_sales_df.groupby("store")["sales_store"].apply(lambda x: calculate_moving_average(x, 7)).reset_index(drop=True)
    merged_sales_df["LAG7_S"] = merged_sales_df.groupby("store")["sales_store"].apply(lambda x: calculate_lag_feature(x, 7)).reset_index(drop=True)

    # Define output columns
    output_columns = [
        "product", "store", "brand_id", "date",
        "quantity", "MA7_P", "LAG7_P",
        "sales_brand", "MA7_B", "LAG7_B",
        "sales_store", "MA7_S", "LAG7_S"
    ]

    # Create the final output
    final_output = merged_sales_df[output_columns]

    # Sort by specified columns
    final_output = final_output.sort_values(by=["product", "brand_id", "store", "date"])

    # Write the output to features.csv
    final_output.to_csv("features.csv", index=False)

if __name__ == "__main__":
    main()
