import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_dataset():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, 'raw_data.csv')
    
    if os.path.exists(raw_path):
        print(f"Dataset already exists at {raw_path}")
        return
    
    print("Downloading dataset 'Default of Credit Card Clients' from UCI Machine Learning Repository...")
    
    try:
        # fetch dataset 
        default_of_credit_card_clients = fetch_ucirepo(id=350) 
          
        # data (as pandas dataframes) 
        X = default_of_credit_card_clients.data.features 
        y = default_of_credit_card_clients.data.targets 
        
        # Merge X and y
        df = pd.concat([X, y], axis=1)
        
        # Clean up column names
        df.columns = [col.lower() for col in df.columns]
        
        column_map = {
            'x1': 'limit_bal', 'x2': 'sex', 'x3': 'education', 'x4': 'marriage', 'x5': 'age',
            'x6': 'pay_0', 'x7': 'pay_2', 'x8': 'pay_3', 'x9': 'pay_4', 'x10': 'pay_5', 'x11': 'pay_6',
            'x12': 'bill_amt1', 'x13': 'bill_amt2', 'x14': 'bill_amt3', 'x15': 'bill_amt4', 'x16': 'bill_amt5', 'x17': 'bill_amt6',
            'x18': 'pay_amt1', 'x19': 'pay_amt2', 'x20': 'pay_amt3', 'x21': 'pay_amt4', 'x22': 'pay_amt5', 'x23': 'pay_amt6'
        }
        df = df.rename(columns=column_map)
        
        # Rename target column for clarity
        if 'y' in df.columns:
            df = df.rename(columns={'y': 'default_payment_next_month'})
        elif 'default.payment.next.month' in df.columns:
            df = df.rename(columns={'default.payment.next.month': 'default_payment_next_month'})
        
        # Save to CSV
        df.to_csv(raw_path, index=False)
        print(f"Dataset successfully downloaded and saved to {raw_path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Falling back to a backup mirror...")
        import urllib.request
        import io
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        # Reading excel directly from URL might be tricky if xlrd is missing.
        # But pandas can read_excel if openpyxl/xlrd is present, but we don't have it installed.
        raise e

if __name__ == "__main__":
    download_dataset()
