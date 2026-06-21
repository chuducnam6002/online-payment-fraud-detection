# src/data_loader.py — Load and read data 

import pandas as pd, os

RAW_DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'payment_fraud.csv')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

NEEDED_COLS = ['step','type','amount','oldbalanceOrg','newbalanceOrig',
               'oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud']

DTYPES = {'step':'int16','type':'category','amount':'float32',
          'oldbalanceOrg':'float32','newbalanceOrig':'float32',
          'oldbalanceDest':'float32','newbalanceDest':'float32',
          'isFraud':'int8','isFlaggedFraud':'int8'}


def load_data(filepath=RAW_DATA_PATH, nrows=None, verbose=True):
    """
    Load raw data from CSV with memory-optimized dtypes.

    Parameters
    ----------
    filepath : str   — path to CSV file
    nrows    : int   — number of rows to load (None = full ~6.3M rows)
                       Use nrows=100_000 for quick testing
    verbose  : bool  — print information after loading

    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            "Please download the dataset from Kaggle and place it in data/raw/"
        )

    df = pd.read_csv(
        filepath,
        dtype=DTYPES,
        usecols=NEEDED_COLS,
        nrows=nrows,
        low_memory=True
    )

    if verbose:
        print(f"✅ Loaded {len(df):,} rows x {df.shape[1]} cols | "
              f"RAM: {df.memory_usage(deep=True).sum()/1e6:.1f} MB | "
              f"Fraud rate: {df['isFraud'].mean()*100:.4f}%")

    return df


def save_processed_data(df, filename='processed_data.csv'):
    """Save the processed DataFrame to data/processed/."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path} ({len(df):,} rows)")


def load_processed_data(filename='processed_data.csv', verbose=True):
    """Load data from data/processed/ (run 02_preprocessing first)."""
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — please run 02_preprocessing.ipynb first.")
    df = pd.read_csv(path)
    if verbose:
        print(f"✅ Loaded processed: {df.shape}")
    return df
