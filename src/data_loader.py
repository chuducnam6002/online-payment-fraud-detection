# src/data_loader.py — Đọc và tải dữ liệu | Người phụ trách: Thắng

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
    Tải dữ liệu thô từ CSV với dtype tối ưu bộ nhớ.

    Parameters
    ----------
    filepath : str   — đường dẫn file CSV
    nrows    : int   — số dòng cần load (None = toàn bộ ~6.3M)
                       Dùng nrows=100_000 để test nhanh
    verbose  : bool  — in thông tin sau khi load

    Example
    -------
    >>> from src.data_loader import load_data
    >>> df = load_data(nrows=100_000)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy: {filepath}\n"
            "Tải dataset từ Kaggle rồi đặt vào data/raw/"
        )
    df = pd.read_csv(filepath, dtype=DTYPES, usecols=NEEDED_COLS,
                     nrows=nrows, low_memory=True)
    if verbose:
        print(f"✅ Loaded {len(df):,} rows x {df.shape[1]} cols | "
              f"RAM: {df.memory_usage(deep=True).sum()/1e6:.1f} MB | "
              f"Fraud rate: {df['isFraud'].mean()*100:.4f}%")
    return df


def save_processed_data(df, filename='processed_data.csv'):
    """Lưu DataFrame đã xử lý vào data/processed/."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path} ({len(df):,} rows)")


def load_processed_data(filename='processed_data.csv', verbose=True):
    """Load dữ liệu từ data/processed/ (cần chạy 02_preprocessing trước)."""
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa có {path} — chạy 02_preprocessing.ipynb trước.")
    df = pd.read_csv(path)
    if verbose:
        print(f"✅ Loaded processed: {df.shape}")
    return df
