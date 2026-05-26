# src/preprocess.py — Tiền xử lý dùng chung | Người phụ trách: Hoàng

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

FEATURE_COLS = ['step','type_encoded','amount',
                'oldbalanceOrg','newbalanceOrig',
                'oldbalanceDest','newbalanceDest',
                'errorBalanceOrig','errorBalanceDest']
TARGET_COL = 'isFraud'


# ------------------------------------------------------------------
# BƯỚC 1 — Feature Engineering
# ------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo feature mới từ insight EDA:
    - errorBalanceOrig : oldbalanceOrg - amount - newbalanceOrig
                         Nếu != 0 → tín hiệu gian lận rất mạnh
    - errorBalanceDest : oldbalanceDest + amount - newbalanceDest
    - type_encoded     : mã hoá cột 'type' thành số

    Bỏ: isFlaggedFraud (tương quan thấp), type (đã encode)
    """
    df = df.copy()
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    df.drop(columns=['isFlaggedFraud', 'type'], inplace=True, errors='ignore')
    return df


# ------------------------------------------------------------------
# BƯỚC 2 — Lọc transaction type
# ------------------------------------------------------------------
def filter_fraud_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chỉ giữ TRANSFER (4) và CASH_OUT (1) vì theo EDA,
    100% gian lận chỉ xảy ra ở 2 loại này → giảm noise.
    """
    fraud_type_codes = [1, 4]  # CASH_OUT=1, TRANSFER=4 (sau LabelEncoder)
    out = df[df['type_encoded'].isin(fraud_type_codes)].copy()
    print(f"✅ Sau lọc type: {len(out):,} / {len(df):,} rows")
    return out


# ------------------------------------------------------------------
# BƯỚC 3 — Tách X/y + Train-Test Split (stratified)
# ------------------------------------------------------------------
def split_features_target(df):
    return df[FEATURE_COLS], df[TARGET_COL]


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Phân chia stratified — giữ tỷ lệ fraud trong cả 2 tập.
    QUAN TRỌNG: luôn dùng stratify=y khi dữ liệu imbalanced.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"✅ Train {len(X_tr):,} (fraud {y_tr.mean()*100:.3f}%) | "
          f"Test {len(X_te):,} (fraud {y_te.mean()*100:.3f}%)")
    return X_tr, X_te, y_tr, y_te


# ------------------------------------------------------------------
# BƯỚC 4 — SMOTE + UnderSampling
# ------------------------------------------------------------------
def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    """
    SMOTE → tăng minority class (fraud) lên sampling_strategy * majority
    RandomUnderSampler → giảm majority class → tránh tập train quá lớn

    QUAN TRỌNG: chỉ áp dụng trên TRAIN, KHÔNG dùng trên test set!
    Áp dụng trên test set = data leakage → kết quả ảo.
    """
    pipeline = ImbPipeline([
        ('over',  SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),
        ('under', RandomUnderSampler(sampling_strategy=0.5,  random_state=random_state)),
    ])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    print(f"✅ Sau SMOTE: Non-fraud {(y_res==0).sum():,} | Fraud {(y_res==1).sum():,} "
          f"({y_res.mean()*100:.2f}%)")
    return X_res, y_res


# ------------------------------------------------------------------
# PIPELINE TỔNG HỢP
# ------------------------------------------------------------------
def preprocess_pipeline(df, filter_types=True, apply_resampling=True,
                         test_size=0.2, random_state=42):
    """
    Chạy toàn bộ pipeline tiền xử lý:
    1. Feature Engineering
    2. (tuỳ chọn) Lọc TRANSFER & CASH_OUT
    3. Train-Test Split stratified
    4. (tuỳ chọn) SMOTE trên tập train

    Returns dict: {X_train, X_test, y_train, y_test}
    """
    print("=" * 50)
    df = engineer_features(df)
    if filter_types:
        df = filter_fraud_types(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    if apply_resampling:
        X_train, y_train = apply_smote(X_train, y_train, random_state=random_state)
    print("✅ Pipeline hoàn tất!\n")
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
