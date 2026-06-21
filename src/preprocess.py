# src/preprocess.py — Shared preprocessing module
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



# STEP 1 — Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features based on EDA insights:

    - errorBalanceOrig: oldbalanceOrg - amount - newbalanceOrig
    If the value is not equal to 0, it is a strong indicator of fraud.

    - errorBalanceDest: oldbalanceDest + amount - newbalanceDest

    - type_encoded: numerical encoding of the 'type' column

    Drop:
    - isFlaggedFraud (low correlation with the target)
    - type (already encoded)
    """
    df = df.copy()
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    df.drop(columns=['isFlaggedFraud', 'type'], inplace=True, errors='ignore')
    return df



# STEP 2 — Filter Transaction Types
def filter_fraud_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only TRANSFER (4) and CASH_OUT (1) because, according to the EDA,
    100% of fraudulent transactions occur only in these two types → reduce noise.
    """
    fraud_type_codes = [1, 4]  # CASH_OUT=1, TRANSFER=4 (after LabelEncoder)
    out = df[df['type_encoded'].isin(fraud_type_codes)].copy()
    print(f"✅ After type filtering: {len(out):,} / {len(df):,} rows")
    return out


# STEP 3 — Split X/y + Train-Test Split (Stratified)
def split_features_target(df):
    return df[FEATURE_COLS], df[TARGET_COL]


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Perform a stratified split — preserve the fraud ratio in both datasets.
    IMPORTANT: always use stratify=y when working with imbalanced data.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"✅ Train {len(X_tr):,} (fraud {y_tr.mean()*100:.3f}%) | "
          f"Test {len(X_te):,} (fraud {y_te.mean()*100:.3f}%)")
    return X_tr, X_te, y_tr, y_te


# BƯỚC 4 — SMOTE + UnderSampling
def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    """
    SMOTE → increase minority class (fraud) up to sampling_strategy * majority
    RandomUnderSampler → reduce majority class → prevent an overly large training set

    IMPORTANT: apply ONLY on TRAIN data, NEVER on the test set!
    Applying it on the test set causes data leakage → artificially inflated results.
    """
    pipeline = ImbPipeline([
        ('over',  SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),
        ('under', RandomUnderSampler(sampling_strategy=0.5,  random_state=random_state)),
    ])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE: Non-fraud {(y_res==0).sum():,} | Fraud {(y_res==1).sum():,} "
          f"({y_res.mean()*100:.2f}%)")
    return X_res, y_res



# COMPLETE PIPELINE
def preprocess_pipeline(df, filter_types=True, apply_resampling=True,
                         test_size=0.2, random_state=42):
    """
    Run the full preprocessing pipeline:
    1. Feature engineering
    2. (optional) Filter TRANSFER & CASH_OUT transactions
    3. Stratified train-test split
    4. (optional) SMOTE on the training set

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
    print("✅ Pipeline completed!\n")
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
