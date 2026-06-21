# src/models.py — Define model, training, and evaluation

import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score)
from xgboost import XGBClassifier

MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')


# MODEL DEFINITIONS
def get_model(model_name: str, random_state=42):
    """
    Return a model object based on name.
    Options: 'xgboost', 'random_forest', 'logistic_regression'

    Parameters are tuned for fraud detection:
    - XGBoost: scale_pos_weight=100 to further handle imbalance
    - RF/LR: class_weight='balanced'
    """
    models = {
        'xgboost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=100,   # weight for minority class
            eval_metric='aucpr',    # PR-AUC better than accuracy for imbalanced data
            random_state=random_state, n_jobs=-1,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state, n_jobs=-1,
        ),
        'logistic_regression': LogisticRegression(
            class_weight='balanced', max_iter=1000,
            random_state=random_state, n_jobs=-1,
        ),
    }

    if model_name not in models:
        raise ValueError(f"Invalid model '{model_name}'. Choose from: {list(models)}")

    return models[model_name]


# TRAIN & SAVE
def train_model(model_name, X_train, y_train, save=True, random_state=42):
    """
    Train a model and save it to models/<model_name>.joblib.
    After training, use load_model() to reload without retraining.
    """
    print(f"🚀 Training {model_name}...")
    model = get_model(model_name, random_state)
    model.fit(X_train, y_train)
    print(f"✅ Done!")

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
        joblib.dump(model, path)
        print(f"   💾 Saved: {path}")

    return model


def load_model(model_name):
    """Load a trained model from disk."""
    path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{model_name}' not found — please run train_model() first.")
    return joblib.load(path)



# EVALUATION
def evaluate_model(model, X_test, y_test, model_name='model', save_figures=True):
    """
    Evaluate model using metrics suitable for imbalanced classification:
    - Classification Report (Precision, Recall, F1 per class)
    - PR-AUC ← MAIN METRIC (better than ROC-AUC for imbalanced data)
    - ROC-AUC
    - Confusion Matrix, ROC Curve, PR Curve

    In fraud detection: prioritize high Recall for Fraud class
    (catch more fraud) and reasonable Precision (avoid false alarms).

    Returns dict {model_name, roc_auc, pr_auc, f1_score}
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}\nEVALUATION: {model_name.upper()}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Legitimate','Fraud']))

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc  = average_precision_score(y_test, y_pred_prob)
    f1      = f1_score(y_test, y_pred)

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}  ← preferred metric for imbalanced data")
    print(f"F1-Score: {f1:.4f}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    _plot_confusion_matrix(y_test, y_pred, model_name, save_figures)
    _plot_roc_pr(y_test, y_pred_prob, model_name, roc_auc, pr_auc, save_figures)

    return dict(model_name=model_name, roc_auc=roc_auc, pr_auc=pr_auc, f1_score=f1)


def _plot_confusion_matrix(y_test, y_pred, name, save):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Legitimate','Fraud'],
        yticklabels=['Legitimate','Fraud'],
        ax=ax
    )
    ax.set(
        title=f'Confusion Matrix — {name}',
        ylabel='Actual',
        xlabel='Predicted'
    )
    plt.tight_layout()

    if save:
        p = os.path.join(FIGURES_DIR, f'confusion_matrix_{name}.png')
        plt.savefig(p, dpi=150)
        print(f"   📸 {p}")

    plt.show()


def _plot_roc_pr(y_test, y_prob, name, roc_auc, pr_auc, save):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.3f}')
    axes[0].plot([0,1],[0,1],'k--')
    axes[0].set(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title=f'ROC Curve — {name}'
    )
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, color='steelblue', lw=2, label=f'AP={pr_auc:.3f}')
    axes[1].set(
        xlabel='Recall',
        ylabel='Precision',
        title=f'PR Curve — {name}'
    )
    axes[1].legend()

    plt.tight_layout()

    if save:
        p = os.path.join(FIGURES_DIR, f'roc_pr_{name}.png')
        plt.savefig(p, dpi=150)
        print(f"   📸 {p}")

    plt.show()


def compare_models(results: list) -> pd.DataFrame:
    """
    Compare multiple models and sort by PR-AUC descending.

    Example
    -------
    >>> r1 = evaluate_model(xgb, X_test, y_test, 'xgboost')
    >>> r2 = evaluate_model(rf,  X_test, y_test, 'random_forest')
    >>> compare_models([r1, r2])
    """
    df = pd.DataFrame(results).set_index('model_name').sort_values('pr_auc', ascending=False)
    print("\n📋 Model comparison:\n", df.to_string())
    return df
