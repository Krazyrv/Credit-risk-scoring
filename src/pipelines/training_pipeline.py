"""
Credit Risk Model Training
Train and evaluate credit scoring model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)


def load_data():
    """Load application data."""
    project_root = Path(__file__).parent.parent
    
    for data_dir in ['raw', 'sample']:
        data_path = project_root / 'data' / data_dir / 'applications.csv'
        if data_path.exists():
            df = pd.read_csv(data_path)
            print(f"📂 Loaded data from {data_path}")
            return df
    
    raise FileNotFoundError("Application data not found. Run data_generator.py first.")


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Engineer features for credit scoring."""
    print("🔧 Engineering features...")
    
    df = df.copy()
    encoders = {}
    
    # Ratio features
    df['loan_to_income'] = df['loan_amount'] / df['income']
    df['payment_to_income'] = (df['loan_amount'] / 60) / (df['income'] / 12)  # Est monthly payment
    
    # Interaction features
    df['income_employment'] = df['income'] * np.log1p(df['employment_length'])
    df['credit_per_line'] = df['credit_history_length'] / np.maximum(df['num_credit_lines'], 1)
    
    # Binned features
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                             labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    df['income_bracket'] = pd.cut(df['income'], bins=[0, 30000, 50000, 75000, 100000, 1000000],
                                  labels=['<30K', '30-50K', '50-75K', '75-100K', '100K+'])
    
    # Encode categorical variables
    categorical_cols = ['loan_purpose', 'home_ownership', 'age_group', 'income_bracket']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    return df, encoders


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare feature matrix."""
    
    feature_cols = [
        'age', 'income', 'employment_length', 'loan_amount',
        'debt_to_income', 'credit_history_length', 'num_credit_lines',
        'num_delinquencies', 'utilization_rate',
        'loan_to_income', 'payment_to_income', 'income_employment', 'credit_per_line',
        'loan_purpose_encoded', 'home_ownership_encoded', 'age_group_encoded', 'income_bracket_encoded'
    ]
    
    X = df[feature_cols].values
    y = df['default'].values
    
    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val) -> Tuple:
    """Train credit risk model."""
    print("🔧 Training model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train Gradient Boosting
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=50,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Validation score
    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"   Validation AUC: {val_auc:.3f}")
    
    return model, scaler


def probability_to_score(prob: float) -> int:
    """Convert default probability to credit score (300-850)."""
    # Higher probability of default = lower score
    score = 300 + (1 - prob) * 550
    return int(np.clip(score, 300, 850))


def get_risk_tier(score: int) -> str:
    """Assign risk tier based on score."""
    if score >= 750:
        return 'Excellent'
    elif score >= 700:
        return 'Good'
    elif score >= 650:
        return 'Fair'
    elif score >= 550:
        return 'Poor'
    else:
        return 'Very Poor'


def evaluate_model(model, scaler, X_test, y_test, feature_names: list) -> Dict:
    """Comprehensive model evaluation."""
    print("📊 Evaluating model...")
    
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    # Credit scores
    scores = [probability_to_score(p) for p in y_proba]
    
    # Score distribution by actual default
    score_df = pd.DataFrame({
        'score': scores,
        'actual_default': y_test,
        'predicted_prob': y_proba
    })
    
    # Risk tier analysis
    score_df['risk_tier'] = score_df['score'].apply(get_risk_tier)
    tier_analysis = score_df.groupby('risk_tier').agg({
        'actual_default': ['mean', 'count']
    }).round(3)
    tier_analysis.columns = ['default_rate', 'count']
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'feature_importance': importance,
        'tier_analysis': tier_analysis.to_dict(),
        'score_stats': {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    }


def print_results(metrics: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("CREDIT RISK MODEL RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"   Precision: {metrics['precision']*100:.1f}%")
    print(f"   Recall: {metrics['recall']*100:.1f}%")
    print(f"   F1 Score: {metrics['f1']*100:.1f}%")
    print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.3f}")
    
    print(f"\n📊 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   True Negatives:  {cm[0][0]:,}")
    print(f"   False Positives: {cm[0][1]:,}")
    print(f"   False Negatives: {cm[1][0]:,}")
    print(f"   True Positives:  {cm[1][1]:,}")
    
    print(f"\n📊 Top 5 Features:")
    for i, (feat, imp) in enumerate(list(metrics['feature_importance'].items())[:5], 1):
        print(f"   {i}. {feat}: {imp:.3f}")
    
    print(f"\n📊 Credit Score Distribution:")
    stats = metrics['score_stats']
    print(f"   Mean: {stats['mean']:.0f}")
    print(f"   Std: {stats['std']:.0f}")
    print(f"   Range: {stats['min']:.0f} - {stats['max']:.0f}")


def save_model(model, scaler, encoders, metrics: Dict, feature_names: list):
    """Save model and metadata."""
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model components
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_names
    }, model_dir / 'credit_model.pkl')
    print(f"💾 Saved model to {model_dir / 'credit_model.pkl'}")
    
    # Save metrics
    metrics_to_save = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'roc_auc': float(metrics['roc_auc']),
        'pr_auc': float(metrics['pr_auc']),
        'confusion_matrix': [[int(x) for x in row] for row in metrics['confusion_matrix']],
        'score_stats': {k: float(v) for k, v in metrics['score_stats'].items()}
    }
    
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Save feature importance
    importance_to_save = {k: float(v) for k, v in metrics['feature_importance'].items()}
    with open(model_dir / 'feature_importance.json', 'w') as f:
        json.dump(importance_to_save, f, indent=2)
    
    # Save scorecard parameters
    scorecard = {
        'score_range': {'min': 300, 'max': 850},
        'risk_tiers': {
            'Excellent': {'min': 750, 'max': 850},
            'Good': {'min': 700, 'max': 749},
            'Fair': {'min': 650, 'max': 699},
            'Poor': {'min': 550, 'max': 649},
            'Very Poor': {'min': 300, 'max': 549}
        }
    }
    
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    print(f"💾 Saved metrics to {model_dir}")


def main():
    """Run training pipeline."""
    print("=" * 60)
    print("CREDIT RISK SCORING - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        print("⚠️ Data not found. Generating sample data...")
        from data_generator import main as generate_data
        generate_data()
        df = load_data()
    
    print(f"\n📊 Dataset: {len(df):,} applications, {df['default'].mean()*100:.1f}% default rate")
    
    # Engineer features
    df, encoders = engineer_features(df)
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train):,}")
    print(f"   Validation: {len(X_val):,}")
    print(f"   Test: {len(X_test):,}")
    
    # Train model
    model, scaler = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, scaler, X_test, y_test, feature_names)
    
    # Print results
    print_results(metrics)
    
    # Save model
    save_model(model, scaler, encoders, metrics, feature_names)
    
    print("\n✅ Training complete!")
    
    return model, scaler, metrics


if __name__ == '__main__':
    main()
