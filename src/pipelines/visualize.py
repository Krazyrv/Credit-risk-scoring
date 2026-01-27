"""
Credit Risk Visualization
Generate charts for model analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib

plt.style.use('seaborn-v0_8-whitegrid')


def plot_score_distribution(df: pd.DataFrame, model_components: dict, save_path: str = None):
    """Plot credit score distribution by default status."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    print("   Computing credit scores...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Load model components
    model = model_components['model']
    scaler = model_components['scaler']
    feature_names = model_components['feature_names']
    
    # Prepare features (simplified version)
    df_temp = df.copy()
    df_temp['loan_to_income'] = df_temp['loan_amount'] / df_temp['income']
    df_temp['payment_to_income'] = (df_temp['loan_amount'] / 60) / (df_temp['income'] / 12)
    df_temp['income_employment'] = df_temp['income'] * np.log1p(df_temp['employment_length'])
    df_temp['credit_per_line'] = df_temp['credit_history_length'] / np.maximum(df_temp['num_credit_lines'], 1)
    
    # Encode categoricals
    for col in ['loan_purpose', 'home_ownership']:
        le = LabelEncoder()
        df_temp[f'{col}_encoded'] = le.fit_transform(df_temp[col].astype(str))
    
    df_temp['age_group'] = pd.cut(df_temp['age'], bins=[0, 25, 35, 45, 55, 100], 
                                  labels=[0, 1, 2, 3, 4])
    df_temp['income_bracket'] = pd.cut(df_temp['income'], bins=[0, 30000, 50000, 75000, 100000, 1000000],
                                       labels=[0, 1, 2, 3, 4])
    df_temp['age_group_encoded'] = df_temp['age_group'].astype(int)
    df_temp['income_bracket_encoded'] = df_temp['income_bracket'].astype(int)
    
    # Get features in correct order
    available_features = [f for f in feature_names if f in df_temp.columns]
    X = df_temp[available_features].values
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    
    # Convert to scores
    scores = 300 + (1 - proba) * 550
    scores = np.clip(scores, 300, 850)
    
    # Plot distribution by default status
    for default_val, color, label in [(0, '#27ae60', 'Non-Default'), (1, '#e74c3c', 'Default')]:
        mask = df['default'] == default_val
        axes[0].hist(scores[mask], bins=30, alpha=0.6, color=color, label=label, density=True)
    
    axes[0].set_xlabel('Credit Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Credit Score Distribution by Default Status', fontweight='bold')
    axes[0].legend()
    
    # Default rate by score band
    score_bands = pd.cut(scores, bins=[300, 500, 550, 600, 650, 700, 750, 800, 850])
    score_df = pd.DataFrame({'score_band': score_bands, 'default': df['default']})
    default_by_band = score_df.groupby('score_band')['default'].mean() * 100
    
    x_labels = [str(b) for b in default_by_band.index]
    axes[1].bar(range(len(default_by_band)), default_by_band.values, color='#e74c3c')
    axes[1].set_xticks(range(len(default_by_band)))
    axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1].set_xlabel('Score Band')
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].set_title('Default Rate by Score Band', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def plot_feature_importance(feature_importance: dict, save_path: str = None):
    """Plot feature importance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get top 10 features
    top_features = dict(list(feature_importance.items())[:10])
    
    features = list(top_features.keys())
    importance = list(top_features.values())
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, color=colors[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(importance):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def plot_confusion_matrix(metrics: dict, save_path: str = None):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = np.array(metrics['confusion_matrix'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'], ax=ax)
    
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def plot_metrics_summary(metrics: dict, save_path: str = None):
    """Plot model metrics summary."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Classification metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [
        metrics['accuracy'] * 100,
        metrics['precision'] * 100,
        metrics['recall'] * 100,
        metrics['f1'] * 100
    ]
    
    colors = ['#27ae60', '#3498db', '#9b59b6', '#e74c3c']
    bars = axes[0].bar(metric_names, metric_values, color=colors)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_title('Classification Metrics', fontweight='bold')
    
    for bar, val in zip(bars, metric_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', fontweight='bold')
    
    # AUC metrics
    auc_names = ['ROC-AUC', 'PR-AUC']
    auc_values = [metrics['roc_auc'], metrics['pr_auc']]
    
    axes[1].bar(auc_names, auc_values, color=['#3498db', '#9b59b6'])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Score')
    axes[1].set_title('AUC Metrics', fontweight='bold')
    
    for i, val in enumerate(auc_values):
        axes[1].text(i, val + 0.02, f'{val:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def plot_default_by_features(df: pd.DataFrame, save_path: str = None):
    """Plot default rate by key features."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Default rate by income bracket
    income_bins = pd.cut(df['income'], bins=[0, 30000, 50000, 75000, 100000, 500000],
                        labels=['<30K', '30-50K', '50-75K', '75-100K', '100K+'])
    default_by_income = df.groupby(income_bins)['default'].mean() * 100
    axes[0, 0].bar(default_by_income.index.astype(str), default_by_income.values, color='#3498db')
    axes[0, 0].set_xlabel('Income Bracket')
    axes[0, 0].set_ylabel('Default Rate (%)')
    axes[0, 0].set_title('Default Rate by Income', fontweight='bold')
    
    # Default rate by delinquencies
    default_by_delinq = df.groupby('num_delinquencies')['default'].mean() * 100
    axes[0, 1].bar(default_by_delinq.index.astype(str), default_by_delinq.values, color='#e74c3c')
    axes[0, 1].set_xlabel('Number of Delinquencies')
    axes[0, 1].set_ylabel('Default Rate (%)')
    axes[0, 1].set_title('Default Rate by Delinquencies', fontweight='bold')
    
    # Default rate by DTI
    dti_bins = pd.cut(df['debt_to_income'], bins=[0, 0.2, 0.3, 0.4, 0.5, 1.0],
                     labels=['<20%', '20-30%', '30-40%', '40-50%', '50%+'])
    default_by_dti = df.groupby(dti_bins)['default'].mean() * 100
    axes[1, 0].bar(default_by_dti.index.astype(str), default_by_dti.values, color='#f39c12')
    axes[1, 0].set_xlabel('Debt-to-Income Ratio')
    axes[1, 0].set_ylabel('Default Rate (%)')
    axes[1, 0].set_title('Default Rate by DTI', fontweight='bold')
    
    # Default rate by employment length
    emp_bins = pd.cut(df['employment_length'], bins=[-1, 1, 3, 5, 10, 50],
                     labels=['<1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr'])
    default_by_emp = df.groupby(emp_bins)['default'].mean() * 100
    axes[1, 1].bar(default_by_emp.index.astype(str), default_by_emp.values, color='#27ae60')
    axes[1, 1].set_xlabel('Employment Length')
    axes[1, 1].set_ylabel('Default Rate (%)')
    axes[1, 1].set_title('Default Rate by Employment', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def generate_all_plots():
    """Generate all visualization plots."""
    print("\n📊 Generating visualizations...")
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'docs' / 'img'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(project_root / 'data' / 'raw' / 'applications.csv')
    
    # Load metrics
    model_dir = project_root / 'models'
    
    with open(model_dir / 'metrics.json', 'r') as f:
        metrics = json.load(f)
    
    with open(model_dir / 'feature_importance.json', 'r') as f:
        feature_importance = json.load(f)
    
    # Load model for score distribution
    model_components = joblib.load(model_dir / 'credit_model.pkl')
    
    # Generate plots
    plot_score_distribution(df, model_components, output_dir / 'score_distribution.png')
    plot_feature_importance(feature_importance, output_dir / 'feature_importance.png')
    plot_confusion_matrix(metrics, output_dir / 'confusion_matrix.png')
    plot_metrics_summary(metrics, output_dir / 'metrics_summary.png')
    plot_default_by_features(df, output_dir / 'default_by_features.png')
    
    print(f"\n✅ All visualizations saved to {output_dir}")


def main():
    generate_all_plots()


if __name__ == "__main__":
    main()
