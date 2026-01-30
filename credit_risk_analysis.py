# =============================================================================
# CREDIT RISK PREDICTION MODEL
# German Credit Dataset Analysis
# Author: Burak Aktaş
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*60)
print("CREDIT RISK PREDICTION MODEL")
print("German Credit Dataset Analysis")
print("="*60)

# =============================================================================
# 1. DATA LOADING & INITIAL EXPLORATION
# =============================================================================
print("\n" + "="*60)
print("1. DATA LOADING & INITIAL EXPLORATION")
print("="*60)

df = pd.read_csv('german_credit_risk.csv', index_col=0)

print(f"\nDataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- Target Variable Distribution ---")
print(df['Risk'].value_counts())
print(f"\nDefault Rate: {(df['Risk']=='bad').mean()*100:.1f}%")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "="*60)
print("2. EXPLORATORY DATA ANALYSIS")
print("="*60)

# Create figures directory
import os
os.makedirs('figures', exist_ok=True)

# 2.1 Target Distribution
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
df['Risk'].value_counts().plot(kind='bar', color=colors, ax=ax, edgecolor='black')
ax.set_title('Credit Risk Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Risk Category')
ax.set_ylabel('Count')
ax.set_xticklabels(['Good (Paid)', 'Bad (Default)'], rotation=0)
for i, v in enumerate(df['Risk'].value_counts().values):
    ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/01_target_distribution.png', dpi=150)
plt.close()
print("✓ Saved: figures/01_target_distribution.png")

# 2.2 Age Distribution by Risk
fig, ax = plt.subplots(figsize=(10, 5))
df[df['Risk']=='good']['Age'].hist(alpha=0.7, bins=20, label='Good', color='#2ecc71', ax=ax)
df[df['Risk']=='bad']['Age'].hist(alpha=0.7, bins=20, label='Bad', color='#e74c3c', ax=ax)
ax.set_title('Age Distribution by Credit Risk', fontsize=14, fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('figures/02_age_distribution.png', dpi=150)
plt.close()
print("✓ Saved: figures/02_age_distribution.png")

# 2.3 Credit Amount Distribution by Risk
fig, ax = plt.subplots(figsize=(10, 5))
df[df['Risk']=='good']['Credit amount'].hist(alpha=0.7, bins=30, label='Good', color='#2ecc71', ax=ax)
df[df['Risk']=='bad']['Credit amount'].hist(alpha=0.7, bins=30, label='Bad', color='#e74c3c', ax=ax)
ax.set_title('Credit Amount Distribution by Risk', fontsize=14, fontweight='bold')
ax.set_xlabel('Credit Amount (DM)')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('figures/03_credit_amount_distribution.png', dpi=150)
plt.close()
print("✓ Saved: figures/03_credit_amount_distribution.png")

# 2.4 Purpose Analysis
fig, ax = plt.subplots(figsize=(12, 6))
purpose_risk = pd.crosstab(df['Purpose'], df['Risk'], normalize='index') * 100
purpose_risk['bad'].sort_values(ascending=True).plot(kind='barh', color='#e74c3c', ax=ax)
ax.set_title('Default Rate by Loan Purpose', fontsize=14, fontweight='bold')
ax.set_xlabel('Default Rate (%)')
ax.set_ylabel('Purpose')
plt.tight_layout()
plt.savefig('figures/04_purpose_default_rate.png', dpi=150)
plt.close()
print("✓ Saved: figures/04_purpose_default_rate.png")

# 2.5 Housing Analysis
fig, ax = plt.subplots(figsize=(8, 5))
housing_risk = pd.crosstab(df['Housing'], df['Risk'], normalize='index') * 100
housing_risk.plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=ax, edgecolor='black')
ax.set_title('Risk Distribution by Housing Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Housing Type')
ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Good', 'Bad'])
plt.tight_layout()
plt.savefig('figures/05_housing_risk.png', dpi=150)
plt.close()
print("✓ Saved: figures/05_housing_risk.png")

# 2.6 Correlation Heatmap (numeric features)
fig, ax = plt.subplots(figsize=(10, 8))
df_numeric = df.select_dtypes(include=[np.number])
correlation = df_numeric.corr()
sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0, ax=ax, fmt='.2f')
ax.set_title('Correlation Matrix - Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_correlation_matrix.png', dpi=150)
plt.close()
print("✓ Saved: figures/06_correlation_matrix.png")

# 2.7 Saving Accounts vs Risk
fig, ax = plt.subplots(figsize=(10, 5))
saving_risk = pd.crosstab(df['Saving accounts'].fillna('unknown'), df['Risk'], normalize='index') * 100
saving_risk['bad'].sort_values(ascending=True).plot(kind='barh', color='#e74c3c', ax=ax)
ax.set_title('Default Rate by Saving Accounts Level', fontsize=14, fontweight='bold')
ax.set_xlabel('Default Rate (%)')
ax.set_ylabel('Saving Accounts')
plt.tight_layout()
plt.savefig('figures/07_saving_accounts_risk.png', dpi=150)
plt.close()
print("✓ Saved: figures/07_saving_accounts_risk.png")

# 2.8 Checking Account vs Risk
fig, ax = plt.subplots(figsize=(10, 5))
checking_risk = pd.crosstab(df['Checking account'].fillna('unknown'), df['Risk'], normalize='index') * 100
checking_risk['bad'].sort_values(ascending=True).plot(kind='barh', color='#e74c3c', ax=ax)
ax.set_title('Default Rate by Checking Account Level', fontsize=14, fontweight='bold')
ax.set_xlabel('Default Rate (%)')
ax.set_ylabel('Checking Account')
plt.tight_layout()
plt.savefig('figures/08_checking_account_risk.png', dpi=150)
plt.close()
print("✓ Saved: figures/08_checking_account_risk.png")

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
print("\n" + "="*60)
print("3. DATA PREPROCESSING")
print("="*60)

# Create a copy for modeling
df_model = df.copy()

# 3.1 Handle Missing Values
print("\n--- Handling Missing Values ---")
print(f"Missing in 'Saving accounts': {df_model['Saving accounts'].isnull().sum()}")
print(f"Missing in 'Checking account': {df_model['Checking account'].isnull().sum()}")

df_model['Saving accounts'] = df_model['Saving accounts'].fillna('unknown')
df_model['Checking account'] = df_model['Checking account'].fillna('unknown')
print("✓ Filled missing values with 'unknown'")

# 3.2 Encode Target Variable
df_model['Risk_Binary'] = (df_model['Risk'] == 'bad').astype(int)
print(f"\n✓ Target encoded: good=0, bad=1")

# 3.3 Feature Engineering
print("\n--- Feature Engineering ---")

# Age groups
df_model['Age_Group'] = pd.cut(df_model['Age'], bins=[0, 25, 35, 45, 55, 100], 
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])

# Credit amount groups
df_model['Credit_Group'] = pd.cut(df_model['Credit amount'], bins=[0, 1000, 3000, 5000, 10000, 20000],
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Duration groups (months)
df_model['Duration_Group'] = pd.cut(df_model['Duration'], bins=[0, 12, 24, 36, 48, 100],
                                     labels=['<1 year', '1-2 years', '2-3 years', '3-4 years', '4+ years'])

# Credit per month (installment estimate)
df_model['Credit_Per_Month'] = df_model['Credit amount'] / df_model['Duration']

print("✓ Created: Age_Group, Credit_Group, Duration_Group, Credit_Per_Month")

# 3.4 Encode Categorical Variables
print("\n--- Encoding Categorical Variables ---")

categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose',
                   'Age_Group', 'Credit_Group', 'Duration_Group']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col + '_Encoded'] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded: {col}")

# 3.5 Prepare Final Feature Set
feature_cols = ['Age', 'Job', 'Credit amount', 'Duration', 'Credit_Per_Month',
                'Sex_Encoded', 'Housing_Encoded', 'Saving accounts_Encoded',
                'Checking account_Encoded', 'Purpose_Encoded']

X = df_model[feature_cols]
y = df_model['Risk_Binary']

print(f"\n--- Final Dataset ---")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"Target Distribution: {y.value_counts().to_dict()}")

# 3.6 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 3.7 Handle Class Imbalance with SMOTE
print("\n--- Handling Class Imbalance ---")
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# 3.8 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled with StandardScaler")

# =============================================================================
# 4. MODEL BUILDING & COMPARISON
# =============================================================================
print("\n" + "="*60)
print("4. MODEL BUILDING & COMPARISON")
print("="*60)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate models
results = []

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Use scaled data for Logistic Regression, original for tree-based
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    })
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

# Results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AUC-ROC', ascending=False)
print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Saved: model_comparison_results.csv")

# =============================================================================
# 5. BEST MODEL ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("5. BEST MODEL ANALYSIS")
print("="*60)

best_model_name = results_df.iloc[0]['Model']
print(f"\nBest Model: {best_model_name}")

# Retrain best model for detailed analysis
if best_model_name == 'Logistic Regression':
    best_model = LogisticRegression(random_state=42, max_iter=1000)
    best_model.fit(X_train_scaled, y_train_balanced)
    y_pred_best = best_model.predict(X_test_scaled)
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test)[:, 1]

# 5.1 Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('figures/09_confusion_matrix.png', dpi=150)
plt.close()
print("✓ Saved: figures/09_confusion_matrix.png")

# 5.2 ROC Curve Comparison
fig, ax = plt.subplots(figsize=(10, 8))
for name, model in models.items():
    if name == 'Logistic Regression':
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/10_roc_curve_comparison.png', dpi=150)
plt.close()
print("✓ Saved: figures/10_roc_curve_comparison.png")

# 5.3 Feature Importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    importance.plot(kind='barh', x='Feature', y='Importance', ax=ax, color='#3498db', legend=False)
    ax.set_title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('figures/11_feature_importance.png', dpi=150)
    plt.close()
    print("✓ Saved: figures/11_feature_importance.png")

# Also create feature importance for Random Forest specifically
rf_model = models['Random Forest']
fig, ax = plt.subplots(figsize=(10, 6))
importance_rf = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

importance_rf.plot(kind='barh', x='Feature', y='Importance', ax=ax, color='#27ae60', legend=False)
ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('figures/11_feature_importance_rf.png', dpi=150)
plt.close()
print("✓ Saved: figures/11_feature_importance_rf.png")

# 5.4 Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_best, target_names=['Good', 'Bad']))

# =============================================================================
# 6. CREDIT RISK SCORING SYSTEM
# =============================================================================
print("\n" + "="*60)
print("6. CREDIT RISK SCORING SYSTEM")
print("="*60)

# Create risk scores (0-1000 scale, higher = better)
df_model['Risk_Probability'] = models['Random Forest'].predict_proba(X)[:, 1]
df_model['Credit_Score'] = ((1 - df_model['Risk_Probability']) * 1000).astype(int)

# Risk categories
def assign_risk_category(score):
    if score >= 750:
        return 'Excellent'
    elif score >= 650:
        return 'Good'
    elif score >= 550:
        return 'Fair'
    elif score >= 450:
        return 'Poor'
    else:
        return 'Very Poor'

df_model['Risk_Category'] = df_model['Credit_Score'].apply(assign_risk_category)

print("\n--- Credit Score Distribution ---")
print(df_model['Risk_Category'].value_counts())

# 6.1 Credit Score Distribution
fig, ax = plt.subplots(figsize=(10, 6))
df_model['Credit_Score'].hist(bins=30, ax=ax, color='#3498db', edgecolor='black')
ax.axvline(x=750, color='#27ae60', linestyle='--', linewidth=2, label='Excellent (750+)')
ax.axvline(x=650, color='#f39c12', linestyle='--', linewidth=2, label='Good (650+)')
ax.axvline(x=550, color='#e67e22', linestyle='--', linewidth=2, label='Fair (550+)')
ax.axvline(x=450, color='#e74c3c', linestyle='--', linewidth=2, label='Poor (450+)')
ax.set_title('Credit Score Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Credit Score')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('figures/12_credit_score_distribution.png', dpi=150)
plt.close()
print("✓ Saved: figures/12_credit_score_distribution.png")

# 6.2 Risk Category by Actual Risk
fig, ax = plt.subplots(figsize=(10, 6))
category_risk = pd.crosstab(df_model['Risk_Category'], df_model['Risk'], normalize='index') * 100
category_order = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
category_risk = category_risk.reindex(category_order)
category_risk.plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=ax, edgecolor='black')
ax.set_title('Actual Default Rate by Risk Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Risk Category')
ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend(['Good', 'Bad'])
plt.tight_layout()
plt.savefig('figures/13_category_validation.png', dpi=150)
plt.close()
print("✓ Saved: figures/13_category_validation.png")

# Save scored dataset
df_model.to_csv('german_credit_scored.csv', index=False)
print("✓ Saved: german_credit_scored.csv")

# =============================================================================
# 7. MODEL EXPORT
# =============================================================================
print("\n" + "="*60)
print("7. MODEL EXPORT")
print("="*60)

import pickle

# Save the best model and preprocessing objects
model_artifacts = {
    'model': models['Random Forest'],
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_cols': feature_cols
}

with open('credit_risk_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("✓ Saved: credit_risk_model.pkl")

# =============================================================================
# 8. SUMMARY
# =============================================================================
print("\n" + "="*60)
print("8. SUMMARY")
print("="*60)

print(f"""
PROJECT SUMMARY
===============

Dataset: German Credit Risk Dataset
Samples: 1,000 customers
Features: 10 (after engineering)

Best Model: {best_model_name}
- AUC-ROC: {results_df.iloc[0]['AUC-ROC']:.4f}
- Accuracy: {results_df.iloc[0]['Accuracy']:.4f}
- F1-Score: {results_df.iloc[0]['F1-Score']:.4f}

Key Insights:
1. Checking account status is the strongest predictor of default
2. Credit amount and duration significantly impact risk
3. Purpose of loan (e.g., vacation/others) shows higher default rates
4. Younger customers tend to have slightly higher default rates

Files Generated:
- figures/ (13 visualization files)
- model_comparison_results.csv
- german_credit_scored.csv
- credit_risk_model.pkl
""")

print("="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
