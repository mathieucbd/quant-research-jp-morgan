"""
Credit risk analysis and forecasting expected loss.
-------------------------------------------
This script loads loans data, fits a logistic regression, and calculate the expected loss of a customer.

Usage:

"""

# ------------------------------
# Imports
# ------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix
from sklearn.linear_model import LogisticRegression

np.random.seed(42)


# ------------------------------
# Load and prepare data
# ------------------------------
def load_loan_data(path: str = "data/Task 3 and 4_Loan_Data.csv") -> pd.DataFrame:
    """Load dataset."""
    df = pd.read_csv(path)
    df = df.set_index('customer_id')
    return df

def preprocess_data(df:pd.DataFrame) -> tuple:
    """Peprocess data: split X and y, train-test split, scale features."""
    X = df.drop(columns=['default'])
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, X_test, y_train, y_test


# ------------------------------
# Model training and evaluation
# ------------------------------
def fit_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Fit logistic regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: LogisticRegression, X_test, y_test) -> dict:
    """Evaluate model performance."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    mse = mean_squared_error(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {'AUC': auc, 'MSE': mse, 'Confusion Matrix': cm}

def cross_validate_model(model, X, y):
    """Perform 5-fold cross-validation and return AUC stats."""
    auc_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    return {'AUC (CV mean)': auc_scores.mean(), 'AUC (CV std)': auc_scores.std()}

def display_evaluation(model, X_train, y_train, X_test, y_test):
    """Print model evaluation metrics in a clean format."""
    holdout_results = evaluate_model(model, X_test, y_test)
    cv_results = cross_validate_model(model, X_train, y_train)

    print("\nModel Evaluation Results:")
    print("-" * 40)
    print(f"AUC (Holdout):     {holdout_results['AUC']:.4f}")
    print(f"MSE (Holdout):     {holdout_results['MSE']:.4f}")
    print(f"AUC (CV mean):     {cv_results['AUC (CV mean)']:.4f}")
    print(f"AUC (CV std):      {cv_results['AUC (CV std)']:.4f}")
    print("\nConfusion Matrix (Holdout):")
    print(holdout_results['Confusion Matrix'])
    print("-" * 40)

# ------------------------------
# Expected Loss Calculation
# ------------------------------
def calculate_expected_loss(model, X_input, loan_amount, recovery_rate=0.1) -> float:
    """Calculate expected loss for a given customer."""
    prob_default = model.predict_proba(X_input)[:, 1]   # Probability of Default
    ead = loan_amount                                   # Exposure at Default
    lgd = 1 - recovery_rate                             # Loss Given Default
    el = prob_default * lgd * ead                       # Expected Loss
    return el[0]


if __name__ == '__main__':
    # Load and preprocess data
    df = load_loan_data()
    X_train_scaled, X_test_scaled, X_test_unscaled, y_train, y_test = preprocess_data(df)

    # Fit model
    model = fit_logistic_regression(X_train_scaled, y_train)

    # Evaluate model
    verbose = True
    if verbose:
        display_evaluation(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Calculate expected loss for a sample customer
    sample_customer_idx = 1 # index of the observation in the test set
    loan_data = X_test_scaled[sample_customer_idx].reshape(1, -1)
    loan_amount = X_test_unscaled.loc[X_test_unscaled.index[sample_customer_idx], 'loan_amt_outstanding']
    expected_loss = calculate_expected_loss(model, loan_data, loan_amount)
    print(f"Expected Loss for the sample customer: ${expected_loss:.2f}")