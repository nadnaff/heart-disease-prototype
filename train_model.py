"""
Heart Disease Prediction - Model Training Script
=================================================
This script trains the machine learning model for heart disease prediction.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='heart_disease.csv'):
    """Load and prepare the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Remove duplicates if any
    df = df.drop_duplicates()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

def prepare_features(df):
    """Split features and target"""
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def train_models(X_train, y_train, X_test, y_test):
    """Train and compare multiple models"""
    print("\nTraining multiple models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    return results[best_model_name]['model'], best_model_name

def tune_hyperparameters(model, model_name, X_train, y_train):
    """Tune hyperparameters using GridSearch"""
    print(f"\nTuning hyperparameters for {model_name}...")
    
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    }
    
    if model_name in param_grids:
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        return model

def evaluate_model(model, X_test, y_test):
    """Detailed model evaluation"""
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, scaler, model_path='heart_disease_model.pkl', scaler_path='scaler.pkl'):
    """Save trained model and scaler"""
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Saving scaler to {scaler_path}...")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully!")

def main():
    """Main training pipeline"""
    print("="*60)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    best_model, model_name = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Tune hyperparameters
    tuned_model = tune_hyperparameters(best_model, model_name, X_train_scaled, y_train)
    
    # Final evaluation
    evaluate_model(tuned_model, X_test_scaled, y_test)
    
    # Save model
    save_model(tuned_model, scaler)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run the Streamlit app with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
