import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_model():
    """Trains and compares Logistic Regression and Decision Tree models."""
    # Load the processed data
    df = pd.read_csv('../data/processed/featured_data.csv')
    
    # Define features (X) and target (y)
    X = df[['title_length', 'keyword_count', 'description_word_count']]
    y = df['is_popular']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Train and Evaluate Logistic Regression ---
    print("Training Logistic Regression Model...")
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train, y_train)
    log_reg_predictions = log_reg_model.predict(X_test)
    log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
    
    print("\n--- Logistic Regression Results ---")
    print(f"Accuracy: {log_reg_accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, log_reg_predictions))

    # --- Train and Evaluate Decision Tree ---
    print("\nTraining Decision Tree Model...")
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)
    tree_accuracy = accuracy_score(y_test, tree_predictions)
    
    print("\n--- Decision Tree Results ---")
    print(f"Accuracy: {tree_accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, tree_predictions))

    # --- Save the best performing model ---
    if tree_accuracy > log_reg_accuracy:
        best_model = tree_model
        model_name = "Decision Tree"
    else:
        best_model = log_reg_model
        model_name = "Logistic Regression"
    
    with open('../models/model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print(f"\nBest performing model ({model_name}) has been saved to models/model.pkl")

if __name__ == "__main__":
    train_model()