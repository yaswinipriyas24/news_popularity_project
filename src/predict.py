import pandas as pd
import pickle
import sys

def predict_popularity(title_length, keyword_count, description_word_count):
    """
    Loads the trained classification model and predicts popularity for a new article.
    """
    try:
        # Load the trained model
        with open('../models/model.pkl', 'rb') as file:
            model = pickle.load(file)
            
        # Create a DataFrame for the new data point
        new_data = pd.DataFrame([[title_length, keyword_count, description_word_count]], 
                                columns=['title_length', 'keyword_count', 'description_word_count'])
        
        # Make a prediction
        prediction = model.predict(new_data)[0]
        
        # Interpret the prediction
        if prediction == 1:
            return "Popular"
        else:
            return "Not Popular"
        
    except FileNotFoundError:
        print("Error: The model file was not found. Please ensure `model.pkl` exists in the project's models directory.")
        sys.exit(1)

if __name__ == "__main__":
    # Example test data for a new article
    test_title_length = 50
    test_keyword_count = 10
    test_description_word_count = 150
    
    predicted_status = predict_popularity(test_title_length, test_keyword_count, test_description_word_count)
    
    print(f"Based on the model, the article is predicted to be: {predicted_status}")