import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model once when the application starts
try:
    with open('models/model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("Error: The model file was not found. Please ensure `model.pkl` exists in the 'models' directory.")

def get_explanation(title_length, keyword_count, description_word_count, prediction):
    """Generates a human-readable explanation for the prediction."""
    explanation = "The prediction is based on the following factors: "
    factors = []

    # Explain the popularity prediction based on feature values
    if prediction == 1: # Popular
        if title_length > 30:
            factors.append("a long title")
        if keyword_count > 5:
            factors.append("a high number of keywords")
        if description_word_count > 100:
            factors.append("a detailed description")
        if not factors:
            explanation = "The prediction is based on a combination of factors."
        else:
            explanation += ", ".join(factors) + "."
    else: # Not Popular
        if title_length < 20:
            factors.append("a short title")
        if keyword_count < 3:
            factors.append("a low number of keywords")
        if description_word_count < 50:
            factors.append("a short description")
        if not factors:
            explanation = "The prediction is based on a combination of factors."
        else:
            explanation += ", ".join(factors) + "."
    return explanation

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        title_length = data['title_length']
        keyword_count = data['keyword_count']
        description_word_count = data['description_word_count']
        
        new_data = pd.DataFrame([[title_length, keyword_count, description_word_count]], 
                                columns=['title_length', 'keyword_count', 'description_word_count'])
        
        prediction = model.predict(new_data)[0]
        
        result = "Popular" if prediction == 1 else "Not Popular"
        explanation = get_explanation(title_length, keyword_count, description_word_count, prediction)
        
        return jsonify({
            "prediction": result,
            "explanation": explanation,
            "title_length": title_length,
            "keyword_count": keyword_count,
            "description_word_count": description_word_count
        })

    except KeyError as e:
        return jsonify({"error": f"Missing key in request data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
