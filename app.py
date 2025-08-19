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


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Extract the features from the request data
        title_length = data['title_length']
        keyword_count = data['keyword_count']
        description_word_count = data['description_word_count']
        
        # Create a DataFrame for the new data point
        new_data = pd.DataFrame([[title_length, keyword_count, description_word_count]], 
                                columns=['title_length', 'keyword_count', 'description_word_count'])
        
        # Make a prediction
        prediction = model.predict(new_data)[0]
        
        # Interpret the prediction
        result = "Popular" if prediction == 1 else "Not Popular"
        
        # Return the result as a JSON response
        return jsonify({
            "prediction": result,
            "title_length": title_length,
            "keyword_count": keyword_count,
            "description_word_count": description_word_count
        })

    except KeyError as e:
        return jsonify({"error": f"Missing key in request data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)