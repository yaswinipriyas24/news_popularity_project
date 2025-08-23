# Social Media Popularity Prediction

This project builds a machine learning pipeline to predict whether a news article will be popular on social media. The pipeline is an end-to-end solution that takes raw data, processes it, trains a model, and deploys it as a live API.

***

## Project Pipeline

The project follows a standard machine learning workflow, organized into these key stages:

### 1. Data Ingestion 📥
This stage involves fetching news articles from an external API and simulating social media metrics. The data is saved to the `data/raw` directory for future processing.

### 2. Feature Engineering 🛠️
The raw data is transformed into a clean dataset with numerical features that a machine learning model can understand. This includes extracting attributes like title length and keyword count from the articles.

### 3. Model Training & Comparison 🤖
The cleaned data is used to train and evaluate a Logistic Regression and a Decision Tree classifier. The best-performing model is saved to the `models/` directory for future use.

### 4. Local Deployment 💻
The trained model is exposed as a simple REST API using the Flask web framework. This allows the model to be used by other applications and is the first step toward a live service.

### 5. Public Deployment 🌐
The project is configured for deployment to a public server using files like `requirements.txt` and `Procfile`. This makes the model accessible on the internet for anyone to use.

***

## Getting Started

### Prerequisites

You need the following software installed on your machine:

* Python 3.8+
* pip (Python package installer)
* Git (for version control and deployment)
* A virtual environment manager (e.g., Anaconda)

### Installation

1.  Clone the repository from GitHub.
    ```bash
    git clone https://github.com/yaswinipriyas24/news_popularity_project.git
    cd news-popularity-project
    ```
2.  Create and activate a virtual environment.
    ```bash
    conda create --name news_env python=3.9
    conda activate news_env
    ```
3.  Install the project dependencies.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

Follow these steps in your terminal to run the entire project locally.

1.  **Data Ingestion**:
    ```bash
    python src/data_ingestion.py
    ```
2.  **Feature Engineering**:
    ```bash
    python src/feature_extraction.py
    ```
3.  **Model Training**:
    ```bash
    python src/model_training.py
    ```
4.  **Local Deployment**:
    ```bash
    python app.py
    ```

Your model API will now be live on `http://127.0.0.1:5000`.

***

## How to Use the API

You can test your live API by sending a POST request to the `/predict` endpoint with your article's feature data.

### Example Request (using PowerShell)

```powershell
Invoke-WebRequest -Uri [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) -Method POST -ContentType "application/json" -Body '{"title_length": 50, "keyword_count": 10, "description_word_count": 150}'
