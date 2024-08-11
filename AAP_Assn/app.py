from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib 
import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app)


# Load the trained model, TF-IDF vectorizer, and Label Encoder
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    X = tfidf.transform([text]).toarray()
    prediction = model.predict(X)
    prediction_proba = model.predict_proba(X)
    label = label_encoder.inverse_transform(prediction)[0]
    ham_prob = prediction_proba[0][0] * 100
    spam_prob = prediction_proba[0][1] * 100
    return jsonify({'prediction': label, 'Probability of Legitimacy': ham_prob, 'Probability of Scam': spam_prob})

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
