from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.pipelines.audio_utils import ffmpeg_read
import joblib 
import numpy as np

def get_model(model_path):
    """Load a Hugging Face model and tokenizer from the specified directory"""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
    return model, processor

# Load the model and processor
asr_model, asr_processor = get_model('models/')

model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    """Translate text from one language to another. This function is 
    called when a POST request is sent to /translate/<from_lang>/<to_lang>/"""

    if 'audio_file' in request.files:
        audio_file = request.files['audio_file']
        sampling_rate = asr_processor.feature_extractor.sampling_rate

        # load as bytes
        inputs = audio_file.read()

        # read bytes as array
        inputs = ffmpeg_read(inputs, sampling_rate=sampling_rate)

        input_features = asr_processor(inputs, sampling_rate=sampling_rate, return_tensors="pt").input_features 
        predicted_ids = asr_model.generate(input_features)
        transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return jsonify({f'text': transcription})
    else:
        return jsonify({'error': 'Text to translate not provided'}), 400

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)