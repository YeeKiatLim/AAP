from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.pipelines.audio_utils import ffmpeg_read
import joblib 
import numpy as np
from Tesseract import TesseractOCR
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import inference
import io
import os
import base64

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
app.config['ROBOFLOW_API_KEY'] = os.getenv('ROBOFLOW_API_KEY')
CORS(app)

load_dotenv()
inference.api_key = app.config['ROBOFLOW_API_KEY']

roboflow_model = inference.get_model("asdfasdfasdfasdfasdf/1", api_key=app.config['ROBOFLOW_API_KEY'])
ocr_model = TesseractOCR()

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

# Elijah's Code
@app.route('/elijah', methods=['GET', 'POST'])
def index():
    detection_results = None
    extracted_texts = []
    processed_image = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            error = "No file selected"
        else:
            file = request.files['file']
            try:
                image = Image.open(io.BytesIO(file.read()))
                results = roboflow_model.infer(image)
                print(results)
                detection_results = []
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()

                for response in results:
                    for prediction in response.predictions:
                        x = prediction.x
                        y = prediction.y
                        width = prediction.width
                        height = prediction.height
                        class_name = prediction.class_name
                        confidence = prediction.confidence

                        left = max(0, x - width / 2)
                        top = max(0, y - height / 2)
                        right = min(image.width, x + width / 2)
                        bottom = min(image.height, y + height / 2)

                        draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)

                        cropped_image = image.crop((left, top, right, bottom))

                        extracted_text = ocr_model.extract_text(cropped_image)
                        extracted_texts.append({
                            'text': extracted_text,
                            'class': class_name,
                            'confidence': confidence,
                            'box': (left, top, right, bottom)
                        })

                        detection_results.append({
                            'class': class_name,
                            'score': confidence,
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height
                        })

                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                processed_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            except Exception as e:
                error = f"An error occurred: {str(e)}"

    return render_template('index.html', detection_results=detection_results, extracted_texts=extracted_texts, processed_image=processed_image, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)