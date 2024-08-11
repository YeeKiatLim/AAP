from flask import Flask, render_template, request, jsonify
from Tesseract import TesseractOCR
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import inference
import io
import os
import base64

app = Flask(__name__)
app.debug = True
app.config['ROBOFLOW_API_KEY'] = os.getenv('ROBOFLOW_API_KEY')

load_dotenv()
inference.api_key = app.config['ROBOFLOW_API_KEY']

roboflow_model = inference.get_model("asdfasdfasdfasdfasdf/1", api_key=app.config['ROBOFLOW_API_KEY'])
ocr_model = TesseractOCR()

# Home Page
@app.route('/', methods=['GET', 'POST'])
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
                # Open the image
                image = Image.open(io.BytesIO(file.read()))

                # Perform object detection
                results = roboflow_model.infer(image)

                # Debug: Print results
                print(results)

                # Initialize an empty list to store detection results
                detection_results = []

                # Create a drawing context
                draw = ImageDraw.Draw(image)

                # Load a font for drawing text (if you have a specific font)
                font = ImageFont.load_default()

                # Iterate through each prediction object in the results
                for response in results:
                    for prediction in response.predictions:
                        # Extract the bounding box coordinates
                        x = prediction.x
                        y = prediction.y
                        width = prediction.width
                        height = prediction.height
                        class_name = prediction.class_name
                        confidence = prediction.confidence

                        # Calculate bounding box
                        left = max(0, x - width / 2)
                        top = max(0, y - height / 2)
                        right = min(image.width, x + width / 2)
                        bottom = min(image.height, y + height / 2)

                        # Draw the bounding box on the image
                        draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)

                        # Crop the detected area from the image
                        cropped_image = image.crop((left, top, right, bottom))

                        # Use Tesseract OCR to extract text from the cropped image
                        extracted_text = ocr_model.extract_text(cropped_image)
                        extracted_texts.append({
                            'text': extracted_text,
                            'class': class_name,
                            'confidence': confidence,
                            'box': (left, top, right, bottom)
                        })

                        # Append the relevant data to detection_results
                        detection_results.append({
                            'class': class_name,
                            'score': confidence,
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height
                        })

                # Convert the image with bounding boxes to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                processed_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            except Exception as e:
                error = f"An error occurred: {str(e)}"

    return render_template('index.html', detection_results=detection_results, extracted_texts=extracted_texts, processed_image=processed_image, error=error)


if __name__ == '__main__':
    app.run()

