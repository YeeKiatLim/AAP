from flask import Flask
from flask import jsonify
from flask import request
from joblib import load

app = Flask(__name__)

classifier = load("models/smsmodel.joblib")
vectorizer = load("models/vectorizer.joblib")

@app.route("/")
def hello():
    return "Navigate to the home page"

@app.route("/classifySMS", methods=["POST"])
def classifySpam():
    text = None

    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        try:
            text = file.read().decode("utf-8")
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif "text" in request.form:
        text = request.form["text"]
    else:
        return jsonify({"error": "No input provided"}), 400

    if text is None or text.strip() == "":
        return jsonify({"error": "Empty input"}), 400

    try:
        X = vectorizer.transform([text])
        result = classifier.predict(X)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    target = ["ham", "spam"]
    return jsonify({"result": target[result[0]]})

if __name__ == "__main__":
    app.run(debug=False)
