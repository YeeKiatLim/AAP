from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import os

def download_model(model_path, model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    processor = AutoProcessor.from_pretrained("PaidDatasetsBad/bad-whisper")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("PaidDatasetsBad/bad-whisper")

    # Save the model and processor to the specified directory
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)

download_model('models/', 'PaidDatasetsBad/bad-whisper')