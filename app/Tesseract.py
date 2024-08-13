import pytesseract
from PIL import Image
import os

class TesseractOCR:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def extract_text(self, image: Image.Image) -> str:
        # Convert image to string using Tesseract
        text = pytesseract.image_to_string(image)
        return text
