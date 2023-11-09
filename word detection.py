import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
image_val = cv2.imread('Encord.png')
image_val = cv2.cvtColor(image_val, cv2.COLOR_BGR2RGB)

# Detecting Words
himage_val, wimage_val,_ = image_val.shape
# Define configuration for Tesseract
config = r'--oem 3 --psm 6'

# Use image_to_data for word detection
detection_result = pytesseract.image_to_data(image_val, config=config, output_type=pytesseract.Output.DICT)

# Create a list to store the detected words
detected_words = []

# Loop through the detected words and draw rectangles
for i in range(len(detection_result['text'])):
    if int(detection_result['conf'][i]) > 0:
        x, y, w, h = detection_result['left'][i], detection_result['top'][i], detection_result['width'][i], detection_result['height'][i]
        cv2.rectangle(image_val, (x, y), (x + w, y + h), (50, 50, 255), 2)
        word = detection_result['text'][i]
        detected_words.append(word)

# Concatenate the detected words into a single string
detected_text = ' '.join(detected_words)

# Display the image with detected text below
cv2.putText(image_val, detected_text, (10, himage_val + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imshow('image_val', image_val)
cv2.waitKey(0)