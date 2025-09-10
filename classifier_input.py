import tensorflow as tf
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps
import io
import matplotlib.pyplot as plt

# Load trained model
loaded_model = tf.keras.models.load_model("digit_classifier.h5")

def preprocess_canvas(canvas):
    # Save canvas content as PostScript in memory
    ps = canvas.postscript(colormode="color")
    img = Image.open(io.BytesIO(ps.encode("utf-8")))

    # Convert to grayscale
    img = img.convert("L")

    # Invert colors (MNIST convention = white digit on black background)
    img = ImageOps.invert(img)

    # Convert PIL to OpenCV format for processing
    img_np = np.array(img)

    # Threshold (binarize)
    _, img_bin = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY)

    # Find bounding box of the digit
    coords = cv2.findNonZero(img_bin)
    if coords is None:  # empty canvas
        return np.zeros((1, 28, 28), dtype="float32")

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the digit
    digit = img_bin[y:y+h, x:x+w]

    # Resize to 20x20 (keep edges sharp with NEAREST)
    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_NEAREST)

    # Pad to 28x28
    padded = np.pad(digit_resized, ((4, 4), (4, 4)), mode="constant", constant_values=0)

    # Normalize
    normalized = padded.astype("float32") / 255.0

    # Reshape for model input
    return normalized.reshape(1, 28, 28)

def draw(event):
    canvas.last_x, canvas.last_y = event.x, event.y

def draw_line(event):
    x, y = event.x, event.y
    # thicker pen for MNIST-like strokes
    canvas.create_line(canvas.last_x, canvas.last_y, x, y, fill='black', width=8)
    canvas.last_x, canvas.last_y = x, y
    coord_label.config(text=f"X: {event.x}, Y: {event.y}")

def clear_canvas():
    canvas.delete('all')
    result_label.config(text="Predicted Digit: -")

def submit_data():
    input_img = preprocess_canvas(canvas)

    # Debug: show what model sees
    plt.imshow(input_img[0], cmap="gray")
    plt.title("Model Input")
    plt.show()

    predictions = loaded_model.predict(input_img)
    predicted_class = np.argmax(predictions[0])
    result_label.config(text=f"Predicted Digit: {predicted_class}")

# Tkinter setup
root = tk.Tk()
root.title('Digit Classifier')
root.geometry('500x500')

coord_label = tk.Label(text='X: -, Y: -')
coord_label.pack()

canvas = tk.Canvas(root, width=200, height=200, background='white')
canvas.pack(pady=10)

clear_button = tk.Button(text='Clear', command=clear_canvas)
clear_button.pack()

submit_button = tk.Button(text='Submit', command=submit_data)
submit_button.pack()

result_label = tk.Label(text='Predicted Digit: -', font=("Arial", 16))
result_label.pack(pady=10)

canvas.bind("<Button-1>", draw)
canvas.bind("<B1-Motion>", draw_line)

root.mainloop()
