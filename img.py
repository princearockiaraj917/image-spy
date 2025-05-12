import tkinter as tk
from tkinter import filedialog
from PIL import Image
from transformers import pipeline
import torch


print("⏳ Loading model... please wait...")
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
print("✅ Model loaded successfully!")


root = tk.Tk()
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)


file_path = filedialog.askopenfilename(title="Choose an image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

if file_path:
    print("🖼️ Image selected! Analyzing...")
    image = Image.open(file_path)
    image.show()

    results = classifier(image)

    label = results[0]['label']
    confidence = round(results[0]['score'] * 100, 2)
    print(f"🤖 This image most likely contains: **{label}** ({confidence}% confidence)")
else:
    print("❌ No file selected.")


