<h1 align="center">🅿️🚗 Parking Slot Detection using Image Processing</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python">
  <img src="https://img.shields.io/badge/OpenCV-Image_Processing-green?logo=opencv">
  <img src="https://img.shields.io/badge/Numpy-Matrix_Operations-yellow?logo=numpy">
</p>

<p align="center">
  🔍 Detects parking slot occupancy in an image using grayscale conversion, filtering, morphology, and contour analysis!
</p>

---

## 🌟 Overview

This project detects **empty** and **occupied** parking spaces in a static parking lot image using custom-built image processing techniques in Python. Instead of relying on deep learning, it uses low-level image filters, morphology, and contour-based classification for slot detection.

---

## 🎯 Features

✅ Manual grayscale conversion  
✅ Mean filter smoothing  
✅ Homomorphic filtering for contrast  
✅ Adaptive thresholding & morphological cleanup  
✅ Contour and bounding box slot detection  
✅ Slot classification (Green: Empty, Red: Occupied)  
✅ Backup mode using Hough lines or fixed grid  

---

## 📂 File Structure

parking-slot-detector/
├── detect_parking.py # Main script
├── test_image.jpg # Input parking lot image
├── results.png # Output image with slot markings
└── README.md # This file

yaml
Copy
Edit

---

## 🧑‍💻 How to Run the Script

### 🛠️ Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/parking-slot-detector.git
cd parking-slot-detector
🐍 Step 2: Set Up Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
📦 Step 3: Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
▶️ Step 4: Run the Script
bash
Copy
Edit
python detect_parking.py
Output:

Displays the image with green (empty) and red (occupied) boxes.

Saves the result as results.png.

📈 Detection Logic
Grayscale: Manually converts RGB to grayscale using 0.299R + 0.587G + 0.114B.

Filtering: Applies mean filter (smoothing) and homomorphic filtering (enhancement).

Morphology: Removes noise using opening/closing.

Contours: Detects and classifies rectangular slot-like shapes.

Classification: Average intensity + standard deviation determines occupancy.

🧪 Example Output
yaml
Copy
Edit
Total slots detected: 20
Empty slots: 13
Occupied slots: 7
Visual:

🟩 Green boxes = empty slots

🟥 Red boxes = occupied slots

Result saved as: results.png

📦 requirements.txt
txt
Copy
Edit
numpy
opencv-python
matplotlib
Pillow
📝 Notes
Works best with top-view images of a parking lot.

No ML models required — works using pure image operations.

You can easily switch in a new image by changing image_path.

❤️ Credits
NumPy

OpenCV

Matplotlib

Pillow (PIL)
