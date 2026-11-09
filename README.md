# 🧠 DigitVision AI: Handwritten Digit Recognizer

An interactive web app that recognizes handwritten digits (0 to 9) in real time. Draw a digit on the canvas, and a Convolutional Neural Network (CNN) trained on the MNIST dataset predicts it, shows its confidence, and even reveals what the network "sees" inside.

**🔗 Live demo:** [Click-To-Open-APP](https://digit-vision-ai.streamlit.app/)

<!-- Add a screenshot or GIF of the app here -->
<img width="1901" height="793" alt="image" src="https://github.com/user-attachments/assets/42dd8d64-f1f9-4731-b5b3-4b775a14ae59" />

<!-- ![DigitVision AI demo](demo.gif) -->

---

## ✨ Features

- **Draw and predict:** draw any digit on a canvas and get an instant prediction.
- **Confidence and accuracy:** see the model's confidence for your drawing, plus its overall test accuracy (**98.79%**).
- **Top 3 guesses and full probability chart** for all ten digits.
- **"What the CNN sees":** visualizes the feature maps from both convolutional layers, so you can see edges and shapes the network detects.
- **Challenge Mode:** a small game where you draw a requested digit and try to build the longest streak.
- **How it Works tab:** a visual pipeline and architecture diagram explaining every step, from drawing to prediction.
- **Prediction history, adjustable pen size and confidence threshold.**
- **Dark, responsive UI** built with custom CSS.

## 🛠️ How it works

```
Draw  ->  Crop  ->  Resize to 20x20  ->  Center in 28x28  ->  CNN  ->  Softmax  ->  Prediction
```

1. **Draw:** the user draws a digit on a 300×300 canvas.
2. **Preprocess:** the drawing is cropped to its bounding box, scaled to fit 20×20 (keeping the aspect ratio), placed in a 28×28 image, and shifted so its center of mass is centered. This mimics how MNIST images were prepared, which noticeably improves accuracy on real drawings.
3. **Predict:** the CNN outputs 10 probabilities via softmax. The highest one is the prediction.
4. **Explain:** feature maps from the Conv layers are extracted and displayed.

## 🏗️ Model architecture

| Layer | Output shape | Parameters |
|---|---|---|
| Conv2D (64 filters, 3×3, ReLU) | 26×26×64 | 640 |
| MaxPooling2D (2×2) | 13×13×64 | 0 |
| Conv2D (32 filters, 3×3, ReLU) | 11×11×32 | 18,464 |
| MaxPooling2D (2×2) | 5×5×32 | 0 |
| Flatten | 800 | 0 |
| Dense (128, ReLU) | 128 | 102,528 |
| Dense (32, ReLU) | 32 | 4,128 |
| Dense (10, Softmax) | 10 | 330 |

**Total trainable parameters:** 126,090

## 📊 Training and results

| Setting | Value |
|---|---|
| Dataset | MNIST (60,000 train / 10,000 test) |
| Preprocessing | Pixel values normalized to 0 to 1 |
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Epochs / batch size | 5 / 50 |
| Validation split | 20% |
| Final validation accuracy | 98.70% |
| **Test accuracy** | **98.79%** |

## 🧰 Tech stack

- **Python 3.12**
- **TensorFlow / Keras** for building and training the CNN
- **Streamlit** for the web app
- **streamlit-drawable-canvas** for the drawing canvas
- **NumPy, Pandas, Pillow** for data handling and image preprocessing
- **HTML/CSS** for the custom UI
- **Google Colab** for model training
- **Streamlit Community Cloud + GitHub** for deployment

## 📁 Project structure

```
digitvision-ai/
├── app.py                 # Streamlit application
├── mnist_cnn.keras        # Trained CNN model
├── metrics.json           # Saved test accuracy
├── requirements.txt       # Python dependencies
└── .streamlit/
    └── config.toml        # Dark theme configuration
```

## 🚀 Run locally

```bash
# 1. Clone the repository
git clone https://github.com/MOHAMMED-ZAKIR-AHMED/digitvision-ai.git
cd digitvision-ai

# 2. Create and activate a virtual environment (Python 3.11 or 3.12)
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows
# venv\Scripts\activate           # Windows CMD / PowerShell
# source venv/bin/activate        # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## ⚠️ Limitations

- The model is trained on MNIST, so very small, tilted, or unusual handwriting styles may reduce confidence.
- Similar-looking digits (for example 1 and 7) can occasionally be confused.

## 🔮 Future improvements

- Train with data augmentation (rotation, shift, zoom) for better real-world handwriting.
- Add an image upload option.
- Compare multiple models (for example MLP vs CNN) inside the app.

## 👤 Author

**Zakir Ahmed**
AI / ML Engineer

[LinkedIn](https://www.linkedin.com/in/zakir00) · [GitHub](https://github.com/MOHAMMED-ZAKIR-AHMED) · [Portfolio](https://zenodev-portfolio.vercel.app/)

If you like this project, consider giving it a ⭐.
