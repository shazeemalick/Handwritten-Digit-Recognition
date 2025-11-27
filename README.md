🧠 Handwritten Digit Recognition using ANN

A simple and effective Artificial Neural Network (ANN) model trained on the MNIST dataset to recognize handwritten digits from 0–9.
This project includes a trained .h5 model, prediction script, and the full workflow for digit classification.



🚀 Project Overview

This project demonstrates a neural network model trained on the popular MNIST dataset to classify handwritten digits.
The model uses a multi-layer ANN with Dense layers and achieves high accuracy on the test set.

The repository includes:

A trained .h5 ANN model

A prediction script (app.py)

Requirements file


📂 Project Structure
Handwritten-Digit-Recognition-ANN/
│── app.py                    # Predicts digit from an image
│── mnist_ann_model.h5        # Trained ANN model
│── requirements.txt          # Dependencies
│── README.md                 # Documentation

🧠 Model Details
Dataset:

MNIST dataset (60,000 training images + 10,000 test images)

Model Architecture:

Input: Flattened 28 × 28 grayscale images

Hidden Layers: Dense layers with ReLU activation

Output Layer: Dense (10 units) with Softmax

Training:

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Metrics: Accuracy

📊 Model Performance

✔ High accuracy on test data

✔ Performs well on unseen handwritten digits

✔ Lightweight ANN architecture

(You may add exact accuracy here if available.)

▶️ Running the Prediction App
1. Install Required Packages
pip install -r requirements.txt

2. Run the App
python app.py


The script will:

Load the ANN model

Accept an input digit image

Preprocess it

Predict the digit

📦 Dataset Information

This project uses the MNIST dataset:

28x28 grayscale images

10 classes (0–9)

Automatically downloaded through Keras

No dataset file is required inside the repo.

👨‍💻 Author

Shahzaib Asif
