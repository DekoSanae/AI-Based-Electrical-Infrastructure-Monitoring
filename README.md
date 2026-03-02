⚡ AI-Based Electrical Infrastructure Monitoring

📌 Project Overview

This project aims to classify key electrical grid components from drone inspection images using deep learning and transfer learning techniques.

The model identifies the following components:

-Transformers
-Utility Poles
-Overhead Power Lines
-Fuses / Cutouts

This project simulates a real-world drone-based inspection scenario where automatic identification of infrastructure elements can support asset monitoring, fault detection, and maintenance planning.

🎯 Problem Statement

Electrical grid inspections increasingly rely on aerial imagery captured by drones. Manually analyzing these images is time-consuming and error-prone.

The goal of this project is to build a deep learning model capable of automatically classifying electrical components from inspection images to support scalable and automated infrastructure monitoring.

🗂 Dataset Description

400 images total

4 classes:

-Transformer
-Pole
-Line
-Fuse

100 images per class

The dataset was split into:

70% Training
15% Validation
15% Test

Images are organized in directory-based labeling format:

dataset/
│
├── train/
├── val/
└── test/

Each folder contains subdirectories for each class.

🧠 Methodology

1️⃣ Data Preprocessing

-Image resizing to 224x224
-Pixel normalization (rescaling)
-Data augmentation (rotation, zoom, shifts, horizontal flip)
-Stratified dataset split

2️⃣ Model Architecture

Transfer learning was applied using MobileNetV2 pretrained on ImageNet.

Why MobileNetV2?

-Lightweight architecture
-Efficient for small datasets
-Suitable for deployment in edge or drone environments

Model structure:

Pretrained MobileNetV2 (feature extractor)
GlobalAveragePooling layer
Dense layer (ReLU)
Dropout (regularization)
Final Dense layer (Softmax activation for 4 classes)

Loss function:

Categorical Crossentropy


Optimizer:

Adam

3️⃣ Training Strategy

-Frozen base model (initial training phase)
-EarlyStopping to prevent overfitting
-ReduceLROnPlateau for adaptive learning rate control
-Validation monitoring during training

📊 Evaluation Metrics

Accuracy
Confusion Matrix
Precision per class
Recall per class
F1-Score

Special attention was given to analyzing misclassifications between visually related classes such as:

-Lines vs Poles
-Transformers vs Poles

🔍 Key Insights

-Transfer learning significantly improved convergence speed.
-Data augmentation helped reduce overfitting.
-Misclassifications mainly occurred when contextual elements dominated the image.

The model performs well in controlled object-centered scenarios.

🚀 Future Improvements

Expand dataset size
Introduce multi-label classification for complex scenes
Implement object detection (YOLO / Faster R-CNN)
Deploy model in a simulated drone inspection pipeline


🛠 Tech Stack

Python
TensorFlow / Keras
MobileNetV2
NumPy
Matplotlib
Scikit-learn

👩‍💻 Author

Lizeth Castillo
Electrical Engineer | Data Science & AI Enthusiast

This project integrates power systems knowledge with computer vision techniques to address real-world infrastructure monitoring challenges.