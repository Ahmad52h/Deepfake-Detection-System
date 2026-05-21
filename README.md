# 🕵️‍♂️ Deepfake Video Detection System

An End-to-End AI system designed to detect manipulated and deepfake videos. Built as a CIS project, it analyzes both spatial and temporal features of video frames to achieve high accuracy.

## ✨ Features
* **Face Extraction:** Uses MTCNN to dynamically detect and crop faces from video frames.
* **Spatial Analysis:** Utilizes an EfficientNet (CNN) architecture to analyze pixel-level distortions.
* **Temporal Analysis:** Employs an LSTM (RNN) network to track sequence consistency and detect unnatural movements.
* **User Interface:** A simple, interactive web app built with Gradio.

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/Deepfake-Detection-System.git](https://github.com/YourUsername/Deepfake-Detection-System.git)
cd Deepfake-Detection-System
