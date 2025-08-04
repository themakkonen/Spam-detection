# Spam Detection

A machine learning project for detecting spam messages (SMS/Email).  
This repository contains:
- A Jupyter Notebook for training and evaluating the spam detection model.
- A Flask-based web application to interact with the trained model.

---

## Features
- Preprocesses SMS/email text data.
- Trains a machine learning classifier (e.g., Naive Bayes/Logistic Regression).
- Simple web interface for real-time spam classification.
- Pre-trained model included (`model.pkl`) for direct usage.

---

## Project Structure

---│
├── app/
│ ├── app.py # Flask server script
│ ├── model.pkl # Trained spam detection model
│ ├── spam.csv # Dataset used for training/testing
│ └── templates/
│ └── index.html # Web UI for input and results
│
├── spam detection.ipynb # Notebook for data analysis and training (PDF in repo)

## Installation

1. **Clone the repository**
```
python app.py

git clone https://github.com/<your-username>/spam-detection.git
cd spam-detection/app
