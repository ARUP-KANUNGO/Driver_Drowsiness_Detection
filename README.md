---

# 🚗 Driver Drowsiness Detection System

A real-time **Driver Drowsiness Detection System** built using **Computer Vision** and **Deep Learning** to enhance road safety by monitoring driver alertness through eye-state analysis.

---

## 📌 Overview

Driver fatigue is one of the leading causes of road accidents worldwide.
This project detects drowsiness by analyzing eye closure patterns in real-time and triggers an alert when fatigue is detected.

The system intelligently distinguishes between **normal blinking** and **prolonged eye closure**, ensuring reliable detection.

---

## 🎯 Key Features

✔ Real-time webcam-based detection
✔ Eye state classification (Open / Closed)
✔ Temporal logic to avoid false alerts
✔ Smooth and stable predictions
✔ Audible alert system for driver safety
✔ Lightweight and efficient model

---

## 🧠 Tech Stack

* **Programming Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Libraries:** NumPy

---

## 🏗️ System Workflow

```text
Webcam Input
     ↓
Face Detection (Haar Cascade)
     ↓
Eye Region Extraction
     ↓
Preprocessing (Resize + Normalize)
     ↓
CNN Model Prediction
     ↓
Temporal Smoothing Logic
     ↓
Drowsiness Detection
     ↓
Alert Trigger
```

---

## 📊 Model Details

* **Input Size:** 224 × 224 × 3
* **Architecture:** CNN (Transfer Learning-based)
* **Output:** Binary Classification (Open / Closed Eyes)
* **Trainable Parameters:** ~3K (Efficient fine-tuning)

---

## 📈 Results

* ✅ **Validation Accuracy:** ~97.2%
* ✅ Robust real-time performance
* ✅ Successfully detects prolonged eye closure
* ✅ Ignores normal blinking
* ✅ Reduced false alerts using smoothing buffer

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ARUP-KANUNGO/Driver_Drowsiness_Detection.git
cd Driver_Drowsiness_Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(If requirements.txt not available, install manually)*

```bash
pip install tensorflow opencv-python numpy
```

---

## ▶️ Usage

### Run Live Detection

```bash
python live_detection.py
```

### Train Model

```bash
python train_model.py
```

---

## 📸 Output

* Real-time face detection
* Eye state prediction displayed on screen
* Alert triggered when drowsiness detected

*(Add screenshots here if possible for better presentation)*

---

## ⚠️ Limitations

* Performance drops in low-light conditions
* Haar Cascade is less robust than modern detectors
* Requires good quality dataset for best results

---

## 🔮 Future Work

* Replace Haar Cascade with deep learning-based detector (e.g., MTCNN / YOLO)
* Use facial landmarks for improved accuracy
* Apply **domain adaptation** for real-world scenarios
* Implement **few-shot learning** for better generalization
* Optimize for mobile / embedded deployment

---

## 📂 Project Structure

```text
Driver_Drowsiness_Detection/
│── live_detection.py
│── train_model.py
│── utils.py
│── accuracy.png
│── README.md
│── .gitignore
```

---

## 📚 References

* Research papers on driver drowsiness detection
* TensorFlow & Keras documentation
* OpenCV official documentation

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub!

---

## 👨‍💻 Author

**Arup Kanungo**
GitHub: [https://github.com/ARUP-KANUNGO](https://github.com/ARUP-KANUNGO)

---
Just say 👍
