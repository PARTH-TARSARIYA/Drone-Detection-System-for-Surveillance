# 🚁 Drone Detection System

## 📌 Overview

This project is a **multimodal Drone Detection System** that uses deep learning to classify whether a drone is present in a video.

The system processes both **video frames and audio signals**, extracts visual and acoustic features, and uses a trained model to detect drones with confidence scores.

This solution can be applied in **surveillance systems, security monitoring, and restricted airspace management**, where accurate drone detection is critical.

---

## 🎯 Features

* Detects drones in video files
* Provides confidence score for predictions
* Supports user-uploaded videos
* Real-time frame-based analysis
* Streamlit-based interactive UI
* Uses both **video and audio features** for improved accuracy
* Detects drone-specific **acoustic signatures**

---

## 🛠️ Tech Stack

This project combines computer vision, audio processing, and deep learning:

* **pandas** – Data manipulation and preprocessing
* **numpy** – Numerical computations and array operations
* **moviepy** – Extracts audio and frames from video files
* **streamlit** – Interactive web application interface
* **joblib** – Model and encoder serialization
* **tensorflow** – Deep learning model development
* **scikit-learn** – Label encoding and evaluation metrics
* **opencv-python-headless** – Video frame extraction and image processing
* **librosa** – Audio feature extraction (e.g., MFCC)

---

## 📂 Dataset

This project uses a combination of publicly available datasets and additional media sources.

### 1. Primary Dataset

* Source: https://github.com/DroneDetectionThesis/Drone-detection-dataset
* Description: Contains labeled drone and non-drone video/image data used for initial training.

### 2. Additional Data (Custom Collection)

To improve model robustness, additional data was collected from:

* Pixabay: https://pixabay.com/
* Pexels: https://www.pexels.com/

Used for:

* Testing on real-world scenarios
* Enhancing dataset diversity

---

## ⚠️ Note

* The dataset is not included due to size constraints
* External media is subject to respective platform licenses
* Used strictly for educational and research purposes

---

## 📁 Project Structure

```
drone-detection/
│── data/
│── models/
│── app.py
│── utils/
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/your-username/drone-detection.git
```

2. Navigate to the project directory:

```
cd drone-detection
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:

```
streamlit run app.py
```

### Steps:

* Upload a video file
* The system extracts frames and audio
* Model processes both inputs
* Output displays:

  * **Drone Detected / Not Detected**
  * **Confidence Score**

---

## 🧠 Model Details

* **Model Type:** Multimodal Deep Learning Model
* **Inputs:**

  * Video Frames (visual features)
  * Audio Signals (acoustic features)
* **Output:** Binary classification (Drone / No Drone)

### 🔄 Processing Pipeline:

1. Extract frames using OpenCV
2. Extract audio using MoviePy
3. Generate:

   * Visual features (from frames)
   * Audio features (MFCC via Librosa)
4. Combine features and perform prediction

---

## 🔊 Why Audio Features?

Drones produce distinctive sound patterns due to their motors and propellers.

Audio helps when:

* Drone is far away
* Visibility is poor
* Object is partially hidden

This improves detection accuracy and robustness.

---

## 📊 Results

* Accuracy: 92%
* Precision: 90%
* Recall: 88%

The model performs well on both training data and real-world test samples.

---

## 📚 References

* Pixabay: https://pixabay.com/
* Pexels: https://www.pexels.com/

This project uses the Drone Detection Dataset provided by:

Svanström, F., Alonso-Fernandez, F., & Englund, C. (2021).  
*A Dataset for Multi-Sensor Drone Detection*.  
Data in Brief, 39, 107521.  
https://doi.org/10.1016/j.dib.2021.107521  

Additional related works:

- Svanström, F. (2020). *Drone Detection and Classification using Machine Learning and Sensor Fusion* (Master’s Thesis)

- Svanström, F., Englund, C., & Alonso-Fernandez, F. (2021).  
  *Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors*  
  Proceedings of ICPR 2020  
  https://doi.org/10.1109/ICPR48806.2021.9413241  

Dataset Source:  
https://github.com/DroneDetectionThesis/Drone-detection-dataset

---

## 🚀 Future Improvements

* Real-time drone detection using live camera feed
* Deploy model using cloud services
* Improve accuracy with larger datasets
* Implement advanced multimodal fusion techniques

---
