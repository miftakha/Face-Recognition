# Face Recognition-Based Attendance System

## Overview
This project is a prototype of a face recognition system developed to improve attendance efficiency and reduce fraudulent attendance in a university environment.

The system performs real-time face detection and recognition using computer vision techniques. It identifies individuals based on facial features extracted from images and matches them against stored data.

---

## Objectives
- Improve attendance efficiency
- Reduce fraudulent attendance (e.g., proxy attendance)
- Explore the implementation of face recognition in academic environments

---

## Features
- Real-time face detection using webcam
- Face recognition using pretrained deep learning model
- Automatic labeling of detected faces
- Unknown face detection

---

## How It Works
1. Capture video input from webcam
2. Detect face locations in each frame
3. Extract facial features (128-dimension encoding)
4. Compare extracted features with stored encodings
5. Identify the closest match using distance threshold (≤ 0.6)
6. Display bounding box and name label on detected face
7. Mark unknown faces if no match is found

---

## Tech Stack
- Python
- OpenCV
- face_recognition (CNN-based pretrained model)
- NumPy
- Haar Cascade (face detection)

---

## Dataset
- Custom dataset using facial images
- Total subjects: 2 individuals
- Images are encoded into 128-dimensional feature vectors

---

## Installation & Usage
### 1. Clone the repository
```bash
git clone https://github.com/miftakha/Face-Recognition.git
cd Face-Recognition
```
### 2. Install dependencies
```bash
pip install opencv-python face-recognition numpy
```
### 3. Run the system
```bash
python main.py
```
---

## Results
- Successfully detects and recognizes registered faces
- Able to label unknown faces correctly
- Works well under ideal lighting conditions
- Achieved approximately 75% functional accuracy as a prototype
- Fast real-time performance

---

## Limitations
- Performance drops in low-light conditions
- Limited dataset (only 2 individuals)
- No database integration
- No user interface (CLI-based execution only)

---

## Future Improvements
- Improve robustness under varying lighting conditions
- Expand dataset for better accuracy
- Integrate with database system
- Develop user interface (web or desktop)
- Enhance model evaluation with performance metrics

## Author
Miftahul Ahmadil Khair
Computer Vision Developer and Team Lead

---

## Reference
This project is based on a research study on face recognition-based attendance systems using deep learning and CNN models.
