# Sleep Detector Program  
### Real-Time Driver Drowsiness Detection System

A real-time computer vision system that detects driver drowsiness using facial landmark analysis and triggers an audible alert to help prevent fatigue-related accidents.

---

## Overview

Driver fatigue is a major cause of road accidents.  
**Drive Alarm** monitors the driver's eye state using a webcam and calculates the **Eye Aspect Ratio (EAR)** to determine drowsiness. If the eyes remain closed for a defined duration, an alarm is triggered.

This project demonstrates:

- Real-time computer vision
- Facial landmark detection (dlib)
- Geometric eye state analysis
- Alert-based safety system implementation

---

## Features

- Real-time face and eye detection  
- Eye Aspect Ratio (EAR) calculation  
- Frame-based drowsiness threshold detection  
- Audible alert system  
- Live EAR display on video feed  

---

## Tech Stack

- Python  
- OpenCV  
- dlib  
- NumPy  
- SciPy  
- Pygame  

---

## Project Structure

```
Drive_alarm/
│
├── project 2.py
├── project 3.py
├── requirements.txt
├── .gitignore
└── assets/
    ├── beep.mp3
    └── shape_predictor_68_face_landmarks.dat (not included in repo)
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/AdityaSharma-Git3207/Sleep-Detector-Program.git
cd Sleep-Detector-Program
```

---

### Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

> Note: `dlib` installation may require Visual Studio Build Tools on Windows.

---

## Required Model File

This project requires the pre-trained **dlib facial landmark predictor** model.

Due to its large size (~100MB), it is not included in this repository.

### Download Steps:

1. Download from:  
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

2. Extract the `.bz2` file

3. Place the extracted file inside:

```
assets/
```

Final structure should look like:

```
assets/shape_predictor_68_face_landmarks.dat
```

---

## Running the Application

```bash
python "project 2.py"
```

Press **`q`** to quit the application.

---

## How It Works

The system calculates the **Eye Aspect Ratio (EAR)** using facial landmarks.

### Detection Logic

- Eyes are considered closed when:

```
EAR < 0.25
```

- If eyes remain closed for:

```
20 consecutive frames
```

The alarm is triggered.

---

## Git Ignore Policy

The following files are intentionally excluded:

- `venv/` → Virtual environment directory  
- `assets/shape_predictor_68_face_landmarks.dat` → Large binary model file  

This keeps the repository lightweight and maintainable.

---

## Future Improvements

- Blink rate monitoring  
- Head pose estimation  
- Performance optimization  
- GUI interface  
- Mobile device integration  

---

## Author

Aditya Sharma  
Open Source Contributor | Computer Vision Enthusiast
