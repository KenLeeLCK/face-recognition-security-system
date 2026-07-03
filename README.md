# Security-Oriented Face Recognition System

A security-oriented open-set face recognition system based on computer vision.  
The system supports face database construction, face recognition, stranger rejection, threshold-based evaluation, FastAPI backend services, and a React frontend for interactive use.

The project is designed for security scenarios where the most critical objective is to reduce **false acceptance**: unknown people should not be incorrectly accepted as registered identities. Therefore, the evaluation pipeline focuses on metrics such as **FAR**, **FRR**, **Recall/TAR**, **Precision**, **F1-score**, **Open-set Accuracy**, and **ROC curves**.

---

## 1. Features

- Face detection and alignment using **MTCNN**
- Face embedding extraction using a deep face recognition model
- Structured multi-template face database stored in `.npz` format
- Open-set recognition with threshold-based stranger rejection
- Weighted multi-template matching using Top-1 and Top-K similarity scores
- Template quality score recording for database analysis and future quality-aware matching
- Identity registration, listing, and removal through backend APIs
- Evaluation pipeline for threshold scanning and FAR-constrained threshold selection
- FastAPI backend and React frontend interaction workflow
- Webcam-based recognition support in the frontend

---

## 2. Project Structure

```text
.
├── face_recognition_system.py       # Core face recognition system
├── evaluate_system.py               # Evaluation pipeline for metrics and threshold selection
├── face_api.py                      # FastAPI backend service
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
│
├── facerecg_react/                  # React frontend
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── App.js
│       └── App.css
│
├── dataset/                         # Local dataset folder, not uploaded to GitHub
│   ├── train/                       # Images used to build the face database
│   └── test/                        # Images used for recognition and evaluation
│
├── results/                         # Evaluation outputs
│   ├── evaluate_summary.json
│   ├── evaluate_metrics_summary.csv
│   ├── evaluate_far_constraints.csv
│   ├── evaluate_threshold_scan.csv
│   ├── ROC_linear.png
│   └── ROC_log.png
│
└── face_database.npz                # Generated face database, not uploaded to GitHub
```

The actual dataset, generated database files, and private face images are not included in this repository for privacy and storage reasons.

---

## 3. Dataset Format

### Training Dataset

The training dataset is used to build the face database. Each known identity should have a separate folder:

```text
dataset/
└── train/
    ├── person_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── image3.jpg
    │
    ├── person_2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── image3.jpg
```

Each subfolder name is treated as the identity name.

### Testing Dataset

The testing dataset is used by the evaluation script. It should contain both known identities and unknown persons:

```text
dataset/
└── test/
    ├── known/
    │   ├── person_1/
    │   │   ├── test1.jpg
    │   │   └── test2.jpg
    │   └── person_2/
    │       ├── test1.jpg
    │       └── test2.jpg
    │
    └── unknown/
        ├── unknown_1.jpg
        ├── unknown_2.jpg
        └── unknown_3.jpg
```

The `known/` folder should contain identities that exist in the training database.  
The `unknown/` folder should contain people who are not registered in the database.

---

## 4. Core System

### `face_recognition_system.py`

This file contains the core face recognition logic.

Main responsibilities:

- Detect faces using MTCNN
- Crop and align detected face regions
- Extract face embeddings
- Build a structured multi-template face database
- Recognize an input face image
- Reject unknown people based on similarity threshold
- Add, update, and remove identities in the database
- Load both new structured databases and compatible legacy databases

### Structured Database Design

The generated `.npz` database stores more than just identity labels and embeddings. It may include:

```text
identity_names
templates
template_labels
template_quality_scores
template_source_images
global_threshold
embedding_dim
model_name
database_version
```

Compared with a single averaged embedding per identity, the structured multi-template database preserves multiple templates for each identity. This helps retain variations caused by pose, lighting, expression, and image quality.

### Matching Strategy

Recognition uses weighted multi-template matching. Instead of relying only on a single template or one averaged embedding, the system combines:

```text
final_score = alpha * top_1_similarity + (1 - alpha) * top_k_average_similarity
```

This makes the recognition decision more stable by considering both the best-matching template and the overall support from the top matching templates of the same identity.

---

## 5. Evaluation System

### `evaluate_system.py`

This script evaluates the system under open-set recognition settings.

Main responsibilities:

- Score known and unknown test images
- Scan thresholds over a configurable range
- Calculate security-oriented metrics
- Select thresholds under FAR constraints
- Generate JSON and CSV summaries
- Generate linear-scale and log-scale ROC curves
- Record sample-level predictions and error cases

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| FAR | False Acceptance Rate. The proportion of unknown people incorrectly accepted as known identities. |
| FRR | False Rejection Rate. The proportion of known people incorrectly rejected as unknown. |
| Recall / TAR | True Acceptance Rate. The proportion of known people correctly recognized. |
| Precision | The proportion of accepted predictions that are correct. |
| F1-score | Harmonic mean of precision and recall. |
| Open-set Accuracy | Overall accuracy considering both known recognition and unknown rejection. |

For security-oriented scenarios, **FAR is the most important metric** because falsely accepting an unknown person is more serious than rejecting a known person.

### ROC Curve

The ROC curve is used to analyze the trade-off between the false acceptance rate and the true acceptance rate under different recognition thresholds.

In this project, the x-axis represents FAR, which measures how often unknown persons are incorrectly accepted as known identities. The y-axis represents TAR / Recall, which measures how often known persons are correctly recognized.

Since this system is designed for security-oriented scenarios, the ROC curve is mainly used to select a threshold that keeps FAR low while maintaining an acceptable recognition rate for known identities. In other words, the curve helps determine a suitable threshold instead of manually choosing one.

---

## 6. Backend API

### `face_api.py`

The backend is implemented with FastAPI.

Main responsibilities:

- Provide API endpoints for the React frontend
- Build the face database
- Recognize uploaded images
- Register new face images
- List registered identities
- Remove identities from the database
- Return database metadata and system metrics

### Common API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check whether the backend is running |
| `/metrics` | GET | Return database and system metadata |
| `/build_database` | POST | Build a face database from the training dataset |
| `/register` | POST | Register a face image and rebuild/update the database workflow |
| `/recognize` | POST | Recognize an uploaded face image |
| `/identities` | GET | List identities and template counts |
| `/remove_identity` | POST | Remove an identity from the current database |

The database path and dataset path used by the API are backend-side paths. For example, if the backend runs on Windows, the path should be a Windows path. If the backend runs on macOS or Linux, the path should be a macOS/Linux path.

---

## 7. Frontend

The frontend is built with React.

Main functions:

- Upload an image for recognition
- Capture an image from webcam for recognition
- Register a new face image
- Build a database from the server-side dataset folder
- List registered identities
- Remove selected identities
- Display basic system metrics

The frontend communicates with the FastAPI backend. By default, it connects to:

```text
http://127.0.0.1:8000
```

If needed, the backend address can be configured with:

```text
REACT_APP_API_URL
```

---

## 8. Installation

### Step 1: Create Python Environment

```bash
python -m venv venv
```

On macOS / Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### Step 2: Install Backend Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install the main dependencies manually:

```bash
pip install numpy opencv-python torch torchvision facenet-pytorch pillow scikit-learn matplotlib pandas fastapi uvicorn python-multipart
```

### Step 3: Install Frontend Dependencies

```bash
cd facerecg_react
npm install
```

---

## 9. Running the Project

### Build Face Database

```bash
python face_recognition_system.py build --dataset_dir ./dataset/train --output face_database.npz
```

### Recognize One Image

```bash
python face_recognition_system.py recognize --image_path ./test.jpg --database face_database.npz
```

Depending on the recognition result, the system will return either a known identity or reject the image as a stranger.

### Run Evaluation

```bash
python evaluate_system.py --device mps
```

Use `--device cpu` if MPS or CUDA is unavailable:

```bash
python evaluate_system.py --device cpu
```

Typical evaluation outputs include:

```text
evaluate_summary.json
evaluate_metrics_summary.csv
evaluate_far_constraints.csv
evaluate_threshold_scan.csv
ROC_linear.png
ROC_log.png
```

### Start Backend

```bash
python face_api.py
```

Or:

```bash
uvicorn face_api:app --reload --host 127.0.0.1 --port 8000
```

Backend address:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### Start Frontend

Open a new terminal:

```bash
cd facerecg_react
npm start
```

Frontend address:

```text
http://localhost:3000
```

---

## 10. Suggested Demo Workflow

For a complete local demonstration, use the following order:

```bash
# 1. Install backend dependencies
pip install -r requirements.txt

# 2. Build the face database
python face_recognition_system.py build --dataset_dir ./dataset/train --output face_database.npz

# 3. Evaluate the system and select a threshold
python evaluate_system.py --device mps

# 4. Start the backend
python face_api.py

# 5. Start the frontend
cd facerecg_react
npm install
npm start
```

Then open:

```text
http://localhost:3000
```

---

## 11. Notes on Threshold Selection

The recognition threshold controls the trade-off between false acceptance and false rejection.

- A higher threshold usually reduces FAR but increases FRR.
- A lower threshold usually improves recall but increases the risk of accepting unknown people.
- In this project, the threshold is selected mainly according to FAR constraints because the system is designed for security scenarios.

For a strict security scenario, the preferred threshold should keep FAR very low, even if some known users are rejected.

---

## 12. Privacy and Repository Notes

Face datasets and generated databases may contain private biometric information. For privacy and storage reasons, the following files and folders should not be uploaded to GitHub:

```text
dataset/
face_database.npz
*.npz
results/
__pycache__/
node_modules/
```

Recommended `.gitignore` entries:

```gitignore
# Python
__pycache__/
*.pyc
venv/
.env

# Face data and generated outputs
dataset/
*.npz
results/
*.json
*.csv
*.png

# Frontend
facerecg_react/node_modules/
facerecg_react/build/
```

---

## 13. Project Summary

This project implements a complete security-oriented face recognition workflow. It includes face detection, embedding extraction, structured database construction, open-set recognition, threshold-based stranger rejection, backend API services, frontend interaction, and FAR-oriented evaluation.

The main design priority is to reduce the possibility that unknown people are falsely accepted as known identities while maintaining acceptable recognition performance for registered identities.

## 14. Team Contributions

| Member | Contributions |
|---|---|
| Chan Kin Lei | Initial algorithm design and pipeline implementation; evaluation metric design and implementation; algorithm and system optimization; backend API design and implementation; frontend interface optimization. |
| Xin Cong | Evaluation metric design and implementation; algorithm and system optimization. |
| Pascoal de Deus Soares | Frontend interface design and implementation. |